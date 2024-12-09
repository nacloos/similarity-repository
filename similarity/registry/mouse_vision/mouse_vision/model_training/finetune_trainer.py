import os
import copy
import time

import torch
import torch.nn as nn
import numpy as np

from torchvision import transforms

import mouse_vision.loss_functions as lf
import mouse_vision.model_training.custom_heads as ch

from mouse_vision.model_training.trainer import Trainer
from mouse_vision.model_training.train_utils import reduce_metric
from mouse_vision.model_training.imagenet_datasets import ImageNetSupervised
from mouse_vision.model_training.training_dataloader_utils import get_dataloaders
from mouse_vision.model_training.trainer_transforms import TRAINER_TRANSFORMS
from mouse_vision.models.model_transforms import MODEL_TRANSFORMS
from mouse_vision.core.default_dirs import IMAGENET_DATA_DIR
from mouse_vision.model_training.train_utils import (
    AverageMeter,
    compute_accuracy,
    check_best_accuracy,
    save_checkpoint,
)


class FinetuneImageNetTrainer(Trainer):
    def __init__(self, config):
        super(FinetuneImageNetTrainer, self).__init__(config)

        # TODO: Remove when TPU finetuner is debugged
        assert not self.use_tpu

        # Load model backbone weights
        if self.config["model_checkpoint"] is None:
            print("No checkpoint provided, finetuning off random initialization")
        else:
            self._load_model_backbone_weights()

        assert hasattr(self, "use_tpu")
        assert hasattr(self, "device")
        assert hasattr(self, "loss_func")

        # We need this condition just in case we loaded results previously using
        # load_checkpoint(). See __init__ in trainer.py and load_checkpoint() in
        # this file. If we loaded results previously, we don't want to overwrite
        # results with an empty dictionary.
        if not hasattr(self, "results"):
            self.results = dict()
            self.results["losses"] = {"train": [], "val": []}
            self.results["accs_top1"] = {"train": [], "val": []}
            self.results["accs_top5"] = {"train": [], "val": []}

        # We need this condition since the best accuracy up to the current epoch
        # could have been loaded from a previous checkpoint, as above.
        if not hasattr(self, "best_acc"):
            self.best_acc = -1.0

        # Wrap the FC of the loss function with DistributedDataParallel if not
        # using TPU
        self.loss_func.to(self.device)
        if not self.use_tpu:
            assert hasattr(self, "gpu_ids")
            if not isinstance(self.loss_func.readout, nn.parallel.DistributedDataParallel):
                self.loss_func.readout = nn.parallel.DistributedDataParallel(
                    self.loss_func.readout, device_ids=self.gpu_ids
                )

        self.is_best = False

    def _load_model_backbone_weights(self):
        assert hasattr(self, "config")
        assert hasattr(self, "model")
        assert hasattr(self, "use_tpu")

        if self.use_tpu:
            from mouse_vision.model_training.gcloud_utils \
                import download_file_from_bucket

            # Download from gs bucket to local directory, where each ordinal
            # has its own copy of the same file so that they are each loading
            # the same data
            self.config["model_checkpoint"] = download_file_from_bucket(
                filename=self.config["model_checkpoint"],
                ordinal=self.rank,
                print_fn=self.print_fn,
                verbose=True
            )

        checkpoint_path = self.config["model_checkpoint"]
        if os.path.isfile(checkpoint_path):
            if self.use_tpu:
                cpt = torch.load(checkpoint_path)
                os.remove(checkpoint_path)
            else:
                assert hasattr(self, "gpu_ids")
                assert len(self.gpu_ids) == 1
                loc = f"cuda:{self.gpu_ids[0]}"
                cpt = torch.load(checkpoint_path, map_location=loc)
                self.print_fn(f"Loaded model checkpoint at '{checkpoint_path}'")
        else:
            raise ValueError(f"No checkpoint at '{checkpoint_path}'")

        # Make sure model state dict is provided
        assert "model_state_dict" in cpt.keys()

        # Load model state dict
        # TODO: Hack for now to deal with inconsistent weight names if you train
        # with DDP vs. without (e.g., on TPU)
        for k, v in cpt["model_state_dict"].items():
            if k.startswith("module."):
                self.model.load_state_dict(cpt["model_state_dict"])
                break
            else:
                self.model.module.load_state_dict(cpt["model_state_dict"])
                break
        self.print_fn("Loaded model backbone weights.")

    def initialize_loss_function(self):
        assert hasattr(self, "config")
        self.check_key("loss_params")
        self.check_key("readout_params")

        assert "class" in self.config["loss_params"].keys()
        assert "class" in self.config["readout_params"].keys()

        # Initialize readout head
        readout_class = self.config["readout_params"]["class"]
        readout_kwargs = copy.deepcopy(self.config["readout_params"])
        readout_kwargs.pop("class")
        readout = ch.__dict__[readout_class](**readout_kwargs)

        # Initialize finetune loss function
        loss_class = self.config["loss_params"]["class"]
        loss_func = lf.__dict__[loss_class](readout)

        return loss_func

    def initialize_dataloader(self):
        assert hasattr(self, "config")
        assert hasattr(self, "use_tpu")
        assert hasattr(self, "model_name")

        self.check_key("optimizer_params")
        self.check_key("dataloader_workers")

        assert "train_batch_size" in self.config["optimizer_params"].keys()
        assert "val_batch_size" in self.config["optimizer_params"].keys()

        params = dict()
        params["dataset_class"] = ImageNetSupervised
        params["image_dir"] = IMAGENET_DATA_DIR
        params["dataset"] = "imagenet"
        params["train_batch_size"] = self.config["optimizer_params"]["train_batch_size"]
        params["val_batch_size"] = self.config["optimizer_params"]["val_batch_size"]
        params["num_workers"] = self.config["dataloader_workers"]

        # Override default preprocessing if given in config file
        image_transforms_type = self.config.get("image_preprocess", None)
        if image_transforms_type is not None:
            assert image_transforms_type in TRAINER_TRANSFORMS.keys()
            train_transforms = TRAINER_TRANSFORMS[image_transforms_type]["train"]
            val_transforms = TRAINER_TRANSFORMS[image_transforms_type]["val"]
        else:
            train_transforms = MODEL_TRANSFORMS[self.model_name]["train"]
            val_transforms = MODEL_TRANSFORMS[self.model_name]["val"]

        print("Using train transforms:", train_transforms)
        print("Using validation transforms:", val_transforms)

        my_transforms = dict()
        my_transforms["train"] = transforms.Compose(train_transforms)
        my_transforms["val"] = transforms.Compose(val_transforms)

        train_loader, val_loader = get_dataloaders(
            params,
            my_transforms=my_transforms,
            device=self.device,
            rank=self.rank,
            world_size=self.world_size,
        )
        return train_loader, val_loader

    def initialize_optimizer(self):
        assert hasattr(self, "config")
        assert hasattr(self, "model")
        assert hasattr(self, "loss_func")
        self.check_key("optimizer_params")

        assert "initial_lr" in self.config["optimizer_params"].keys()
        assert "momentum" in self.config["optimizer_params"].keys()
        assert "weight_decay" in self.config["optimizer_params"].keys()

        # Make sure none of the model parameters require gradients
        for param in self.model.parameters():
            param.requires_grad = False

        # Since we are finetuning, we do not optimize the model's parameters
        optim = torch.optim.SGD(
            list(self.loss_func.trainable_parameters()),
            lr=self.config["optimizer_params"]["initial_lr"],
            momentum=self.config["optimizer_params"]["momentum"],
            nesterov=self.config["optimizer_params"].get("nesterov", False),
            weight_decay=self.config["optimizer_params"]["weight_decay"],
        )
        return optim

    def train_one_epoch(self):
        assert hasattr(self, "train_loader")
        assert hasattr(self, "use_tpu")
        assert hasattr(self, "model")
        assert hasattr(self, "loss_func")
        assert hasattr(self, "device")

        losses = AverageMeter("Loss", ":.4e")
        top1 = AverageMeter("Acc@1", ":6.2f")
        top5 = AverageMeter("Acc@5", ":6.2f")
        num_steps = len(self.train_loader)

        # Notably, since we are doing finetuning, the model should always be
        # in eval mode. The loss function's FC layer should be in train mode.
        self.set_model_to_eval()
        self.loss_func.train()
        end = time.time()
        for i, (data, labels) in enumerate(self.train_loader):
            if not self.use_tpu:
                # For TPU, we have already assigned it to the device
                data = data.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward propagation
            with torch.no_grad():
                # Forward propagate through the backbone without gradients
                model_outputs = self.model(data)
                # autoencoder variants
                if isinstance(model_outputs, dict):
                    assert "encoder_output" in model_outputs.keys()
                    model_outputs = model_outputs["encoder_output"]

            loss, predictions = self.loss_func(model_outputs, labels)

            # Backward propagation
            loss.backward()

            # Update parameters
            if self.use_tpu:
                import torch_xla.core.xla_model as xm

                xm.optimizer_step(self.optimizer)
            else:
                self.optimizer.step()

            # Metrics
            acc1, acc5 = compute_accuracy(
                output=predictions, target=labels, topk=(1, 5)
            )

            if self.use_tpu:
                rep_loss = loss.item()
                rep_acc1 = acc1.item()
                rep_acc5 = acc5.item()
            else:
                rep_loss = reduce_metric(loss, self.world_size).item()
                rep_acc1 = reduce_metric(acc1, self.world_size).item()
                rep_acc5 = reduce_metric(acc5, self.world_size).item()

            losses.update(rep_loss, data.size(0))
            top1.update(rep_acc1, data.size(0))
            top5.update(rep_acc5, data.size(0))

            total_time = time.time() - end
            end = time.time()

            #if (not self.use_tpu) and (self.rank == 0):
            #if (self.rank == 0):
            if self.use_tpu or self.rank == 0:
                print_str = (
                    f"[Epoch {self.current_epoch}; Step {i+1}/{num_steps}] "
                    f"Train Loss {rep_loss:.6f}; Train Accuracy: {rep_acc1:.6f}; "
                    f"Time {total_time:.3f} s"
                )
                self.print_fn(f"{print_str}")

        average_loss = losses.avg
        average_top1 = top1.avg
        average_top5 = top5.avg
        if self.use_tpu:
            # Average across TPU replicas
            average_loss = xm.mesh_reduce("train_average_loss", average_loss, np.mean)
            average_top1 = xm.mesh_reduce("train_average_top1", average_top1, np.mean)
            average_top5 = xm.mesh_reduce("train_average_top5", average_top5, np.mean)

        # Print train results over entire dataset
        msg_str = "[Epoch {}] Train Loss: {:.6f}; Train Accuracy: {:.6f}".format(
            self.current_epoch, average_loss, average_top1
        )
        if self.use_tpu:
            # xm.master_print only prints on the first TPU core
            self.print_fn(msg_str)
        elif self.rank == 0:
            # Print all reduce result on one gpu
            self.print_fn(msg_str)

        self.results["losses"]["train"].append(average_loss)
        self.results["accs_top1"]["train"].append(average_top1)
        self.results["accs_top5"]["train"].append(average_top5)

    def validate(self):
        assert hasattr(self, "val_loader")
        assert hasattr(self, "use_tpu")
        assert hasattr(self, "model")
        assert hasattr(self, "loss_func")
        assert hasattr(self, "device")

        losses = AverageMeter("Loss", ":.4e")
        top1 = AverageMeter("Acc@1", ":6.2f")
        top5 = AverageMeter("Acc@5", ":6.2f")
        num_steps = len(self.val_loader)

        self.set_model_to_eval()
        assert hasattr(self.loss_func, "readout")
        self.loss_func.eval()
        with torch.no_grad():
            for i, (data, labels) in enumerate(self.val_loader):
                if not self.use_tpu:
                    # For TPU, we have already assigned it to the device
                    data = data.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)

                model_outputs = self.model(data)
                # autoencoder variants
                if isinstance(model_outputs, dict):
                    assert "encoder_output" in model_outputs.keys()
                    model_outputs = model_outputs["encoder_output"]
                loss, predictions = self.loss_func(model_outputs, labels)

                # Metrics
                acc1, acc5 = compute_accuracy(
                    output=predictions, target=labels, topk=(1, 5)
                )

                if self.use_tpu:
                    rep_loss = loss.item()
                    rep_acc1 = acc1.item()
                    rep_acc5 = acc5.item()
                else:
                    rep_loss = reduce_metric(loss, self.world_size).item()
                    rep_acc1 = reduce_metric(acc1, self.world_size).item()
                    rep_acc5 = reduce_metric(acc5, self.world_size).item()

                losses.update(rep_loss, data.size(0))
                top1.update(rep_acc1, data.size(0))
                top5.update(rep_acc5, data.size(0))

                if (not self.use_tpu) and (self.rank == 0):
                    print_str = (
                        f"[Epoch {self.current_epoch}; Step {i+1}/{num_steps}] "
                        f"Val Loss {rep_loss:.6f}; Val Accuracy: {rep_acc1:.6f}"
                    )
                    self.print_fn(f"{print_str}")

        average_loss = losses.avg
        average_top1 = top1.avg
        average_top5 = top5.avg
        if self.use_tpu:
            # Average across TPU replicas
            import torch_xla.core.xla_model as xm

            average_loss = xm.mesh_reduce("val_average_loss", average_loss, np.mean)
            average_top1 = xm.mesh_reduce("val_average_top1", average_top1, np.mean)
            average_top5 = xm.mesh_reduce("val_average_top5", average_top5, np.mean)

        # Print val results over entire dataset
        msg_str = "[Epoch {}] Val Loss: {:.6f}; Val Accuracy: {:.6f}".format(
            self.current_epoch, average_loss, average_top1
        )
        if self.use_tpu:
            # xm.master_print only prints on the first TPU core
            self.print_fn(msg_str)
        elif self.rank == 0:
            # Print all reduce result on one gpu
            self.print_fn(msg_str)

        self.results["losses"]["val"].append(average_loss)
        self.results["accs_top1"]["val"].append(average_top1)
        self.results["accs_top5"]["val"].append(average_top5)

        # Check if current top-1 accuracy is best
        self.best_acc, self.is_best = check_best_accuracy(average_top1, self.best_acc)

    def save_checkpoint(self):
        assert hasattr(self, "current_epoch")
        assert hasattr(self, "model")
        assert hasattr(self, "loss_func")
        assert hasattr(self, "optimizer")
        assert hasattr(self, "results")
        assert hasattr(self, "save_dir")
        assert hasattr(self, "use_tpu")
        assert hasattr(self, "config")
        assert hasattr(self.loss_func, "readout")
        assert "save_freq" in self.config.keys()

        curr_state = {
            "epoch": self.current_epoch,
            "loss_readout_state_dict": self.loss_func.readout.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "results": self.results,
            "curr_best_acc": self.best_acc,
        }

        # Every "save_freq" epochs, save into new checkpoint file and relevant keys
        # to database. Overwrite existing checkpoint otherwise.
        save_epoch = None
        if (self.current_epoch % self.config["save_freq"] == 0) or (
            self.current_epoch + 1 == self.config["num_epochs"]
        ):
            save_epoch = self.current_epoch
            self._save_to_db(
                curr_state=curr_state, save_keys=["epoch", "results", "curr_best_acc"]
            )

        save_checkpoint(
            state=curr_state,
            save_dir=self.save_dir,
            is_best=self.is_best,
            save_epoch=save_epoch,
            rank=self.rank,
            tpu=self.use_tpu,
        )

    def load_checkpoint(self):
        assert hasattr(self, "model")
        assert hasattr(self, "optimizer")
        assert hasattr(self, "config")
        assert hasattr(self, "use_tpu")
        assert hasattr(self, "loss_func")
        self.check_key("resume_checkpoint")

        checkpoint_path = self.config["resume_checkpoint"]
        if os.path.isfile(checkpoint_path):
            if self.use_tpu:
                cpt = torch.load(checkpoint_path)
            else:
                assert len(self.gpu_ids) == 1
                loc = f"cuda:{self.gpu_ids[0]}"
                cpt = torch.load(checkpoint_path, map_location=loc)
            self.print_fn(f"Loaded checkpoint at '{checkpoint_path}'")
        else:
            raise ValueError(f"No checkpoint at '{checkpoint_path}'")

        # Make sure keys are in the checkpoint
        assert "epoch" in cpt.keys()
        assert "results" in cpt.keys()
        assert "loss_readout_state_dict" in cpt.keys()
        assert "optimizer" in cpt.keys()
        assert "curr_best_acc" in cpt.keys()

        # Load current epoch, +1 since we stored the last completed epoch
        self.current_epoch = cpt["epoch"] + 1

        # Load results
        assert not hasattr(self, "results")
        self.results = cpt["results"]

        # Load loss function's readout state dict
        assert hasattr(self.loss_func, "readout")
        self.loss_func.to(self.device)
        if not self.use_tpu:
            assert hasattr(self, "gpu_ids")
            self.loss_func.readout = nn.parallel.DistributedDataParallel(
                self.loss_func.readout, device_ids=self.gpu_ids
            )
        self.loss_func.readout.load_state_dict(cpt["loss_readout_state_dict"])

        # Load optimizer state dict
        self.optimizer.load_state_dict(cpt["optimizer"])

        # Load the current best accuracy
        self.best_acc = cpt["curr_best_acc"]


