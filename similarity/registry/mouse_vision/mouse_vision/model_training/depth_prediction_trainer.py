import os, copy

import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms

import mouse_vision.loss_functions as lf

from mouse_vision.core.default_dirs import PBRNET_DATA_DIR
from mouse_vision.models.model_transforms import MODEL_TRANSFORMS
from mouse_vision.model_training.train_utils import reduce_metric
from mouse_vision.model_training.other_datasets import PBRNetDepth
from mouse_vision.model_training.trainer import Trainer
from mouse_vision.model_training.training_dataloader_utils import get_dataloaders
from mouse_vision.model_training.train_utils import (
    AverageMeter,
    compute_accuracy,
    check_best_loss,
    save_checkpoint,
)


class DepthPredictionTrainer(Trainer):
    def __init__(self, config):
        super(DepthPredictionTrainer, self).__init__(config)

        # Load model backbone weights
        if self.config["model_checkpoint"] is None:
            print("No checkpoint provided, depth prediction training off random initialization")
        else:
            if self.config["resume_checkpoint"] is not None:
                print("Resume checkpoint was found, not loading pretrained ImageNet weights.")
            else:
                print("Loading pretrained ImageNet weights.")
                self._load_model_backbone_weights()

        # We need this condition just in case we loaded results previously using
        # load_checkpoint(). See __init__ in trainer.py and load_checkpoint() in
        # this file. If we loaded results previously, we don't want to overwrite
        # results with an empty dictionary.
        if not hasattr(self, "results"):
            self.results = dict()
            self.results["losses"] = {"train": [], "val": []}

        # We need this condition since the best (lowest) loss up to the current epoch
        # could have been loaded from a previous checkpoint, as above.
        if not hasattr(self, "best_loss"):
            self.best_loss = np.inf

        self.is_best = False

        # Place the loss function parameters onto multi-gpu
        self.loss_func = self.loss_func.to(self.device)
        assert hasattr(self, "gpu_ids")
        assert hasattr(self.loss_func, "decoder")
        if not isinstance(self.loss_func.decoder, nn.parallel.DistributedDataParallel):
            self.loss_func.decoder = nn.parallel.DistributedDataParallel(
                self.loss_func.decoder, device_ids=self.gpu_ids, broadcast_buffers=False
            )

    def _load_model_backbone_weights(self):
        assert hasattr(self, "config")
        assert hasattr(self, "model")
        assert hasattr(self, "use_tpu")
        assert not self.use_tpu

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
        assert "state_dict" in cpt.keys()

        # Load model state dict
        for k, v in cpt["state_dict"].items():
            if k.startswith("module."):
                self.model.load_state_dict(cpt["state_dict"], strict=False)
                break
            else:
                self.model.module.load_state_dict(cpt["state_dict"], strict=False)
                break
        self.print_fn("Loaded model backbone weights.")

    def initialize_loss_function(self):
        assert hasattr(self, "config")
        self.check_key("loss_params")

        assert "class" in self.config["loss_params"].keys()

        loss_class = self.config["loss_params"]["class"]
        loss_func = lf.__dict__[loss_class]()
        return loss_func

    def initialize_optimizer(self):
        assert hasattr(self, "config")
        assert hasattr(self, "model")
        self.check_key("optimizer_params")

        assert "initial_lr" in self.config["optimizer_params"].keys()
        assert "weight_decay" in self.config["optimizer_params"].keys()

        if self.config["optimizer"] == "Adam":
            print("Using Adam optimizer")
            optim = torch.optim.Adam(
                list(self.model.parameters()) + list(self.loss_func.trainable_parameters()),
                lr=self.config["optimizer_params"]["initial_lr"],
                weight_decay=self.config["optimizer_params"]["weight_decay"],
            )
        else:
            print("Using default SGD optimizer")
            assert "momentum" in self.config["optimizer_params"].keys()
            assert self.config["optimizer"] == "SGD"
            optim = torch.optim.SGD(
                list(self.model.parameters()) + list(self.loss_func.trainable_parameters()),
                lr=self.config["optimizer_params"]["initial_lr"],
                momentum=self.config["optimizer_params"]["momentum"],
                weight_decay=self.config["optimizer_params"]["weight_decay"],
            )

        return optim

    def initialize_dataloader(self):
        assert hasattr(self, "config")
        assert hasattr(self, "use_tpu")
        assert hasattr(self, "model_name")

        self.check_key("optimizer_params")
        self.check_key("dataloader_workers")

        assert "train_batch_size" in self.config["optimizer_params"].keys()
        assert "val_batch_size" in self.config["optimizer_params"].keys()

        params = dict()
        params["dataset_class"] = PBRNetDepth
        params["image_dir"] = PBRNET_DATA_DIR
        params["dataset"] = "pbrnet"
        params["train_batch_size"] = self.config["optimizer_params"]["train_batch_size"]
        params["val_batch_size"] = self.config["optimizer_params"]["val_batch_size"]
        params["num_workers"] = self.config["dataloader_workers"]

        my_transforms = dict()
        my_transforms["train"] = transforms.Compose(
            MODEL_TRANSFORMS[self.model_name]["train"]
        )
        my_transforms["val"] = transforms.Compose(
            MODEL_TRANSFORMS[self.model_name]["val"]
        )

        train_loader, val_loader = get_dataloaders(
            params,
            my_transforms=my_transforms,
            device=self.device,
            rank=self.rank,
            world_size=self.world_size,
        )
        return train_loader, val_loader

    def train_one_epoch(self):
        assert hasattr(self, "train_loader")
        assert hasattr(self, "use_tpu")
        assert hasattr(self, "model")
        assert hasattr(self, "loss_func")
        assert hasattr(self, "device")

        losses = AverageMeter("Loss", ":.4e")
        num_steps = len(self.train_loader)

        self.set_model_to_train()
        self.loss_func.train()
        for i, (data, target) in enumerate(self.train_loader):
            if not self.use_tpu:
                # For TPU, we have already assigned it to the device
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward propagation
            model_pred = self.model(data)
            if isinstance(model_pred, dict):
                assert "output" in model_pred.keys()
                model_pred = model_pred["output"]
            loss = self.loss_func(model_pred, target)

            # Backward propagation
            loss.backward()

            # Update parameters
            if self.use_tpu:
                import torch_xla.core.xla_model as xm

                xm.optimizer_step(self.optimizer)
            else:
                self.optimizer.step()

            # Metrics
            if self.use_tpu:
                rep_loss = loss.item()
            else:
                rep_loss = reduce_metric(loss, self.world_size).item()

            losses.update(rep_loss, data.size(0))

            if (not self.use_tpu) and (self.rank == 0):
                print_str = (
                    f"[Epoch {self.current_epoch}; Step {i+1}/{num_steps}] "
                    f"Train Loss {rep_loss:.6f}"
                )
                self.print_fn(f"{print_str}")

        average_loss = losses.avg
        if self.use_tpu:
            # Average across TPU replicas
            average_loss = xm.mesh_reduce("train_average_loss", average_loss, np.mean)

        # Print train results over entire dataset
        msg_str = "[Epoch {}] Train Loss: {:.6f}".format(
            self.current_epoch, average_loss
        )
        if self.use_tpu:
            # xm.master_print only prints on the first TPU core
            self.print_fn(msg_str)
        elif self.rank == 0:
            # print all reduce result on one gpu
            self.print_fn(msg_str)

        self.results["losses"]["train"].append(average_loss)

    def validate(self):
        assert hasattr(self, "val_loader")
        assert hasattr(self, "use_tpu")
        assert hasattr(self, "model")
        assert hasattr(self, "loss_func")
        assert hasattr(self, "device")

        losses = AverageMeter("Loss", ":.4e")
        num_steps = len(self.val_loader)

        self.set_model_to_eval()
        self.loss_func.eval()
        with torch.no_grad():
            for i, (data, target) in enumerate(self.val_loader):
                if not self.use_tpu:
                    # For TPU, we have already assigned it to the device
                    data = data.to(self.device, non_blocking=True)
                    target = target.to(self.device, non_blocking=True)

                model_pred = self.model(data)
                if isinstance(model_pred, dict):
                    assert "output" in model_pred.keys()
                    model_pred = model_pred["output"]
                loss = self.loss_func(model_pred, target)

                # Metrics
                if self.use_tpu:
                    rep_loss = loss.item()
                else:
                    rep_loss = reduce_metric(loss, self.world_size).item()

                losses.update(rep_loss, data.size(0))

                if (not self.use_tpu) and (self.rank == 0):
                    print_str = (
                        f"[Epoch {self.current_epoch}; Step {i+1}/{num_steps}] "
                        f"Val Loss {rep_loss:.6f}"
                    )
                    self.print_fn(f"{print_str}")

        average_loss = losses.avg
        if self.use_tpu:
            # Average across TPU replicas
            import torch_xla.core.xla_model as xm

            average_loss = xm.mesh_reduce("val_average_loss", average_loss, np.mean)

        # Print val results over entire dataset
        msg_str = "[Epoch {}] Val Loss: {:.6f}".format(self.current_epoch, average_loss)
        if self.use_tpu:
            # xm.master_print only prints on the first TPU core
            self.print_fn(msg_str)
        elif self.rank == 0:
            # print all reduce result on one gpu
            self.print_fn(msg_str)

        self.results["losses"]["val"].append(average_loss)

        # Check if current loss is best
        self.best_loss, self.is_best = check_best_loss(average_loss, self.best_loss)

    def save_checkpoint(self):
        assert hasattr(self, "current_epoch")
        assert hasattr(self, "model")
        assert hasattr(self, "optimizer")
        assert hasattr(self, "results")
        assert hasattr(self, "save_dir")
        assert hasattr(self, "best_loss")
        assert hasattr(self, "use_tpu")
        assert hasattr(self, "config")
        assert "save_freq" in self.config.keys()

        curr_state = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "decoder_state_dict": self.loss_func.decoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "results": self.results,
            "curr_best_loss": self.best_loss,
        }

        # Every "save_freq" epochs, save into new checkpoint file and relevant keys
        # to database. Overwrite existing checkpoint otherwise.
        save_epoch = None
        if (self.current_epoch % self.config["save_freq"] == 0) or (
            self.current_epoch + 1 == self.config["num_epochs"]
        ):
            save_epoch = self.current_epoch
            self._save_to_db(
                curr_state=curr_state, save_keys=["epoch", "results", "curr_best_loss"]
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
        assert hasattr(self, "device")
        assert hasattr(self, "model")
        assert hasattr(self, "optimizer")
        assert hasattr(self, "config")
        assert hasattr(self, "use_tpu")
        self.check_key("resume_checkpoint")

        checkpoint_path = self.config["resume_checkpoint"]
        if os.path.isfile(checkpoint_path):
            if self.use_tpu:
                cpt = torch.load(checkpoint_path)
            else:
                assert len(self.gpu_ids) == 1  # one subprocess per gpu
                cpt = torch.load(
                    checkpoint_path, map_location="cuda:{}".format(self.gpu_ids[0])
                )
            self.print_fn(f"Loaded checkpoint at '{checkpoint_path}'")
        else:
            raise ValueError(f"No checkpoint at '{checkpoint_path}'")

        # Make sure keys are in the checkpoint
        assert "epoch" in cpt.keys()
        assert "results" in cpt.keys()
        assert "model_state_dict" in cpt.keys()
        assert "decoder_state_dict" in cpt.keys()
        assert "optimizer" in cpt.keys()
        assert "curr_best_loss" in cpt.keys()

        # Load current epoch, +1 since we stored the last completed epoch
        self.current_epoch = cpt["epoch"] + 1

        # Load results
        assert not hasattr(self, "results")
        self.results = cpt["results"]

        # Load model state dict
        self.model.load_state_dict(cpt["model_state_dict"])

        # Place the loss function parameters onto multi-GPU
        self.loss_func = self.loss_func.to(self.device)
        assert hasattr(self, "gpu_ids")
        assert hasattr(self.loss_func, "decoder")
        if not isinstance(self.loss_func.decoder, nn.parallel.DistributedDataParallel):
            self.loss_func.decoder= nn.parallel.DistributedDataParallel(
                self.loss_func.decoder, device_ids=self.gpu_ids, broadcast_buffers=False
            )
        # Load the projection and prediction MLP weights
        self.loss_func.decoder.load_state_dict(cpt["decoder_state_dict"])

        # Load optimizer state dict
        self.optimizer.load_state_dict(cpt["optimizer"])

        # Load the current best accuracy
        assert not hasattr(self, "best_loss")
        self.best_loss = cpt["curr_best_loss"]

