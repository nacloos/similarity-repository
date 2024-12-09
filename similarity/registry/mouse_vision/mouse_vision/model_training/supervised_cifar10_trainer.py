import os
import sys

import numpy as np
import torch
import torch.distributed as dist
from torchvision import transforms

import mouse_vision.loss_functions as lf

from mouse_vision.model_training.train_utils import reduce_metric
from mouse_vision.model_training.trainer import Trainer
from mouse_vision.models.model_transforms import MODEL_TRANSFORMS
from mouse_vision.core.default_dirs import CIFAR10_DATA_DIR
from mouse_vision.model_training.training_dataloader_utils import get_dataloaders
from mouse_vision.model_training.train_utils import (
    AverageMeter,
    compute_accuracy,
    check_best_accuracy,
    save_checkpoint,
)


class SupervisedCIFAR10Trainer(Trainer):
    def __init__(self, config):
        super(SupervisedCIFAR10Trainer, self).__init__(config)

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

        self.is_best = False

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
        assert "momentum" in self.config["optimizer_params"].keys()
        assert "weight_decay" in self.config["optimizer_params"].keys()

        optim = torch.optim.SGD(
            self.model.parameters(),
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
        params["image_dir"] = CIFAR10_DATA_DIR
        params["dataset"] = "cifar10"
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
        top1 = AverageMeter("Acc@1", ":6.2f")
        top5 = AverageMeter("Acc@5", ":6.2f")
        num_steps = len(self.train_loader)

        self.set_model_to_train()
        for i, (data, labels) in enumerate(self.train_loader):
            if not self.use_tpu:
                # For TPU, we have already assigned it to the device
                # use non_blocking: https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/5
                data = data.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward propagation
            loss, predictions = self.loss_func(self.model, data, labels)

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

            if (not self.use_tpu) and (self.rank == 0):
                print_str = (
                    f"[Epoch {self.current_epoch}; Step {i+1}/{num_steps}] "
                    f"Train Loss {rep_loss:.6f}; Train Accuracy: {rep_acc1:.6f}"
                )
                self.print_fn(f"{print_str}")

        average_loss = losses.avg
        average_top1 = top1.avg
        average_top5 = top5.avg
        if self.use_tpu:
            # average across TPU replicas
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
            # print all reduce result on one gpu
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
        with torch.no_grad():
            for i, (data, labels) in enumerate(self.val_loader):
                if not self.use_tpu:
                    # For TPU, we have already assigned it to the device
                    # use non_blocking: https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/5
                    data = data.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)

                loss, predictions = self.loss_func(self.model, data, labels)

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
            # print all reduce result on one gpu
            self.print_fn(msg_str)

        self.results["losses"]["val"].append(average_loss)
        self.results["accs_top1"]["val"].append(average_top1)
        self.results["accs_top5"]["val"].append(average_top5)

        # Check if current top-1 accuracy is best
        self.best_acc, self.is_best = check_best_accuracy(average_top1, self.best_acc)

    def save_checkpoint(self):
        assert hasattr(self, "current_epoch")
        assert hasattr(self, "model")
        assert hasattr(self, "optimizer")
        assert hasattr(self, "results")
        assert hasattr(self, "save_dir")
        assert hasattr(self, "use_tpu")
        assert hasattr(self, "config")
        assert "save_freq" in self.config.keys()

        curr_state = {
            "epoch": self.current_epoch,
            "state_dict": self.model.state_dict(),
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
        self.check_key("resume_checkpoint")

        checkpoint_path = self.config["resume_checkpoint"]
        if os.path.isfile(checkpoint_path):
            if self.use_tpu:
                # on TPU, the Trainer locally saves each core's copy of the same saved parameters
                # to a different local file name, and there is a single TPU device we are loading to
                cpt = torch.load(checkpoint_path)
            else:
                # According to: https://stackoverflow.com/questions/61642619/pytorch-distributed-data-parallel-confusion
                # When saving the parameters (or any tensor for that matter)
                # PyTorch includes the device where it was stored. On gpu,
                # this is always gpu 0. Therefore, we ensure we map the SAME parameters to each
                # other gpu (not just gpu 0), otherwise it will load the same model
                # multiple times on 1 gpu.
                assert(len(self.gpu_ids) == 1) # one subprocess per gpu
                cpt = torch.load(checkpoint_path,
                                 map_location = "cuda:{}".format(self.gpu_ids[0]))
            self.print_fn(f"Loaded checkpoint at '{checkpoint_path}'")
        else:
            raise ValueError(f"No checkpoint at '{checkpoint_path}'")

        # Make sure keys are in the checkpoint
        assert "epoch" in cpt.keys()
        assert "results" in cpt.keys()
        assert "state_dict" in cpt.keys()
        assert "optimizer" in cpt.keys()
        assert "curr_best_acc" in cpt.keys()

        # Load current epoch, +1 since we stored the last completed epoch
        self.current_epoch = cpt["epoch"] + 1

        # Load results
        assert not hasattr(self, "results")
        self.results = cpt["results"]

        # Load model state dict
        self.model.load_state_dict(cpt["state_dict"])

        # Load optimizer state dict
        self.optimizer.load_state_dict(cpt["optimizer"])

        # Load the current best accuracy
        self.best_acc = cpt["curr_best_acc"]

