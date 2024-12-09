import os
import sys
import copy
from itertools import chain
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torchvision import transforms

import mouse_vision.loss_functions as lf
from mouse_vision.model_training.train_utils import reduce_metric
from mouse_vision.model_training.imagenet_datasets import ImageNetContrastive
from mouse_vision.model_training.trainer import Trainer
from mouse_vision.model_training.custom_optimizers import LARS
from mouse_vision.models.model_transforms import MODEL_TRANSFORMS
from mouse_vision.core.default_dirs import IMAGENET_DATA_DIR
from mouse_vision.model_training.training_dataloader_utils import get_dataloaders
from mouse_vision.model_training.train_utils import (
    AverageMeter,
    check_best_loss,
    save_checkpoint,
    build_paramwise_options,
    annealing_cos
)


class SimCLRTrainer(Trainer):
    def __init__(self, config):
        super(SimCLRTrainer, self).__init__(config)

        # TPU training runs in principle, but we recommend against it
        # because training is slow with it & SyncBN is currently unsupported
        # which is important for performance
        assert not self.use_tpu

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
        if not self.use_tpu:
            assert hasattr(self, "gpu_ids")
            if not isinstance(self.loss_func.neck, nn.parallel.DistributedDataParallel):
                self.loss_func.neck = nn.parallel.DistributedDataParallel(self.loss_func.neck,
                                                                                 device_ids=self.gpu_ids)

    def initialize_loss_function(self):
        assert hasattr(self, "config")
        self.check_key("loss_params")

        assert "class" in self.config["loss_params"].keys()
        assert "model_output_dim" in self.config["loss_params"].keys()

        loss_class = self.config["loss_params"]["class"]
        loss_kwargs = copy.deepcopy(self.config["loss_params"])
        loss_kwargs.pop("class")
        loss_kwargs["tpu"] = self.use_tpu
        loss_func = lf.__dict__[loss_class](**loss_kwargs)
        return loss_func

    def initialize_optimizer(self):
        assert hasattr(self, "config")
        assert hasattr(self, "model")
        assert hasattr(self, "loss_func")
        self.check_key("optimizer_params")

        assert "initial_lr" in self.config["optimizer_params"].keys()
        assert "momentum" in self.config["optimizer_params"].keys()
        assert "weight_decay" in self.config["optimizer_params"].keys()

        named_parameters = chain(self.model.named_parameters(), self.loss_func.named_parameters())
        # Taken from: https://github.com/open-mmlab/OpenSelfSup/blob/ed5000482b0d8b816cd8a6fbbb1f97da44916fed/configs/selfsup/simclr/r50_bs256_ep200.py#L73-L75
        DEFAULT_PARAMWISE_OPTIONS = {"(bn|gn)(\d+)?.(weight|bias)": dict(weight_decay=0., lars_exclude=True),
                                     "bias": dict(weight_decay=0., lars_exclude=True)}
        if self.config["optimizer_params"].get("paramwise_options", DEFAULT_PARAMWISE_OPTIONS) is not None:
            params_to_train = build_paramwise_options(named_parameters=named_parameters,
                                                      paramwise_options=self.config["optimizer_params"].get("paramwise_options", DEFAULT_PARAMWISE_OPTIONS))
        else:
            params_to_train = named_parameters

        optim = LARS(
            params_to_train,
            lr=self.config["optimizer_params"]["initial_lr"],
            momentum=self.config["optimizer_params"]["momentum"],
            weight_decay=self.config["optimizer_params"]["weight_decay"],
        )
        return optim

    def adjust_learning_rate(self):
        # Cosine Annealing Schedule
        # Adapted from: https://github.com/open-mmlab/mmcv/blob/bcf85026c3f2683212d2a3a25f58102b3e5f75ef/mmcv/runner/hooks/lr_updater.py#L228-L248
        # using default values from: https://github.com/open-mmlab/OpenSelfSup/blob/ed5000482b0d8b816cd8a6fbbb1f97da44916fed/configs/selfsup/simclr/r50_bs256_ep200.py#L78-L83
        assert hasattr(self, "optimizer")
        assert hasattr(self, "config")
        assert hasattr(self, "use_tpu")

        self.check_key("optimizer_params")
        assert "initial_lr" in self.config["optimizer_params"].keys()

        initial_lr = self.config["optimizer_params"]["initial_lr"]
        min_lr = self.config["optimizer_params"].get("min_lr", 0.0)
        warmup_epochs = self.config["optimizer_params"].get("warmup_epochs", 10)
        warmup_ratio = self.config["optimizer_params"].get("warmup_ratio", 0.0001)
        # logic confirmed from: https://github.com/google-research/simclr/blob/67b562524223e65c0414a15c3bd7ac007d28f629/tf2/model.py#L103-L108
        # and adapted since our epochs are 0 indexed so epoch 0 is the first epoch
        if (warmup_epochs > 0) and (self.current_epoch + 1 <= warmup_epochs):
            k = (1 - (((float)(self.current_epoch + 1)) / warmup_epochs)) * (1 - warmup_ratio)
            new_lr = (1 - k) * initial_lr
        else:
            curr_factor = ((float)((self.current_epoch + 1) - warmup_epochs)) / (self.config["num_epochs"] - warmup_epochs)
            new_lr = annealing_cos(start=initial_lr,
                                   end=min_lr,
                                   factor=curr_factor)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

        self.print_fn(f"Updating learning rate of Cosine Annealing to: {new_lr}")

    def initialize_dataloader(self):
        assert hasattr(self, "config")
        assert hasattr(self, "use_tpu")
        assert hasattr(self, "model_name")

        self.check_key("optimizer_params")
        self.check_key("dataloader_workers")

        assert "train_batch_size" in self.config["optimizer_params"].keys()
        assert "val_batch_size" in self.config["optimizer_params"].keys()

        params = dict()
        params["dataset_class"] = ImageNetContrastive
        params["image_dir"] = IMAGENET_DATA_DIR
        params["dataset"] = "imagenet"
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
            world_size=self.world_size
        )
        return train_loader, val_loader

    def train_one_epoch(self):
        assert hasattr(self, "config")
        assert hasattr(self, "train_loader")
        assert hasattr(self, "use_tpu")
        assert hasattr(self, "model")
        assert hasattr(self, "loss_func")
        assert hasattr(self, "device")

        losses = AverageMeter("Loss", ":.4e")
        num_steps = len(self.train_loader)

        self.set_model_to_train()
        # since the loss function now has parameters
        self.loss_func.train()
        assert self.loss_func.training
        for i, data in enumerate(self.train_loader):
            if not self.use_tpu:
                data = data.to(self.device, non_blocking=True)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward propagation
            loss = self.loss_func(self.model, data)

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
        elif (self.rank == 0):
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
        # since the loss function now has parameters
        self.loss_func.eval()
        assert not self.loss_func.training
        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                if not self.use_tpu:
                    data = data.to(self.device, non_blocking=True)

                loss = self.loss_func(self.model, data)

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
        msg_str = "[Epoch {}] Val Loss: {:.6f}".format(
            self.current_epoch, average_loss
        )
        if self.use_tpu:
            # xm.master_print only prints on the first TPU core
            self.print_fn(msg_str)
        elif (self.rank == 0):
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
        assert hasattr(self, "loss_func")
        assert hasattr(self, "use_tpu")
        assert hasattr(self, "config")
        assert "save_freq" in self.config.keys()
        if not self.use_tpu:
            assert isinstance(self.loss_func.neck, nn.parallel.DistributedDataParallel)

        curr_state = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "neck_state_dict": self.loss_func.neck.state_dict(),
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
                curr_state=curr_state,
                save_keys=["epoch", "results", "curr_best_loss"]
            )

        save_checkpoint(
            state=curr_state,
            save_dir=self.save_dir,
            is_best=self.is_best,
            save_epoch=save_epoch,
            rank=self.rank,
            tpu=self.use_tpu
        )

    def load_checkpoint(self):
        assert hasattr(self, "device")
        assert hasattr(self, "model")
        assert hasattr(self, "optimizer")
        assert hasattr(self, "config")
        assert hasattr(self, "use_tpu")
        assert hasattr(self, "loss_func")
        self.check_key("resume_checkpoint")
        if not self.use_tpu:
            assert not isinstance(self.loss_func.neck, nn.parallel.DistributedDataParallel)

        checkpoint_path = self.config["resume_checkpoint"]
        if os.path.isfile(checkpoint_path):
            if self.use_tpu:
                cpt = torch.load(checkpoint_path)
            else:
                assert(len(self.gpu_ids) == 1) # one subprocess per gpu
                cpt = torch.load(checkpoint_path,
                                 map_location = "cuda:{}".format(self.gpu_ids[0]))
            self.print_fn(f"Loaded checkpoint at '{checkpoint_path}'")
        else:
            raise ValueError(f"No checkpoint at '{checkpoint_path}'")

        # Make sure keys are in the checkpoint
        assert "epoch" in cpt.keys()
        assert "results" in cpt.keys()
        assert "model_state_dict" in cpt.keys()
        assert "neck_state_dict" in cpt.keys()
        assert "optimizer" in cpt.keys()
        assert "curr_best_loss" in cpt.keys()

        # Load current epoch, +1 since we stored the last completed epoch
        self.current_epoch = cpt["epoch"] + 1

        # Load results
        assert not hasattr(self, "results")
        self.results = cpt["results"]

        # Load model state dict
        self.model.load_state_dict(cpt["model_state_dict"])

        # Place the loss function parameters onto multi-gpu
        self.loss_func = self.loss_func.to(self.device)
        if not self.use_tpu:
            assert hasattr(self, "gpu_ids")
            if not isinstance(self.loss_func.neck, nn.parallel.DistributedDataParallel):
                self.loss_func.neck = nn.parallel.DistributedDataParallel(self.loss_func.neck,
                                                                                 device_ids=self.gpu_ids)
        self.loss_func.neck.load_state_dict(cpt["neck_state_dict"])

        # Load optimizer state dict
        self.optimizer.load_state_dict(cpt["optimizer"])

        # Load the current best accuracy
        assert not hasattr(self, "best_loss")
        self.best_loss = cpt["curr_best_loss"]
