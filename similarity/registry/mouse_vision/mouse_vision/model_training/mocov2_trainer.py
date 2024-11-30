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
from mouse_vision.core.model_loader_utils import get_model
from mouse_vision.model_training.train_utils import reduce_metric
from mouse_vision.model_training.imagenet_datasets import ImageNetContrastive
from mouse_vision.model_training.trainer import Trainer
from mouse_vision.models.model_transforms import MODEL_TRANSFORMS
from mouse_vision.core.default_dirs import IMAGENET_DATA_DIR
from mouse_vision.model_training.training_dataloader_utils import get_dataloaders
from mouse_vision.model_training.train_utils import (
    AverageMeter,
    check_best_accuracy,
    save_checkpoint,
    build_paramwise_options,
    annealing_cos
)


class MoCov2Trainer(Trainer):
    def __init__(self, config):
        super(MoCov2Trainer, self).__init__(config)

        # TPU training fails due to esoteric all_gather dtype issue
        assert not self.use_tpu

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
            self.best_acc = np.inf

        self.is_best = False

        # Place the loss function parameters onto multi-gpu
        self.loss_func = self.loss_func.to(self.device)
        if not self.use_tpu:
            assert hasattr(self, "gpu_ids")
            if not isinstance(self.loss_func, nn.parallel.DistributedDataParallel):
                self.loss_func = nn.parallel.DistributedDataParallel(self.loss_func,
                                                                     device_ids=self.gpu_ids)

    def initialize_model(self):
        assert hasattr(self, "device")
        self.check_key("model")

        model = get_model(
            self.config["model"],
            trained=False,
            model_family=self.config.get("model_family", None),
        )

        model = model.to(self.device)

        # we don't wrap the model in DDP
        # since it will be passed to the loss function
        # which will wrap everything in DDP

        model_name = self.config["model"]

        return model, model_name

    def initialize_loss_function(self):
        assert hasattr(self, "config")
        self.check_key("loss_params")

        assert "class" in self.config["loss_params"].keys()
        assert "model_output_dim" in self.config["loss_params"].keys()

        loss_class = self.config["loss_params"]["class"]
        loss_kwargs = copy.deepcopy(self.config["loss_params"])
        loss_kwargs.pop("class")
        loss_kwargs["encoder_q_backbone"] = self.model
        loss_kwargs["encoder_k_backbone"] = get_model(self.model_name,
                                                      trained=False,
                                                      model_family=self.config.get("model_family", None),
                                                     )
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

        optim = torch.optim.SGD(
            # since the model parameters are passed in through the loss
            self.loss_func.parameters(),
            lr=self.config["optimizer_params"]["initial_lr"],
            momentum=self.config["optimizer_params"]["momentum"],
            weight_decay=self.config["optimizer_params"]["weight_decay"],
        )
        return optim

    def adjust_learning_rate(self):
        # Cosine Annealing Schedule
        # Adapted from: https://github.com/open-mmlab/mmcv/blob/bcf85026c3f2683212d2a3a25f58102b3e5f75ef/mmcv/runner/hooks/lr_updater.py#L228-L248
        # using default values from: https://github.com/open-mmlab/OpenSelfSup/blob/796cd68297d57233ad59aaf0cf0d7f75926bfa6f/configs/selfsup/moco/r50_v2.py#L76
        assert hasattr(self, "optimizer")
        assert hasattr(self, "config")
        assert hasattr(self, "use_tpu")

        self.check_key("optimizer_params")
        assert "initial_lr" in self.config["optimizer_params"].keys()

        initial_lr = self.config["optimizer_params"]["initial_lr"]
        min_lr = self.config["optimizer_params"].get("min_lr", 0.0)
        curr_factor = ((float)(self.current_epoch + 1)) / (self.config["num_epochs"])
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
        # queue length has to evenly divide batch size
        params["drop_last"] = True

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
        top1 = AverageMeter("Acc@1", ":6.2f")
        top5 = AverageMeter("Acc@5", ":6.2f")
        num_steps = len(self.train_loader)

        self.set_model_to_train()
        # since the loss function now has parameters
        self.loss_func.train()
        assert self.loss_func.training
        # extra sanity check in case loss_func.train
        # resets model.train as model is fed into it
        assert self.model.training
        for i, data in enumerate(self.train_loader):
            if not self.use_tpu:
                data = data.to(self.device, non_blocking=True)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward propagation
            loss_dict = self.loss_func(data, mode="train")

            # Backward propagation
            loss_dict['loss'].backward()

            # Update parameters
            if self.use_tpu:
                import torch_xla.core.xla_model as xm

                xm.optimizer_step(self.optimizer)
            else:
                self.optimizer.step()

            # Metrics
            # acc1/acc5 are (K+1)-way contrast classifier accuracy
            # as is done here:
            # https://github.com/facebookresearch/moco/blob/90e244a23d135e8facca41f8aa47541cc30c61b2/main_moco.py#L305-L307
            if self.use_tpu:
                rep_loss = loss_dict['loss'].item()
                rep_acc1 = loss_dict['acc1'].item()
                rep_acc5 = loss_dict['acc5'].item()
            else:
                rep_loss = reduce_metric(loss_dict['loss'], self.world_size).item()
                rep_acc1 = reduce_metric(loss_dict['acc1'], self.world_size).item()
                rep_acc5 = reduce_metric(loss_dict['acc5'], self.world_size).item()

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
        elif (self.rank == 0):
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
        # since the loss function now has parameters
        self.loss_func.eval()
        assert not self.loss_func.training
        # extra sanity check in case loss_func.train
        # resets model.train as model is fed into it
        assert not self.model.training
        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                if not self.use_tpu:
                    data = data.to(self.device, non_blocking=True)

                loss_dict = self.loss_func(data, mode="val")

                # Metrics
                if self.use_tpu:
                    rep_loss = loss_dict['loss'].item()
                    rep_acc1 = loss_dict['acc1'].item()
                    rep_acc5 = loss_dict['acc5'].item()
                else:
                    rep_loss = reduce_metric(loss_dict['loss'], self.world_size).item()
                    rep_acc1 = reduce_metric(loss_dict['acc1'], self.world_size).item()
                    rep_acc5 = reduce_metric(loss_dict['acc5'], self.world_size).item()

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
        elif (self.rank == 0):
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
        assert hasattr(self, "best_acc")
        assert hasattr(self, "loss_func")
        assert hasattr(self, "use_tpu")
        assert hasattr(self, "config")
        assert "save_freq" in self.config.keys()
        if not self.use_tpu:
            assert isinstance(self.loss_func, nn.parallel.DistributedDataParallel)

        curr_state = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(), # we save the model state dict separately for neural fits
            "loss_state_dict": self.loss_func.state_dict(),
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
                curr_state=curr_state,
                save_keys=["epoch", "results", "curr_best_acc"]
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
            assert not isinstance(self.loss_func, nn.parallel.DistributedDataParallel)

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
        assert "loss_state_dict" in cpt.keys()
        assert "optimizer" in cpt.keys()
        assert "curr_best_acc" in cpt.keys()

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
            if not isinstance(self.loss_func, nn.parallel.DistributedDataParallel):
                self.loss_func = nn.parallel.DistributedDataParallel(self.loss_func,
                                                                     device_ids=self.gpu_ids)
        self.loss_func.load_state_dict(cpt["loss_state_dict"])

        # Load optimizer state dict
        self.optimizer.load_state_dict(cpt["optimizer"])

        # Load the current best accuracy
        self.best_acc = cpt["curr_best_acc"]
