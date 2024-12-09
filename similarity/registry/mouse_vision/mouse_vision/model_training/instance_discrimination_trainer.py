import os
import time

import torch
import torch.nn as nn
import torch.distributed as dist

import numpy as np

from torchvision import transforms

import mouse_vision.loss_functions as lf

from mouse_vision.model_training.train_utils import reduce_metric
from mouse_vision.model_training.imagenet_datasets import ImageNetIndex
from mouse_vision.model_training.other_datasets import DMLocomotionIndex
from mouse_vision.core.default_dirs import IMAGENET_DATA_DIR, DMLOCOMOTION_DATA_DIR
from mouse_vision.loss_functions.memory_bank import MemoryBank
from mouse_vision.models.model_transforms import MODEL_TRANSFORMS
from mouse_vision.model_training.trainer import Trainer
from mouse_vision.model_training.training_dataloader_utils import get_dataloaders
from mouse_vision.model_training.train_utils import (
    AverageMeter,
    check_best_accuracy,
    save_checkpoint,
)


class InstanceDiscriminationTrainer(Trainer):
    def __init__(self, config):
        super(InstanceDiscriminationTrainer, self).__init__(config)

        # Initialized in self._initialize_loss_function()
        assert hasattr(self, "memory_bank")

        # Initialized in self._initialize_dataloader()
        assert hasattr(self, "training_ordered_labels")

        assert hasattr(self, "use_tpu")
        assert hasattr(self, "device")
        assert hasattr(self, "loss_func")

        if not hasattr(self, "results"):
            self.results = dict()
            self.results["losses"] = {"train": []}
            self.results["accs_top1"] = {"train": [], "val": []}

        # We need this condition since the best accuracy up to the current epoch
        # could have been loaded from a previous checkpoint, as above.
        if not hasattr(self, "best_acc"):
            self.best_acc = -1.0

        # Wrap the embedding function of the loss function with DistributedDataParallel
        self.loss_func.to(self.device)
        if not self.use_tpu:
            assert hasattr(self, "gpu_ids")
            if not isinstance(self.loss_func._embedding_func, nn.parallel.DistributedDataParallel):
                self.loss_func._embedding_func = nn.parallel.DistributedDataParallel(
                    self.loss_func._embedding_func, device_ids=self.gpu_ids
                )

        self.is_best = False

    def _initialize_memory_bank(self):
        assert hasattr(self, "config")
        assert hasattr(self, "device")
        assert hasattr(self, "train_loader")

        self.check_key("loss_params")
        assert "embedding_dim" in self.config["loss_params"].keys()

        if self.use_tpu:
            memory_bank_size = len(self.train_loader._loader.dataset)
        else:
            memory_bank_size = len(self.train_loader.dataset)
        embedding_dim = self.config["loss_params"]["embedding_dim"]

        # Initialize the memory bank with random values
        memory_bank = MemoryBank(
            memory_bank_size, embedding_dim, self.device
        )

        return memory_bank

    def initialize_loss_function(self):
        assert hasattr(self, "config")
        self.check_key("loss_params")

        assert "class" in self.config["loss_params"].keys()
        assert "m" in self.config["loss_params"].keys()
        assert "gamma" in self.config["loss_params"].keys()
        assert "tau" in self.config["loss_params"].keys()
        assert "embedding_dim" in self.config["loss_params"].keys()
        assert "model_output_dim" in self.config["loss_params"].keys()

        # Loss class and class parameters
        loss_class = self.config["loss_params"]["class"]
        model_output_dim = self.config["loss_params"]["model_output_dim"]
        m = self.config["loss_params"]["m"]
        gamma = self.config["loss_params"]["gamma"]
        tau = self.config["loss_params"]["tau"]
        embedding_dim = self.config["loss_params"]["embedding_dim"]
        self.memory_bank = self._initialize_memory_bank()

        loss_func = lf.__dict__[loss_class](
            self.memory_bank.get_memory_bank(),
            model_output_dim,
            m=m,
            gamma=gamma,
            tau=tau,
            embedding_dim=embedding_dim,
        )
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
        params["dataset_class"] = ImageNetIndex
        params["image_dir"] = IMAGENET_DATA_DIR
        params["dataset"] = "imagenet"
        params["return_indices"] = True
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

        if self.use_tpu:
            train_set = train_loader._loader.dataset.dataset.samples
        else:
            train_set = train_loader.dataset.dataset.samples

        self.training_ordered_labels = np.array(
            [train_set[i][1] for i in range(len(train_set))]
        )

        print("Labels shape:", self.training_ordered_labels.shape)

        return train_loader, val_loader

    def initialize_optimizer(self):
        assert hasattr(self, "config")
        assert hasattr(self, "model")
        assert hasattr(self, "loss_func")
        self.check_key("optimizer_params")

        assert "initial_lr" in self.config["optimizer_params"].keys()
        assert "weight_decay" in self.config["optimizer_params"].keys()

        if self.config.get("optimizer", "SGD") == "Adam":
            print("Using Adam optimizer")
            optim = torch.optim.Adam(
                list(self.model.parameters()) + list(self.loss_func.trainable_parameters()),
                lr=self.config["optimizer_params"]["initial_lr"],
                weight_decay=self.config["optimizer_params"]["weight_decay"],
            )
        else:
            print("Using default SGD optimizer")
            assert "momentum" in self.config["optimizer_params"].keys()
            optim = torch.optim.SGD(
                list(self.model.parameters()) + list(self.loss_func.trainable_parameters()),
                lr=self.config["optimizer_params"]["initial_lr"],
                momentum=self.config["optimizer_params"]["momentum"],
                weight_decay=self.config["optimizer_params"]["weight_decay"],
            )

        return optim

    def train_one_epoch(self):
        assert hasattr(self, "config")
        assert hasattr(self, "train_loader")
        assert hasattr(self, "model")
        assert hasattr(self, "loss_func")
        assert hasattr(self, "device")
        assert hasattr(self, "use_tpu")

        losses = AverageMeter("Loss", ":.4e")
        num_steps = len(self.train_loader)

        self.set_model_to_train()
        self.loss_func.train()
        for i, (indices, data, labels) in enumerate(self.train_loader):
            end = time.time()

            if not self.use_tpu:
                indices = indices.to(self.device, non_blocking=True)
                data = data.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward propagation
            outputs = self.model(data)
            loss, entries_to_update, data_loss, noise_loss = self.loss_func(
                outputs, indices
            )

            # Perform gathering operation
            if self.use_tpu:
                import torch_xla.core.xla_model as xm
                entries = xm.all_gather(entries_to_update, dim=0)
                # casting to int32 from int64 since gather does not work on int64
                all_indices = xm.all_gather(indices.int(), dim=0)
            else:
                # First define the list of tensors
                entries = []
                all_indices = []
                for _ in range(self.world_size):
                    entries.append(entries_to_update)
                    all_indices.append(indices)
                # This gather operation is in-place
                dist.all_gather(entries, entries_to_update)
                dist.all_gather(all_indices, indices)

                # Concatenate gathered list of tensors into tensor
                entries = torch.cat(entries, dim=0)
                all_indices = torch.cat(all_indices, dim=0)
            assert entries.ndim == 2, entries.shape

            # Backward propagation
            loss.backward()

            # Update parameters
            if self.use_tpu:
                import torch_xla.core.xla_model as xm

                xm.optimizer_step(self.optimizer)
            else:
                self.optimizer.step()

            # After each batch, update the memory bank for each GPU.
            with torch.no_grad():
                self.memory_bank.update(all_indices, entries)
                self.loss_func.update_memory_bank(self.memory_bank.get_memory_bank())

            # Step time
            step_time = time.time() - end

            # Metrics
            if self.use_tpu:
                reduced_loss = loss.item()
                reduced_data_loss = data_loss.item()
                reduced_noise_loss = noise_loss.item()
            else:
                reduced_loss = reduce_metric(loss, self.world_size).item()
                reduced_data_loss = reduce_metric(data_loss, self.world_size).item()
                reduced_noise_loss = reduce_metric(noise_loss, self.world_size).item()

            losses.update(reduced_loss, data.size(0))

            if (not self.use_tpu) and (self.rank == 0):
                print_str = (
                    f"[Epoch {self.current_epoch}; Step {i+1}/{num_steps}] "
                    f"Train Loss {reduced_loss:.3f}; Data Loss {reduced_data_loss:.3f}; "
                    f"Noise Loss {reduced_noise_loss:.3f}; Time {step_time:.3f} sec"
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
            self.print_fn(msg_str)

        # Append metric to results dictionary
        self.results["losses"]["train"].append(average_loss)

    def validate(self):
        assert hasattr(self, "val_loader")
        assert hasattr(self, "model")
        assert hasattr(self, "loss_func")
        assert hasattr(self, "world_size")
        assert hasattr(self, "device")
        assert hasattr(self, "use_tpu")

        top1 = AverageMeter("Acc@1", ":6.2f")
        num_steps = len(self.val_loader)

        self.set_model_to_eval()
        self.loss_func.eval()
        with torch.no_grad():
            for i, (_, data, labels) in enumerate(self.val_loader):
                if not self.use_tpu:
                    data, labels = data.to(self.device), labels.to(self.device)

                # Obtain model outputs
                outputs = self.model(data)

                # Metrics
                accuracy = self._compute_accuracy(outputs, labels)
                if self.use_tpu:
                    reduce_accuracy = accuracy.item()
                else:
                    reduce_accuracy = reduce_metric(accuracy, self.world_size).item()

                top1.update(reduce_accuracy, data.size(0))

                if (not self.use_tpu) and (self.rank == 0):
                    print_str = (
                        f"[Epoch {self.current_epoch}; Step {i+1}/{num_steps}]; "
                        f"Val Accuracy: {reduce_accuracy:.6f}"
                    )
                    self.print_fn(f"{print_str}")

        average_top1 = top1.avg
        if self.use_tpu:
            # Average across TPU replicas
            import torch_xla.core.xla_model as xm

            average_top1 = xm.mesh_reduce("val_average_top1", average_top1, np.mean)

        # Print val results over entire dataset
        msg_str = "[Epoch {}] Val Accuracy: {:.6f}".format(
            self.current_epoch, average_top1
        )
        if self.use_tpu:
            # xm.master_print only prints on the first TPU core
            self.print_fn(msg_str)
        elif self.rank == 0:
            self.print_fn(msg_str)

        # Append metric to results dictionary
        self.results["accs_top1"]["val"].append(average_top1)

        # Check if current top-1 accuracy is best
        self.best_acc, self.is_best = check_best_accuracy(average_top1, self.best_acc)

    def _compute_accuracy(self, outputs, labels):
        """
        Get top-1 accuracy by obtaining nearest neighbour from memory bank.

        Inputs:
            outputs : (torch.Tensor) model outputs
            labels  : (torch.Tensor) actual labels of data

        Outputs:
            accuracy : (torch.Tensor) top-1 accuracy
        """
        assert hasattr(self, "training_ordered_labels")
        assert hasattr(self, "loss_func")

        self.loss_func.eval()
        with torch.no_grad():
            # Model output
            # 128-dimensional embedding
            outputs = self.loss_func._embedding_func(outputs)

            # 1) Obtain inner products of features with memory bank
            all_inner_products = self.memory_bank.get_all_inner_products(outputs)
            # 2) Obtain top-1 nearest neighbour memory bank index
            _, nearest_idx = torch.topk(all_inner_products, k=1, dim=1)
            nearest_idx = nearest_idx.squeeze().cpu().numpy()
            # 3) Obtain labels for the top-1 nearest neighbour
            nearest_neighbour_labels = torch.from_numpy(
                self.training_ordered_labels[nearest_idx]
            )
            assert nearest_neighbour_labels.shape == labels.shape
            # 4) Compute accuracy
            num_correct = torch.eq(nearest_neighbour_labels, labels.cpu()).sum()
            accuracy = num_correct / float(labels.shape[0])

        return accuracy.to(self.device)

    def save_checkpoint(self):
        assert hasattr(self, "use_tpu")
        assert hasattr(self, "current_epoch")
        assert hasattr(self, "model")
        assert hasattr(self, "optimizer")
        assert hasattr(self, "memory_bank")
        assert hasattr(self, "results")
        assert hasattr(self, "best_acc")
        assert hasattr(self, "loss_func")
        assert hasattr(self, "config")
        assert "save_freq" in self.config.keys()

        curr_state = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "embedding_func": self.loss_func._embedding_func.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "memory_bank": self.memory_bank.as_tensor(),
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
        assert hasattr(self, "device")
        assert hasattr(self, "use_tpu")
        assert hasattr(self, "memory_bank")
        assert hasattr(self, "loss_func")

        checkpoint_path = self.config["resume_checkpoint"]
        if os.path.isfile(checkpoint_path):
            if self.use_tpu:
                # On TPU, the Trainer locally saves each core's copy of the
                # same saved parameters to a different local file name, and
                # there is a single TPU device we are loading to
                cpt = torch.load(checkpoint_path)
            else:
                assert hasattr(self, "gpu_ids")
                assert len(self.gpu_ids) == 1
                loc = f"cuda:{self.gpu_ids[0]}"
                cpt = torch.load(checkpoint_path, map_location=loc)
            self.print_fn(f"Loaded checkpoint at '{checkpoint_path}'")
        else:
            raise ValueError(f"No checkpoint at '{checkpoint_path}'")

        # Make sure keys are in the checkpoint
        assert "epoch" in cpt.keys()
        assert "results" in cpt.keys()
        assert "model_state_dict" in cpt.keys()
        assert "embedding_func" in cpt.keys()
        assert "memory_bank" in cpt.keys()
        assert "optimizer" in cpt.keys()
        assert "curr_best_acc" in cpt.keys()

        # Load current epoch, +1 since we stored the last completed epoch
        self.current_epoch = cpt["epoch"] + 1

        # Load results
        assert not hasattr(self, "results")
        self.results = cpt["results"]

        # Load model state dict
        self.model.load_state_dict(cpt["model_state_dict"])

        # Load embedding function (model output -> low-dimension embedding)
        self.loss_func.to(self.device)
        if not self.use_tpu:
            assert hasattr(self, "gpu_ids")
            self.loss_func._embedding_func = nn.parallel.DistributedDataParallel(
                self.loss_func._embedding_func, device_ids=self.gpu_ids
            )
        self.loss_func._embedding_func.load_state_dict(cpt["embedding_func"])

        # Load memory bank
        memory_bank = cpt["memory_bank"].to(self.device)
        self.memory_bank.set_memory_bank(memory_bank)

        # Update memory bank in loss function
        memory_bank = self.memory_bank.get_memory_bank()
        self.loss_func.update_memory_bank(memory_bank)

        # Load optimizer state dict
        self.optimizer.load_state_dict(cpt["optimizer"])

        # Load the current best accuracy
        assert not hasattr(self, "best_acc")
        self.best_acc = cpt["curr_best_acc"]

class DMLocomotionInstanceDiscriminationTrainer(InstanceDiscriminationTrainer):
    def __init__(self, config):
        super(DMLocomotionInstanceDiscriminationTrainer, self).__init__(config=config)

    def initialize_dataloader(self):
        assert hasattr(self, "config")
        assert hasattr(self, "use_tpu")
        assert hasattr(self, "model_name")

        self.check_key("optimizer_params")
        self.check_key("dataloader_workers")

        assert "train_batch_size" in self.config["optimizer_params"].keys()
        # no val set in this case
        assert "val_batch_size" not in self.config["optimizer_params"].keys()

        params = dict()
        params["dataset_class"] = DMLocomotionIndex
        params["image_dir"] = DMLOCOMOTION_DATA_DIR
        params["dataset"] = "dmlocomotion"
        params["num_train_imgs"] = self.config["optimizer_params"].get("num_train_imgs", None)
        params["img_sampler_seed"] = self.config["optimizer_params"].get("img_sampler_seed", 0)
        params["return_indices"] = True
        params["train_batch_size"] = self.config["optimizer_params"]["train_batch_size"]
        params["num_workers"] = self.config["dataloader_workers"]

        my_transforms = dict()
        my_transforms["train"] = transforms.Compose(
            MODEL_TRANSFORMS[self.model_name]["train"]
        )

        train_loader, val_loader = get_dataloaders(
            params,
            my_transforms=my_transforms,
            device=self.device,
            rank=self.rank,
            world_size=self.world_size
        )
        assert(val_loader is None)
        if self.use_tpu:
            train_set = train_loader._loader.dataset.dataset.samples
        else:
            train_set = train_loader.dataset.dataset.samples

        self.training_ordered_labels = np.array(
            [train_set[i][1] for i in range(len(train_set))]
        )

        print("Labels shape:", self.training_ordered_labels.shape)

        return train_loader, val_loader

    def validate(self):
        # no val set in this case
        pass

if __name__ == "__main__":
    t = InstanceDiscriminationTrainer("configs/test_ir_trainer_test.json")
    t.save_checkpoint()
    curr_mem_bank = t.memory_bank.as_tensor()

    t.config["resume_checkpoint"] = t.config["save_dir"] + "checkpoint_epoch_0.pt"
    t.config["gpus"] = [5, 6]
    t = InstanceDiscriminationTrainer(t.config)
    new_mem_bank = t.memory_bank.as_tensor()
    for mem in t.memory_bank.get_memory_bank():
        print(torch.equal(curr_mem_bank, mem.cpu()))
    for mem in t.loss_func.memory_bank:
        print(torch.equal(curr_mem_bank, mem.cpu()))

