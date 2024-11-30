import os
import random
import shutil

import torch

import numpy as np
import torch.nn as nn

from mouse_vision.core.model_loader_utils import get_model
from mouse_vision.model_training.dbinterface import MongoInterface
from mouse_vision.model_training.train_utils import parse_config


class Trainer:
    def __init__(self, config, config_file_path=None):
        if isinstance(config, str):
            # Read configuration file first if filepath
            self.config = parse_config(config)
        else:
            assert isinstance(config, dict)
            self.config = config

        # Set up checkpoint save directory
        self.save_dir = self._make_save_dir()

        # Set reproducibility seed
        self._set_seed()

        # Set device, print function, model, loss, etc.
        self.device = self._set_device()
        self.model, self.model_name = self.initialize_model()
        self.train_loader, self.val_loader = self.initialize_dataloader()
        self.loss_func = self.initialize_loss_function()
        self.optimizer = self.initialize_optimizer()

        # Set MongoDB Interface
        if not self.config.get("use_mongodb", False):
            self.use_mongodb = self.config.get("use_mongodb", False)
            self.database = None
        else:
            self.use_mongodb = True
            self.database = MongoInterface(
                database_name=self.config["db_name"],
                collection_name=self.config["coll_name"],
                port=self.config["port"],
                print_fn=self.print_fn,
            )

        # This will be changed depending on whether or not we are loading from a
        # checkpoint. See the derived class' implementation of load_checkpoint().
        self.current_epoch = 0

        # Before doing anything else, save the configuration file to exp directory
        # so we can remember the exact experiment settings.
        self._save_config_file(self.config["filepath"])

        # If resume_checkpoint is provided, then load from checkpoint
        self.check_key("resume_checkpoint")
        if self.config["resume_checkpoint"] is not None:
            if self.use_tpu:
                from mouse_vision.model_training.gcloud_utils \
                    import download_file_from_bucket

                # Download from gs bucket to local directory, where each ordinal has
                # its own copy of the same file so that they are each loading the same
                # data
                self.config["resume_checkpoint"] = download_file_from_bucket(
                    filename=self.config["resume_checkpoint"],
                    ordinal=self.rank,
                    print_fn=self.print_fn,
                )
            self.load_checkpoint()
            if self.use_tpu:
                # Remove the file locally after loading from gs bucket
                os.remove(self.config["resume_checkpoint"])

    def check_key(self, key):
        assert hasattr(self, "config")
        assert key in self.config.keys(), f"{key} undefined in config file."

    def _make_save_dir(self):
        self.check_key("save_dir")
        save_dir = self.config["save_dir"]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        return save_dir

    def _save_config_file(self, config_file_path):
        assert hasattr(self, "use_tpu")
        assert hasattr(self, "save_dir")

        config_filename = config_file_path.split("/")[-1]
        config_copy_name = os.path.join(self.save_dir, f"{config_filename}")
        if self.rank == 0:
            shutil.copyfile(config_file_path, config_copy_name)
            if self.use_tpu:
                from mouse_vision.model_training.gcloud_utils import save_file_to_bucket
                save_file_to_bucket(filename=config_copy_name)

    def _save_to_db(self, curr_state, save_keys):
        # NOTE: these keys purposefully exclude the state dicts because the state
        # dicts have NOT been coordinated across tpu cores yet, that is only done
        # by the xm.save() cmd in save_checkpoint. These included keys to be saved
        # to the database, however, are the same across tpu cores since they are
        # the result of xm.mesh_reduce().
        if not self.use_mongodb:
            return
        record = {"exp_id": self.config["exp_id"]}
        record.update({k: curr_state[k] for k in save_keys})
        if self.rank == 0:
            self.database.save(record)

    def _set_seed(self):
        """
        Sets the random seed to make entire training process reproducible.

        Inputs:
            seed : (int) random seed
        """
        self.check_key("seed")
        seed = self.config["seed"]

        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # If using multi-GPU
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    def _set_device(self):
        self.check_key("gpus")
        self.check_key("tpu")

        self.print_fn = print
        self.use_tpu = False

        if self.config["tpu"]:
            # TPU device; Use TPU
            assert not self.config["gpus"], f"Cannot enable both TPU and GPU."

            import torch_xla.core.xla_model as xm

            device = xm.xla_device()
            self.use_tpu = True
            self.rank = xm.get_ordinal()
            self.world_size = xm.xrt_world_size()
            self.print_fn = xm.master_print
            self.print_fn(f"Using TPU...")

        elif self.config["gpus"]:
            # GPU device; Use GPU
            assert torch.cuda.is_available(), "Cannot use GPU."
            assert not self.config["tpu"], f"Cannot enable both TPU and GPU."

            import torch.distributed as dist

            # each subprocess gets its own gpu
            self.gpu_ids = self.config["gpus"]
            assert(len(self.gpu_ids) == 1)
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            torch.cuda.set_device(self.gpu_ids[0])
            device = torch.device(f"cuda:{self.gpu_ids[0]}")
            self.print_fn(f"Subprocess {self.rank} is on GPU {self.gpu_ids}. {self.world_size} GPUs total.")

        else:
            # CPU not supported, makes code ugly
            # and we likely won't test this use case anyway
            raise ValueError

        return device

    def adjust_learning_rate(self):
        # TODO: This may need to be reimplemented in a derived class if, for
        # example, different layers have different learning rates or learning
        # rate schedules.

        assert hasattr(self, "optimizer")
        assert hasattr(self, "config")
        assert hasattr(self, "use_tpu")

        self.check_key("optimizer_params")
        assert "lr_decay_schedule" in self.config["optimizer_params"].keys()
        assert "lr_decay_rate" in self.config["optimizer_params"].keys()
        assert "initial_lr" in self.config["optimizer_params"].keys()

        initial_lr = self.config["optimizer_params"]["initial_lr"]
        lr_decay_schedule = self.config["optimizer_params"]["lr_decay_schedule"]
        lr_decay_rate = self.config["optimizer_params"]["lr_decay_rate"]
        assert isinstance(lr_decay_rate, float)

        steps = np.sum(self.current_epoch >= np.asarray(lr_decay_schedule))
        if steps > 0:
            new_lr = initial_lr * (lr_decay_rate ** steps)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = new_lr

            self.print_fn(f"Updating learning rate to: {new_lr}")

    def initialize_model(self):
        assert hasattr(self, "device")
        assert hasattr(self, "use_tpu")
        self.check_key("model")

        model = get_model(
            self.config["model"],
            trained=False,
            model_family=self.config.get("model_family", None),
        )

        model = model.to(self.device)

        # gpu training
        if not self.use_tpu:
            assert hasattr(self, "gpu_ids")
            model = nn.parallel.DistributedDataParallel(model,
                                                        device_ids=self.gpu_ids)

        model_name = self.config["model"]

        return model, model_name

    def train(self):
        """
        Main entry point for training a model.
        """
        assert hasattr(self, "train_loader")
        self.check_key("save_freq")
        self.check_key("num_epochs")

        for i in range(self.current_epoch, self.config["num_epochs"]):
            self.current_epoch = i
            # See warning in:
            # https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
            if self.use_tpu:
                self.train_loader._loader.sampler.set_epoch(i)
            else:
                self.train_loader.sampler.set_epoch(i)

            self.adjust_learning_rate()
            self.train_one_epoch()
            self.validate()
            self.save_checkpoint()

        self.close_db()

    def set_model_to_train(self):
        self.model.train()
        assert self.model.training

    def set_model_to_eval(self):
        self.model.eval()
        assert not self.model.training

    def close_db(self):
        if not self.use_mongodb:
            return
        if self.rank == 0:
            self.database.sync_with_host()

    def initialize_loss_function(self):
        """
        This function should return an instance of the loss function class.
        """
        raise NotImplementedError

    def initialize_dataloader(self):
        """
        This function should return a length-two tuple of PyTorch dataloaders,
        where the first loader is for the training set and the second loader
        is for the validation set.
        """
        raise NotImplementedError

    def initialize_optimizer(self):
        """
        This function should return a PyTorch optimizer object.
        """
        raise NotImplementedError

    def train_one_epoch(self):
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError

    def save_checkpoint(self):
        raise NotImplementedError

    def load_checkpoint(self):
        raise NotImplementedError

