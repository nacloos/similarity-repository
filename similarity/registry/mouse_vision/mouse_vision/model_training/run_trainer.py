import os
import torch.distributed as dist

import torch
# WARNING: only uncomment the line below for debugging purposes
#          as it slows down training significantly, especially on TPU
#torch.autograd.set_detect_anomaly(True)

from mouse_vision.core.default_dirs import MODEL_SAVE_DIR
from mouse_vision.model_training.train_utils import (
    construct_save_dir,
    get_resume_checkpoint_path,
    parse_config,
)
from mouse_vision.model_training.supervised_cifar10_trainer import (
    SupervisedCIFAR10Trainer,
)
from mouse_vision.model_training.supervised_imagenet_trainer import (
    SupervisedImageNetTrainer,
)
from mouse_vision.model_training.instance_discrimination_trainer import (
    InstanceDiscriminationTrainer, DMLocomotionInstanceDiscriminationTrainer,
)
from mouse_vision.model_training.simclr_trainer import (
    SimCLRTrainer,
)
from mouse_vision.model_training.relative_location_trainer import (
    RelativeLocationTrainer,
)
from mouse_vision.model_training.mocov2_trainer import (
    MoCov2Trainer,
)
from mouse_vision.model_training.finetune_trainer import (
    FinetuneImageNetTrainer,
)
from mouse_vision.model_training.rotnet_trainer import (
    RotNetTrainer,
)
from mouse_vision.model_training.simsiam_trainer import (
    SimSiamTrainer,
)
from mouse_vision.model_training.autoencoder_trainer import (
    AutoEncoderTrainer,
)
from mouse_vision.model_training.depth_prediction_trainer import (
    DepthPredictionTrainer,
)

def train(config_file):
    # we use normal print functions here, as a further
    # assurance/sanity check that the number of print statements
    # is the number of GPUs/TPU cores
    if config_file["trainer"] == "SupervisedCIFAR10":
        print("Using Supervised CIFAR10 Trainer")
        trainer = SupervisedCIFAR10Trainer(config_file)
    elif config_file["trainer"] == "SupervisedImageNet":
        print("Using Supervised ImageNet Trainer")
        trainer = SupervisedImageNetTrainer(config_file)
    elif config_file["trainer"] == "InstanceDiscrimination":
        print("Using Instance Discrimination Trainer")
        trainer = InstanceDiscriminationTrainer(config_file)
    elif config_file["trainer"] == "DMLocomotionInstanceDiscriminationTrainer":
        print("Using DMLocomotion Instance Discrimination Trainer")
        trainer = DMLocomotionInstanceDiscriminationTrainer(config_file)
    elif config_file["trainer"] == "SimCLR":
        print("Using SimCLR Trainer")
        trainer = SimCLRTrainer(config_file)
    elif config_file["trainer"] == "RelativeLocation":
        print("Using Relative Location Trainer")
        trainer = RelativeLocationTrainer(config_file)
    elif config_file["trainer"] == "MoCov2":
        print("Using MoCov2 Trainer")
        trainer = MoCov2Trainer(config_file)
    elif config_file["trainer"] == "FinetuneImageNet":
        print("Using Finetune ImageNet Trainer")
        trainer = FinetuneImageNetTrainer(config_file)
    elif config_file["trainer"] == "RotNet":
        print("Using RotNet Trainer")
        trainer = RotNetTrainer(config_file)
    elif config_file["trainer"] == "SimSiam":
        print("Using SimSiam Trainer")
        trainer = SimSiamTrainer(config_file)
    elif config_file["trainer"] == "AutoEncoder":
        print("Using AutoEncoder Trainer")
        trainer = AutoEncoderTrainer(config_file)
    elif config_file["trainer"] == "DepthPrediction":
        print("Using DepthPrediction Trainer")
        trainer = DepthPredictionTrainer(config_file)
    else:
        raise ValueError("Invalid task.")

    trainer.train()

def gpu_train(rank, args):
    # overwrite config file with the gpu id that process is on
    assert(isinstance(args, dict))
    args["gpus"] = [args["gpus"][rank]]
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args["world_size"],
        rank=rank
    )
    train(args)

def tpu_train(rank, args):
    train(args)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--resume-epoch", type=str, default=None)
    ARGS = parser.parse_args()

    config_file = ARGS.config
    if not os.path.isfile(config_file):
        raise ValueError(f"{config_file} is invalid.")
    else:
        config_file = parse_config(config_file)
        config_file["filepath"] = ARGS.config

    if "save_dir" not in config_file.keys():
        if "save_prefix" not in config_file.keys():
            config_file["save_prefix"] = MODEL_SAVE_DIR
        config_file["save_dir"] = construct_save_dir(
            save_prefix=config_file["save_prefix"], config=config_file
        )

    if ARGS.resume_epoch is not None:
        config_file["resume_checkpoint"] = get_resume_checkpoint_path(
            save_dir=config_file["save_dir"], resume_epoch=ARGS.resume_epoch
        )

    tpu = config_file["tpu"]
    if tpu:
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.xla_multiprocessing as xmp
        from mouse_vision.model_training.gcloud_utils import configure_tpu

        configure_tpu(tpu)

        xmp.spawn(tpu_train,
                  args=(config_file,),
                  nprocs=None,
                  start_method="fork")
    else:
        import torch.multiprocessing as mp

        # configure address and port to listen to
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = f"{config_file.get('ddp_port', '8888')}"

        # determine world size
        if not isinstance(config_file["gpus"], list):
            config_file["gpus"] = [config_file["gpus"]]
        # dist.get_world_size() does not work since it is yet to be initialized
        config_file["world_size"] = len(config_file["gpus"])

        mp.spawn(gpu_train,
                 args=(config_file,),
                 nprocs=config_file["world_size"])

