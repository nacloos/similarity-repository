import os
import json
import shutil
import random

import numpy as np
import regex as re

import torch
import torchvision.models
import torch.optim as optim
import torch.distributed as dist

from math import cos, pi

LATEST_CKPT_NAME = "checkpoint.pt"
BEST_CKPT_NAME = "model_best.pt"
EPOCH_CKPT_PREFIX = "checkpoint_epoch_"


#######################################################
# Configuration file parser
#######################################################


def parse_config(config_file):
    """
    Parses a json configuration file into a dictionary

    Inputs:
        config_file : (string) path to config file

    Outputs:
        config      : (dict) dictionary where key is parameter name and value
                      is the desired parameter value
    """
    with open(config_file, "r") as f:
        config = json.load(f)
        return config


#######################################################
# Logging metrics for each epoch in training
#######################################################


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


#######################################################
# Metrics and other utility funcs
#######################################################


def compute_accuracy(output, target, topk=(1,)):
    """Adapted from PyTorch tutorial."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res


def reduce_metric(metric, world_size):
    """
    Reduces a metric across all processes.

    Inputs:
        metric     : (torch.Tensor) can be loss, top-1 accuracy, etc.
        world_size : (int) number of processes (i.e., GPUs)
    """
    with torch.no_grad():
        avg_metric = metric / world_size
        dist.all_reduce(avg_metric)
        return avg_metric


def get_save_checkpoint_path(save_dir, is_best=False, save_epoch=None):
    fname = LATEST_CKPT_NAME

    if is_best:
        assert save_epoch is None
        fname = BEST_CKPT_NAME
    elif save_epoch is not None:
        fname = EPOCH_CKPT_PREFIX + "{}.pt".format(save_epoch)

    fname = os.path.join(save_dir, fname)
    return fname


def save_checkpoint(state, save_dir, is_best, save_epoch, rank=0, tpu=False):
    fname = get_save_checkpoint_path(save_dir=save_dir)
    if tpu:
        import torch_xla.core.xla_model as xm
        xm.save(state, fname)
    else:
        if rank == 0:
            # All processes should see same parameters as they all start from same
            # random parameters and gradients are synchronized in backward passes.
            # Therefore, saving it in one process is sufficient.
            torch.save(state, fname)

    if tpu and (rank == 0):
        # save checkpoint to gs bucket
        from mouse_vision.model_training.gcloud_utils import save_file_to_bucket
        save_file_to_bucket(
            fname, autoremove=False
        )  # do not remove yet, in case needed below

    if is_best:
        best_ckpt_path = get_save_checkpoint_path(save_dir=save_dir, is_best=is_best)
        if rank == 0:
            shutil.copyfile(fname, best_ckpt_path)
            if tpu:
                save_file_to_bucket(best_ckpt_path)

    if save_epoch is not None:
        save_epoch_ckpt_path = get_save_checkpoint_path(
            save_dir=save_dir, save_epoch=save_epoch
        )
        if rank == 0:
            shutil.copyfile(fname, save_epoch_ckpt_path)
            if tpu:
                save_file_to_bucket(save_epoch_ckpt_path)

    if tpu and (rank == 0):
        # remove local copy since we have uploaded to gs bucket now
        os.remove(fname)


def construct_save_dir(save_prefix, config):
    # Save model and stats directory.
    save_dir = os.path.join(
        save_prefix,
        "{}/{}/{}".format(config["db_name"], config["coll_name"], config["exp_id"]),
    )
    return save_dir


def get_resume_checkpoint_path(save_dir, resume_epoch="last"):
    """WARNING: IF FILENAME CONVENTIONS EVER CHANGE, BE SURE TO CHANGE THIS FUNCTION"""
    resume_path = ""
    if resume_epoch == "last":
        resume_path = get_save_checkpoint_path(save_dir=save_dir)
    elif resume_epoch == "best":
        resume_path = get_save_checkpoint_path(save_dir=save_dir, is_best=True)
    else:
        resume_path = get_save_checkpoint_path(
            save_dir=save_dir, save_epoch=resume_epoch
        )
    return resume_path


def check_best_accuracy(current_acc, best_acc):
    if current_acc > best_acc:
        return current_acc, True
    return best_acc, False


def check_best_loss(current_loss, best_loss):
    if current_loss < best_loss:
        return current_loss, True
    return best_loss, False


def build_paramwise_options(named_parameters,
                            paramwise_options):
    # Adapted from: https://github.com/open-mmlab/OpenSelfSup/blob/ed5000482b0d8b816cd8a6fbbb1f97da44916fed/openselfsup/apis/train.py#L144-L166
    assert isinstance(paramwise_options, dict)
    params = []
    for name, param in named_parameters:
        param_group = {'params': [param]}
        if not param.requires_grad:
            params.append(param_group)
            continue

        for regexp, options in paramwise_options.items():
            if re.search(regexp, name):
                for key, value in options.items():
                    param_group[key] = value

        # otherwise use the global settings
        params.append(param_group)

    return params


def annealing_cos(start, end, factor, weight=1):
    """Calculate annealing cos learning rate.
    Taken from: https://github.com/open-mmlab/mmcv/blob/bcf85026c3f2683212d2a3a25f58102b3e5f75ef/mmcv/runner/hooks/lr_updater.py#L401-L416
    Cosine anneal from `weight * start + (1 - weight) * end` to `end` as
    percentage goes from 0.0 to 1.0.
    Args:
        start (float): The starting learning rate of the cosine annealing.
        end (float): The ending learing rate of the cosine annealing.
        factor (float): The coefficient of `pi` when calculating the current
            percentage. Range from 0.0 to 1.0.
        weight (float, optional): The combination factor of `start` and `end`
            when calculating the actual starting learning rate. Default to 1.
    """
    cos_out = cos(pi * factor) + 1
    return end + 0.5 * weight * (start - end) * cos_out


if __name__ == "__main__":

    # save_checkpoint({"test":23}, "./", True, 3)

    # Check best accuracy test
    print(check_best_accuracy(0.8, 0.7))  # (0.8, True)
    print(check_best_accuracy(0.7, 0.8))  # (0.8, False)
