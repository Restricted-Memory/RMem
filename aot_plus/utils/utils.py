import sys
import datetime
import glob
import os
from shutil import copy, copytree
import numpy as np
import random
import torch
from torch import nn
from pathlib import Path


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get(self):
        return self.avg


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def get_format_time():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")


def make_log_dir(log_dir: str, name: str):
    new_log_dir = os.path.join(
        log_dir, f"{get_format_time()}_{name}")
    if os.path.isdir(new_log_dir):
        raise Exception(f"{new_log_dir} already exist ! abort ...")
    os.makedirs(new_log_dir)
    return new_log_dir


IGNORE_FOLDER_LIST = [
    '__pycache__', 'dataset', 'datasets', 'logs', 'debug_logs', 'eval_logs', 'outputs', 'results']
INCLUDE_EXTENSION_LIST = ['.py', '.sh']


def ignore_non_py_file(dir, file_list):
    ignore_list = []
    for file in file_list:
        if os.path.isdir(
                os.path.join(dir, file)):
            if file in IGNORE_FOLDER_LIST:
                ignore_list.append(file)
        else:
            if Path(file).suffix not in INCLUDE_EXTENSION_LIST:
                ignore_list.append(file)
    return ignore_list


def copy_codes(log_dir, ignore_function=ignore_non_py_file):
    code_dir = os.path.join(log_dir, "code")
    if os.path.isdir(code_dir):
        raise Exception(f"{code_dir} already exist ! abort ...")
    copytree(".", code_dir, ignore=ignore_function)


class Tee(object):
    def __init__(self, filename):
        self.file_name = filename
        with open(self.file_name, "w") as f:
            pass
        self.stdout = sys.stdout

    def close(self):
        sys.stdout = self.stdout

    def write(self, data):
        with open(self.file_name, "a") as f:
            f.write(data)
        self.stdout.write(data)

    def flush(self):
        self.stdout.flush()


def save_model(model: nn.Module, save_dir: str, suffice: str):
    model_path = os.path.join(
        save_dir, f"{type(model).__name__}_{suffice}.model")
    print(f"saving model to {model_path}")
    torch.save(
        model.state_dict(),
        model_path)


def load_model(model: nn.Module, load_dir: str, suffice: str, map_location=None):
    model_path = os.path.join(
        load_dir, f"{type(model).__name__}_{suffice}.model")
    print(f"loading model from {model_path}")
    model.load_state_dict(
        torch.load(model_path, map_location=map_location))
