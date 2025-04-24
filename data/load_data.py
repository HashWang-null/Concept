import os
from os import path as osp
from glob import glob
from pathlib import Path


def interface(train=True):
    return {
        "source": list(),
        "target": list(),
    }

