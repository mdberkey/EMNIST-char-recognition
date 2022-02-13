#Importing the module required for the project

import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.datasets import EMNIST
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torch.utils.data import random_split
from torchvision.utils import make_grid
import matplotlib
import matplotlib.pyplot as plt
import pickle 
import requests
import zipfile
import os, shutil
from PIL import Image
%matplotlib inline


def main():
    matplotlib.rcParams['figure.facecolor'] = '#ffffff'

    # The project name
    project_name = 'emnist-char-recognition'