import os

import package.config
from package.augment import Augment
from package.utils import get_Instance_List
import random
import numpy as np
import torch
import yaml
from typing import NamedTuple


def set_seed():
    # 设置random库的种子
    random_seed = 42
    random.seed(random_seed)
    # 设置numpy库的种子
    np_seed = 42
    np.random.seed(np_seed)
    # 设置torch库的种子
    torch_seed = 42
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)


def generate_Dataset():
    filepath = r'D:\ProgramProject\PycharmProject\ObjectDetectionFinal\detectImageFinal\data\data_train.txt'
    instance_list = get_Instance_List(filepath)
    method = Augment()
    print("Start!")
    method.main(instance_list, save_dirs=r'D:\ProgramProject\PycharmProject\ObjectDetectionFinal\dataset')


def generate_yaml():
    filepath = r'D:\ProgramProject\PycharmProject\ObjectDetectionFinal\dataset'

    # Define a NamedTuple using the typing module
    class TrainData(NamedTuple):
        data: str
        train: str
        val: str
        test: str
        names: dict

    # Create an instance of the NamedTuple
    train_data_instance = TrainData(
        data=filepath,
        train='images/train',
        val='images/val',
        test=None,
        names={value: key for key, value in package.config.LABELMAP.items()}
    )

    with open(os.path.join(filepath, 'train_data.yaml'), 'w') as f:
        # Use _asdict() to convert the NamedTuple instance to a dictionary
        yaml.dump(train_data_instance._asdict(), f, default_flow_style=False)


if __name__ == '__main__':
    set_seed()
    # generate_Dataset()
    generate_yaml()
