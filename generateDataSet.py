import os

import package.config
from package.augment import Augment
from package.utils import get_Instance_List
import random
import numpy as np
import torch
import yaml
from typing import NamedTuple, Tuple
import cv2


def xyhw_2xyxy(box: Tuple[float, float, float, float], height: int, width: int) -> Tuple[float, float, float, float]:
    print(box)
    x_center_ratio, y_center_ratio, h_box_ratio, w_box_ratio = box

    # 计算 xyxy 格式的框
    x_center = x_center_ratio * width
    y_center = y_center_ratio * height
    h_box = h_box_ratio * height
    w_box = w_box_ratio * width

    x_min = max(0, x_center - w_box / 2)
    y_min = max(0, y_center - h_box / 2)
    x_max = min(width, x_center + w_box / 2)
    y_max = min(height, y_center + h_box / 2)

    return x_min, y_min, x_max, y_max


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


def visuallizeBox():
    filepath = r'D:\ProgramProject\PycharmProject\ObjectDetectionFinal\dataset\images\train'
    visualizepath = r'D:\ProgramProject\PycharmProject\ObjectDetectionFinal\visual'
    for root, subdir, files in os.walk(filepath):
        for file in files:
            imagepath = os.path.join(root, file)
            labelpath = imagepath.replace('images', 'labels').replace('.png', '.txt')
            with open(labelpath, 'r') as f:
                raw_boxes = f.readlines()
            image = cv2.imread(imagepath)
            for raw_box in raw_boxes:
                item = raw_box.split()
                box = tuple([float(i) for i in item[1:]])
                box_n = xyhw_2xyxy(box, image.shape[0], image.shape[1])

                # Convert coordinates to integers
                box_n = tuple(map(int, box_n))

                # Draw bounding box on the image
                cv2.rectangle(image, (box_n[0], box_n[1]), (box_n[2], box_n[3]), (0, 255, 0), 2)
            cv2.imwrite(os.path.join(visualizepath, file), image)
            print(file)


if __name__ == '__main__':
    set_seed()
    # generate_Dataset()
    # generate_yaml()

    visuallizeBox()
