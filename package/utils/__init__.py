import cv2
import numpy as np
import pandas as pd
import matplotlib as plt
import os
from typing import Tuple, List, Union, Optional
from pathlib import Path

from package.utils.instance import Instance
from package.config import LABELMAP


def get_Instance(data: str, basicpath: Path) -> Instance:
    item = data.split()
    filepath = basicpath / 'image' / item[0]
    temp_cls, temp_id = item[0].split('/')
    temp_id = temp_id.split('.')[0]
    if (temp_cls == 'ships'):
        temp_cls = 'ship'
    cls = LABELMAP[temp_cls]
    basic_id = f'{temp_cls}_{temp_id}'
    format_id = 0
    # print(filepath.__str__())
    data = cv2.imread(filepath.__str__())
    temp_boxes = item[1:]
    boxes = []
    for box in temp_boxes:
        bbox = box.split(',')
        assert len(bbox) == 5
        assert bbox[-1] == temp_cls, f'{bbox[-1]}!={temp_cls}'
        fbox = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        boxes.append(fbox)
    ins = Instance(
        filepath=Path(filepath),
        basic_id=basic_id,
        format_id=format_id,
        cls=cls,
        data=data,
        boxes=boxes
    )
    return ins


def get_Instance_List(filepath: str) -> List[Instance]:
    basicpath = Path(os.path.abspath(os.path.dirname(filepath)))
    with open(filepath, 'r') as f:
        data: List[str] = f.readlines()
    basic_Instance_List: List[Instance] = []

    for i, item in enumerate(data):
        ins = get_Instance(item, basicpath)
        basic_Instance_List.append(ins)
        print(i)
    return basic_Instance_List


if __name__ == '__main__':
    filepath = r'D:\ProgramProject\PycharmProject\ObjectDetectionFinal\detectImageFinal\data\data_train.txt'
    get_Instance_List(filepath)
