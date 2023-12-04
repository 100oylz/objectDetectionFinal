from dataclasses import dataclass
import numpy as np
from typing import List, Tuple
from pathlib import Path

FORMAT = '{} {} {} {} {}\n'
IDFORMAT = '{}_{}'


def xyxy_xyhw(box: Tuple[float, float, float, float], info: Tuple[int, int]):
    x_min, y_min, x_max, y_max = box
    height, width = info
    x_center, y_center = (x_min + x_max) / 2 / width, (y_min + y_max) / 2 / height
    ins_height, ins_width = (y_max - y_min) / height, (x_max - x_min) / width
    return x_center, y_center, ins_height, ins_width


@dataclass
class Instance():
    filepath: Path
    basic_id: str
    format_id: int
    data: np.ndarray
    cls: int
    boxes: List[Tuple[float, float, float, float]]

    def to_str(self):
        str_list = []
        for box in self.boxes:
            assert len(box) == 4
            str_list.append(FORMAT.format(self.cls, *xyxy_xyhw(box, self.get_hw())))
        return str_list

    def get_hw(self):
        return self.data.shape[1], self.data.shape[0]

    def get_id(self):
        return IDFORMAT.format(self.basic_id, self.format_id)
