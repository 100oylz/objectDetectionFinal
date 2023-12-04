import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple


class BasicClass(ABC):
    @classmethod
    @abstractmethod
    def apply(cls, image: np.ndarray, params: Dict):
        pass

    @classmethod
    @abstractmethod
    def getParams(cls):
        pass

    @classmethod
    @abstractmethod
    def changeBoxes(cls, boxes: List[Tuple[float, float, float, float]], info: Tuple[int, int], params: Dict):
        pass
