import numpy as np
import cv2
from package.augment._basicclass import BasicClass
from typing import List, Tuple, Dict

class BrightnessAdjustment(BasicClass):
    @classmethod
    def apply(cls, image: np.ndarray, params: Dict):
        brightness = params.get('brightness', 0)

        # Apply brightness adjustment
        adjusted_image = cv2.convertScaleAbs(image, alpha=1, beta=brightness)

        return adjusted_image

    @classmethod
    def getParams(cls):
        brightness = np.random.uniform(-50, 50)
        params = {'brightness': brightness}
        return params

    @classmethod
    def changeBoxes(cls, boxes: List[Tuple[float, float, float, float]], info: Tuple[int, int], params: Dict):
        # Brightness adjustment usually doesn't affect bounding boxes
        return boxes


class HueSaturationAdjustment(BasicClass):
    @classmethod
    def apply(cls, image: np.ndarray, params: Dict):
        hue = params.get('hue', 0)
        saturation = params.get('saturation', 1)

        # Convert image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Adjust hue and saturation
        hsv_image[:, :, 0] = (hsv_image[:, :, 0] + hue) % 180
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation, 0, 255)

        # Convert back to BGR color space
        adjusted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

        return adjusted_image

    @classmethod
    def getParams(cls):
        hue = np.random.uniform(-10, 10)
        saturation = np.random.uniform(0.5, 1.5)
        params = {'hue': hue, 'saturation': saturation}
        return params

    @classmethod
    def changeBoxes(cls, boxes: List[Tuple[float, float, float, float]], info: Tuple[int, int], params: Dict):
        # Hue and saturation adjustments might affect bounding boxes, but it's complex to handle
        # In practice, you may choose not to modify bounding boxes for simplicity
        return boxes


class ContrastAdjustment(BasicClass):
    @classmethod
    def apply(cls, image: np.ndarray, params: Dict):
        contrast = params.get('contrast', 1)

        # Apply contrast adjustment
        adjusted_image = cv2.convertScaleAbs(image, alpha=contrast, beta=0)

        return adjusted_image

    @classmethod
    def getParams(cls):
        contrast = np.random.uniform(0.5, 1.5)
        params = {'contrast': contrast}
        return params

    @classmethod
    def changeBoxes(cls, boxes: List[Tuple[float, float, float, float]], info: Tuple[int, int], params: Dict):
        # Contrast adjustment usually doesn't affect bounding boxes
        return boxes


class SaturationAdjustment(BasicClass):
    @classmethod
    def apply(cls, image: np.ndarray, params: Dict):
        saturation = params.get('saturation', 1)

        # Convert image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Adjust saturation
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation, 0, 255)

        # Convert back to BGR color space
        adjusted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

        return adjusted_image

    @classmethod
    def getParams(cls):
        saturation = np.random.uniform(0.5, 1.5)
        params = {'saturation': saturation}
        return params

    @classmethod
    def changeBoxes(cls, boxes: List[Tuple[float, float, float, float]], info: Tuple[int, int], params: Dict):
        # Saturation adjustment might affect bounding boxes, but it's complex to handle
        # In practice, you may choose not to modify bounding boxes for simplicity
        return boxes