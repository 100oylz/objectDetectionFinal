import numpy as np
import cv2
from package.augment._basicclass import BasicClass
from typing import List, Tuple, Dict

EXIST_PROB = 0.5
SCALE_PROB = 0.6


class TranslationTransformation(BasicClass):
    @classmethod
    def apply(cls, image: np.ndarray, params: Dict):
        tx, ty = params.get('tx', 0), params.get('ty', 0)
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        translated_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
        return translated_image

    @classmethod
    def getParams(cls):
        tx = np.random.uniform(-20, 20)
        ty = np.random.uniform(-20, 20)
        params = {'tx': tx, 'ty': ty}
        return params

    @classmethod
    def changeBoxes(cls, boxes: List[Tuple[float, float, float, float]], info: Tuple[int, int], params: Dict):
        tx, ty = params.get('tx', 0), params.get('ty', 0)
        newboxes = []
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            xmin_n, ymin_n, xmax_n, ymax_n = xmin + tx, ymin + ty, xmax + tx, ymax + ty
            xmin_n, ymin_n, xmax_n, ymax_n = np.clip(xmin_n, 0, info[0]), np.clip(ymin_n, 0, info[1]), np.clip(xmax_n,
                                                                                                               0, info[
                                                                                                                   0]), np.clip(
                ymax_n, 0, info[1])

            if (xmax_n - xmin_n) / (xmax - xmin) < EXIST_PROB:
                continue
            if (ymax_n - ymin_n) / (ymax - ymin) < EXIST_PROB:
                continue
            if (xmax_n - xmin_n) * (ymax_n - ymin_n) / (xmax - xmin) / (ymax - ymin) < EXIST_PROB * SCALE_PROB:
                continue
            newboxes.append((xmin_n, ymin_n, xmax_n, ymax_n))
        return newboxes


class RotationTransformation(BasicClass):
    @classmethod
    def apply(cls, image: np.ndarray, params: Dict):
        angle = params.get('angle', 0)
        rotation_matrix = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, 1)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
        return rotated_image

    @classmethod
    def getParams(cls):
        angle = np.random.uniform(-30, 30)
        params = {'angle': angle}
        return params

    @classmethod
    def changeBoxes(cls, boxes: List[Tuple[float, float, float, float]], info: Tuple[int, int], params: Dict):
        angle = params['angle']
        center_x, center_y = info[0] // 2, info[1] // 2

        # Convert angle to radians
        angle_rad = np.radians(angle)

        # Adjust bounding box coordinates after rotation
        rotated_boxes = []
        rotate_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]])
        for box in boxes:
            x_min, y_min, x_max, y_max = box
            xmin, ymin, xmax, ymax = box

            # Translate coordinates to center
            x_min -= center_x
            y_min -= center_y
            x_max -= center_x
            y_max -= center_y

            point1 = np.array([x_min, y_min])
            point2 = np.array([x_min, y_max])
            point3 = np.array([x_max, y_min])
            point4 = np.array([x_max, y_max])

            point1 = np.dot(rotate_matrix, point1)
            point2 = np.dot(rotate_matrix, point2)
            point3 = np.dot(rotate_matrix, point3)
            point4 = np.dot(rotate_matrix, point4)

            # Rotate coordinates
            x_min_rot = np.min([point1[0], point2[0], point3[0], point4[0]])
            y_min_rot = np.min([point1[1], point2[1], point3[1], point4[1]])
            x_max_rot = np.max([point1[0], point2[0], point3[0], point4[0]])
            y_max_rot = np.max([point1[1], point2[1], point3[1], point4[1]])

            # Translate coordinates back to original position
            x_min_rot += center_x
            y_min_rot += center_y
            x_max_rot += center_x
            y_max_rot += center_y

            x_min_rot, y_min_rot, x_max_rot, y_max_rot = np.clip(x_min_rot, 0, info[0]), np.clip(y_min_rot, 0,
                                                                                                 info[1]), np.clip(
                x_max_rot, 0, info[0]), np.clip(y_max_rot, 0, info[1])

            if (x_max_rot - x_min_rot) / (xmax - xmin) < EXIST_PROB:
                continue
            if (y_max_rot - y_min_rot) / (ymax - ymin) < EXIST_PROB:
                continue
            if (x_max_rot - x_min_rot) * (y_max_rot - y_min_rot) / (xmax - xmin) / (
                    ymax - ymin) < EXIST_PROB * SCALE_PROB:
                continue
            # Update rotated box coordinates
            rotated_boxes.append((x_min_rot, y_min_rot, x_max_rot, y_max_rot))

        return rotated_boxes


class ScalingTransformation(BasicClass):
    @classmethod
    def apply(cls, image: np.ndarray, params: Dict):
        scaling_factor = params.get('scaling_factor', 1)
        scaled_image = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor)
        return scaled_image

    @classmethod
    def getParams(cls):
        scaling_factor = np.random.uniform(0.8, 1.2)
        params = {'scaling_factor': scaling_factor}
        return params

    @classmethod
    def changeBoxes(cls, boxes: List[Tuple[float, float, float, float]], info: Tuple[int, int], params: Dict):
        scaling_factor = params.get('scaling_factor', 1)
        return [(box[0] * scaling_factor, box[1] * scaling_factor, box[2] * scaling_factor, box[3] * scaling_factor) for
                box in boxes]


class FlipTransformation(BasicClass):
    @classmethod
    def apply(cls, image: np.ndarray, params: Dict):
        flip_code = params.get('flip_code', 1)
        flipped_image = cv2.flip(image, flip_code)
        return flipped_image

    @classmethod
    def getParams(cls):
        flip_code = np.random.choice([1, 0, -1])
        params = {'flip_code': flip_code}
        return params

    @classmethod
    def changeBoxes(cls, boxes: List[Tuple[float, float, float, float]], info: Tuple[int, int], params: Dict):

        if params['flip_code'] == 1:
            # Handle horizontal flip (left-right flip)
            return [(info[0] - box[2], box[1], info[0] - box[0], box[3]) for box in boxes]
        elif params['flip_code'] == 0:
            # Handle vertical flip (up-down flip)
            return [(box[0], info[1] - box[3], box[2], info[1] - box[1]) for box in boxes]
        elif params['flip_code'] == -1:
            # Handle both vertical and horizontal flip
            return [(info[0] - box[2], info[1] - box[3], info[0] - box[0], info[1] - box[1]) for box in boxes]
        else:
            # Handle other cases if needed
            return boxes
