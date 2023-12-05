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
        w, h = info
        nw, nh = w * 1, h * 1
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, 1)
        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        # the move only affects the translation, so update the translation
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]

        # 矫正bbox坐标
        # rot_mat是最终的旋转矩阵
        # 获取原始bbox的四个中点，然后将这四个点转换到旋转后的坐标系下
        rot_bboxes = list()
        for bbox in boxes:
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[2]
            ymax = bbox[3]
            point1 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymin, 1]))
            point2 = np.dot(rot_mat, np.array([xmax, (ymin + ymax) / 2, 1]))
            point3 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymax, 1]))
            point4 = np.dot(rot_mat, np.array([xmin, (ymin + ymax) / 2, 1]))
            # 合并np.array
            concat = np.vstack((point1, point2, point3, point4))
            # 改变array类型
            concat = concat.astype(np.int32)
            # 得到旋转后的坐标
            rx, ry, rw, rh = cv2.boundingRect(concat)
            rx_min = rx
            ry_min = ry
            rx_max = rx + rw
            ry_max = ry + rh

            # 加入list中
            rot_bboxes.append([rx_min, ry_min, rx_max, ry_max])

        return rot_bboxes


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
