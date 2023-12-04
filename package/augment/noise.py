from typing import Dict, List, Tuple

import numpy as np
from package.augment._basicclass import BasicClass
import cv2


class GaussianNoise(BasicClass):
    @classmethod
    def apply(cls, image: np.ndarray, params: Dict):
        # 添加高斯噪声
        mean = params.get('mean', 0)
        std_dev = params.get('std_dev', 1)
        noise = np.random.normal(mean, std_dev, image.shape)
        noisy_image = image + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    @classmethod
    def getParams(cls):
        # 获取高斯噪声的参数
        mean = np.random.uniform(-1, 1)
        std_dev = np.random.uniform(0, 2)

        params = {'mean': mean, 'std_dev': std_dev}
        return params

    @classmethod
    def changeBoxes(cls, boxes: List[Tuple[float, float, float, float]], info: Tuple[int, int], params: Dict):
        return boxes


class SaltPepperNoise(BasicClass):
    @classmethod
    def apply(cls, image: np.ndarray, params: Dict):
        # 添加椒盐噪声
        salt_prob = params.get('salt_prob', 0.01)
        pepper_prob = params.get('pepper_prob', 0.01)

        noisy_image = np.copy(image)

        # 添加盐噪声
        salt_mask = np.random.random(image.shape) < salt_prob
        noisy_image[salt_mask] = 255

        # 添加椒噪声
        pepper_mask = np.random.random(image.shape) < pepper_prob
        noisy_image[pepper_mask] = 0

        return noisy_image.astype(np.uint8)

    @classmethod
    def getParams(cls):
        # 获取椒盐噪声的参数
        salt_prob = np.random.uniform(0, 0.02)
        pepper_prob = np.random.uniform(0, 0.02)
        params = {'salt_prob': salt_prob, 'pepper_prob': pepper_prob}
        return params

    @classmethod
    def changeBoxes(cls, boxes: List[Tuple[float, float, float, float]], info: Tuple[int, int], params: Dict):
        # 对于椒盐噪声，不进行位置变换，直接返回原始框
        return boxes


class UniformNoise(BasicClass):
    @classmethod
    def apply(cls, image: np.ndarray, params: Dict):
        # 添加均匀噪声
        low = params.get('low', 0)
        high = params.get('high', 255)
        noise = np.random.uniform(low, high, image.shape)
        noisy_image = image + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    @classmethod
    def getParams(cls):
        # 获取均匀噪声的参数
        low = np.random.uniform(0, 5)
        high = np.random.uniform(250, 255)
        params = {'low': low, 'high': high}
        return params

    @classmethod
    def changeBoxes(cls, boxes: List[Tuple[float, float, float, float]], info: Tuple[int, int], params: Dict):
        # 不进行框的变化
        return boxes


class GaussianMixtureNoise(BasicClass):
    """
    高斯混合噪声类，模拟由多个高斯分布组合而成的图像噪声。
    """

    @classmethod
    def apply(cls, image: np.ndarray, params: Dict):
        num_components = params.get('num_components', 3)
        weights = params.get('weights', np.ones(num_components) / num_components)
        means = params.get('means', np.random.uniform(0, 255, num_components))
        std_devs = params.get('std_devs', np.random.uniform(1, 20, num_components))

        noise = np.zeros_like(image, dtype=np.float32)

        for i in range(num_components):
            component_noise = np.random.normal(means[i], std_devs[i], image.shape) * weights[i]
            noise += component_noise

        noisy_image = image + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    @classmethod
    def getParams(cls):
        num_components = np.random.randint(2, 4)
        weights = np.random.dirichlet(np.ones(num_components))
        means = np.random.uniform(-1, 1, num_components)
        std_devs = np.random.uniform(0, 2, num_components)

        params = {'num_components': num_components, 'weights': weights, 'means': means, 'std_devs': std_devs}
        return params

    @classmethod
    def changeBoxes(cls, boxes: List[Tuple[float, float, float, float]], info: Tuple[int, int], params: Dict):
        return boxes


class ImpulseNoise(BasicClass):
    """
    脉冲噪声类，模拟图像中的盐噪声和椒噪声。
    """

    @classmethod
    def apply(cls, image: np.ndarray, params: Dict):
        salt_prob = params.get('salt_prob', 0.01)
        pepper_prob = params.get('pepper_prob', 0.01)

        noisy_image = np.copy(image)

        # 添加盐噪声
        salt_mask = np.random.random(image.shape) < salt_prob
        noisy_image[salt_mask] = 255

        # 添加椒噪声
        pepper_mask = np.random.random(image.shape) < pepper_prob
        noisy_image[pepper_mask] = 0

        return noisy_image.astype(np.uint8)

    @classmethod
    def getParams(cls):
        salt_prob = np.random.uniform(0, 0.02)
        pepper_prob = np.random.uniform(0, 0.02)
        params = {'salt_prob': salt_prob, 'pepper_prob': pepper_prob}
        return params

    @classmethod
    def changeBoxes(cls, boxes: List[Tuple[float, float, float, float]], info: Tuple[int, int], params: Dict):
        # 对于脉冲噪声，不进行位置变换，直接返回原始框
        return boxes


class PeriodicNoise(BasicClass):
    """
    周期性噪声类，模拟图像中的周期性噪声。
    """

    @classmethod
    def apply(cls, image: np.ndarray, params: Dict):
        frequency = params.get('frequency', 0.1)
        amplitude = params.get('amplitude', 50)

        rows, cols, channels = image.shape
        x = np.arange(cols)
        y = np.arange(rows)

        X, Y = np.meshgrid(x, y)

        # Generate independent noise for each channel
        noise_r = amplitude * np.sin(2 * np.pi * frequency * X / cols)
        noise_g = amplitude * np.sin(2 * np.pi * frequency * X / cols)
        noise_b = amplitude * np.sin(2 * np.pi * frequency * X / cols)

        noisy_image = image + np.stack([noise_r, noise_g, noise_b], axis=-1)
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    @classmethod
    def getParams(cls):
        frequency = np.random.uniform(0.05, 0.2)
        amplitude = np.random.uniform(2, 5)
        params = {'frequency': frequency, 'amplitude': amplitude}
        return params

    @classmethod
    def changeBoxes(cls, boxes: List[Tuple[float, float, float, float]], info: Tuple[int, int], params: Dict):
        # 对于周期性噪声，不进行位置变换，直接返回原始框
        return boxes


class BlurNoise(BasicClass):
    """
    模糊噪声类，模拟图像中的模糊效果。
    """

    @classmethod
    def apply(cls, image: np.ndarray, params: Dict):
        kernel_size = params.get('kernel_size', 5)

        noisy_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return noisy_image

    @classmethod
    def getParams(cls):
        kernel_size = np.random.choice([3, 5, 7])
        params = {'kernel_size': kernel_size}
        return params

    @classmethod
    def changeBoxes(cls, boxes: List[Tuple[float, float, float, float]], info: Tuple[int, int], params: Dict):
        # 对于模糊噪声，不进行位置变换，直接返回原始框
        return boxes
