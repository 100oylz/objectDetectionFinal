import random
from typing import Tuple, List, Dict
from copy import deepcopy
from sklearn.model_selection import train_test_split
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2
import concurrent
import gc
from package.augment.noise import GaussianNoise, GaussianMixtureNoise, SaltPepperNoise, BlurNoise, \
    UniformNoise, PeriodicNoise
from package.augment.affine import ScalingTransformation, RotationTransformation, FlipTransformation, \
    TranslationTransformation
from package.augment.pixel import BrightnessAdjustment, ContrastAdjustment, SaturationAdjustment, \
    HueSaturationAdjustment
from package.utils.instance import Instance
from package.augment._basicclass import BasicClass

AUGMENTCLASSES: Tuple[BasicClass, ...] = (
    GaussianNoise, GaussianMixtureNoise, SaltPepperNoise, BlurNoise,
    UniformNoise, PeriodicNoise, ScalingTransformation,
    FlipTransformation,
    TranslationTransformation, BrightnessAdjustment, ContrastAdjustment, SaturationAdjustment,
    HueSaturationAdjustment)
FORMATIDDICT: Dict[int, BasicClass] = {(1 << i): AUGMENTCLASSES[i] for i in range(len(AUGMENTCLASSES))}
AUGMENTNOISE: Tuple[BasicClass, ...] = (
    GaussianNoise, GaussianMixtureNoise, SaltPepperNoise, BlurNoise,
    UniformNoise, PeriodicNoise
)


def process_instance(instance, save_image_dirs, save_label_dirs):
    format_id = instance.get_id()
    save_image_path = save_image_dirs / f'{format_id}.png'
    save_label_path = save_label_dirs / f'{format_id}.txt'
    cv2.imwrite(str(save_image_path), instance.data)
    boxes_string = instance.to_str()
    with open(str(save_label_path), 'w') as f:
        f.writelines(boxes_string)
    print(f'{format_id} Saved!')
    gc.collect()


class Augment():

    def __init__(self):
        self.class_length = len(AUGMENTCLASSES)
        self.noise_length = len(AUGMENTNOISE)
        self.ensure_max_id = 1 << (self.class_length + 1)
        self.noise_min_id = 1 << (self.noise_length + 1)
        self.noise_list = [i for i in range(self.noise_length)]
        self.other_list = [i for i in range(self.noise_length + 1, self.class_length)]

    def get_random_format_id(self):
        id_list = []
        id_list.append(random.choice(self.noise_list))
        k = random.randint(3, 5)
        id_list.extend(random.sample(self.other_list, k))
        format_id = 0
        for id in id_list:
            format_id += 1 << id
        return format_id

    def sparse_format_id(self, format_id) -> Tuple[List[BasicClass], List[int]]:
        augment_func_list = []
        id_list = []
        for i in range(self.class_length - 1, -1, -1):
            id = 1 << i
            if (format_id < id):
                continue
            else:
                format_id -= id
                id_list.append(id)
                augment_func_list.append(FORMATIDDICT[id])
        return augment_func_list, id_list

    def apply(self, instance: Instance):
        format_id = self.get_random_format_id()
        augment_func_list, id_list = self.sparse_format_id(format_id)
        newinstance = deepcopy(instance)

        for augment_func in augment_func_list:
            image = newinstance.data
            params = augment_func.getParams()
            newinstance.data = augment_func.apply(image, params)
            newinstance.boxes = augment_func.changeBoxes(newinstance.boxes, newinstance.get_hw(), params)
            newinstance.format_id = format_id
        return newinstance

    def randomSample(self, basic_instance_list: List[Instance], min_num: int, max_num: int):
        assert min_num < max_num

        changed_instance_list = []
        format_id_set_list = []

        def process_instance(i, basic_instance):
            num = random.randint(min_num, max_num)
            format_id_set = set()
            for _ in range(num):
                changed_instance = self.apply(basic_instance)
                while changed_instance.format_id in format_id_set:
                    changed_instance = self.apply(basic_instance)
                changed_instance_list.append(changed_instance)
                format_id_set.add(changed_instance.format_id)

            format_id_set_list.append(format_id_set)
            print(i)

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_instance, i, basic_instance)
                       for i, basic_instance in enumerate(basic_instance_list)]

            # 等待所有任务完成
            concurrent.futures.wait(futures)

        return changed_instance_list, format_id_set_list

    def main(self, basic_instance_list: List[Instance], save_dirs: str):

        train_instance_list, valid_instance_list = train_test_split(basic_instance_list, test_size=0.2, shuffle=True,
                                                                    stratify=[i.cls for i in basic_instance_list])
        save_dirs = Path(save_dirs)
        save_train_image_dirs = save_dirs / 'images' / 'train'
        save_train_label_dirs = save_dirs / 'labels' / 'train'
        save_valid_image_dirs = save_dirs / 'images' / 'val'
        save_valid_label_dirs = save_dirs / 'labels' / 'val'

        changed_instance_list, changed_format_id_set_list = self.randomSample(train_instance_list, 1, 2)
        print(changed_format_id_set_list)
        gc.collect()
        # 使用 ThreadPoolExecutor 进行并行计算
        with ThreadPoolExecutor() as executor:
            # 并行处理验证集
            executor.map(lambda instance: process_instance(instance, save_valid_image_dirs, save_valid_label_dirs),
                         valid_instance_list)

            # 并行处理训练集和增强后的实例
            executor.map(lambda instance: process_instance(instance, save_train_image_dirs, save_train_label_dirs),
                         train_instance_list + changed_instance_list)

        print('Finished!')
