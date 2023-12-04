from package.augment import Augment
from package.utils import get_Instance_List

if __name__ == '__main__':
    filepath = r'D:\ProgramProject\PycharmProject\ObjectDetectionFinal\detectImageFinal\data\data_train.txt'
    instance_list = get_Instance_List(filepath)
    method = Augment()
    print("Start!")
    method.main(instance_list, save_dirs=r'D:\ProgramProject\PycharmProject\ObjectDetectionFinal\dataset')
