from package.augment import Augment
from package.utils import get_Instance_List
import random
import numpy as np
import torch
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

if __name__ == '__main__':

    filepath = r'D:\ProgramProject\PycharmProject\ObjectDetectionFinal\detectImageFinal\data\data_train.txt'
    instance_list = get_Instance_List(filepath)
    method = Augment()
    print("Start!")
    method.main(instance_list, save_dirs=r'D:\ProgramProject\PycharmProject\ObjectDetectionFinal\dataset')
