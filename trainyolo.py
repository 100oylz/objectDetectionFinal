from ultralytics import YOLO
import time

parameters_dict = {
    'data': 'dataset/train_data.yaml',
    'epochs': 300,
    'imgsz': 640,
    'batch': -1,
    'device': 0,
    'resume': True,
    'name': 'train',
    'project': 'detect',
    'verbose': True,
    'optimizer': 'NAdam',
    'seed': int(time.time()),
    'amp': False,
    'weight_decay': 5e-4,
    'box': 8.0,
    'cls': 0.5,
    'dfl': 2.0,
    'save_period': 50
}

if __name__ == '__main__':
    model = YOLO('cfg/yolov8n.yaml').load('./preTrainedModel/yolov8n.pt')
    results = model.train(**parameters_dict)
