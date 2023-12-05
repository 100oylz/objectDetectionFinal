from ultralytics import YOLO
import torch
import os
import cv2

modelpath = './best.pt'



class predict():
    def __init__(self):
        self.model = YOLO(modelpath, task='detect')


    def detect_image(self, image_path):
        image=cv2.imread(image_path)
        results = self.model(image, stream=True, device='cpu')
        boxresults = []
        for result in results:
            boxes = result.boxes  # Boxes object for bbox outputs
            for cls, conf, xyxy in zip(boxes.cls, boxes.conf, boxes.xyxy):
                # [ymin,xmin,ymax,xmax,conf,class]
                boxresults.append(
                    [xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item(), conf.item(), cls.item()])

        return boxresults


if __name__ == '__main__':
    model = predict()
    print(model.detect_image(
        r'D:\ProgramProject\PycharmProject\ObjectDetection_YOLO\dataset\images\val\0de4324225fe64cb236feca017916367.jpg'))
