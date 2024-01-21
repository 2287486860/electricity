from ultralytics import YOLO
from clearml import Task
# task = Task.init(project_name="yolov8", task_name="测试新手")

# model = YOLO('yolov8n.pt')
# model.train(data='./ultralytics/cfg/models/v8/yolov8.yaml',epochs=2)

model  = YOLO("./ultralytics/cfg/models/v8/inception.yaml").train(**{'cfg':'ultralytics/cfg/default.yaml'},epochs=2,data='ultralytics/cfg/datasets/warn.yaml')
