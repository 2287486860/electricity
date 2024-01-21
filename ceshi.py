from ultralytics import YOLO

# %matplotlib inline
yolo = YOLO(r"C:\Users\田海阳\Desktop\best.pt", task="detect")

result = yolo(source=r'D:\pycharm\work\clip\ultralytics-main\datasets\warn\test\images\16a.jpg',save=True,show=True)
# from ultralytics import YOLO
# from ultralytics.nn import modul
# # Load a model
# model = YOLO("yolov8n.pt")  # load an official model
# model = YOLO(r"C:\Users\田海阳\Desktop\yolov8.pt")  # load a custom model
#
# # Predict with the model
# results = model(r'D:\pycharm\work\clip\ultralytics-main\datasets\warn\test\images\16a.jpg',show=True)  # predict on an image




