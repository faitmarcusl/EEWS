from ultralytics import YOLO
import torch

# SETTINGS 
gpu_accel = True
model_path1 = 'models/yolov8s.pt'
model_path2 = 'models/yolov8n.pt'
print_verbose_info = False
minimum_confidence = 0.45

# Initialize both YOLO models
if gpu_accel:
    torch.cuda.set_device(0)
model1 = YOLO(model_path1, task='detect')
model2 = YOLO(model_path2, task='detect')

def set_mode(new_mode):
    global mode
    mode = new_mode
    global model
    if model == 0: 
        model = model1
    elif model == 1:
        model = model2

def predict_with_model(model, img):
    results = model.predict(img, verbose=print_verbose_info, device=0, show=False, save=False)
    result = results[0]
    box = result.boxes
    
    detections = []
    
    if len(box) > 0:
        for i in range(len(box)):
            conf = box.conf[i].item()
            
            if conf > minimum_confidence:
                coords = box.xyxy[i].tolist()
                coords = [round(x) for x in coords]
                cls = result.names[box.cls[i].item()]
                xmin, ymin, xmax, ymax = coords[0], coords[1], coords[2], coords[3]
                
                detections.append((xmin, ymin, xmax, ymax, cls))
    
    return detections

def pred(img):
    # Predict using the first model
    detections1 = predict_with_model(model1, img)
    
    # Predict using the second model
    detections2 = predict_with_model(model2, img)
    
    # Combine detections from both models
    combined_detections = detections1 + detections2
    
    return combined_detections