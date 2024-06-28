from ultralytics import YOLO
import torch

# SETTINGS 
gpu_accel = True
model_path = 'models/yolov8l.pt'
print_verbose_info = False
minimum_confidence = 0.45



if gpu_accel:
    torch.cuda.set_device(0)
model = YOLO(model_path,task='detect') #pytorch weights almost 30 fps


#def pred(img):
#    results = model.predict(img,verbose=print_verbose_info,device=0,show=False,save=False)
#    result = results[0]
#    box = result.boxes
#    # capture screen and pass it thru mode 
#    if len(box) > 0:                 # if some detection
#        n = len(box) 
#        for i in range(n):
#            n = i
#            conf = box.conf[n].item()
#            #print(rl   
#            if conf > minimum_confidence :           # confidence is greater than threshold
#                coords = box.xyxy[n].tolist()
#                coords = [round(x) for x in coords]
#                cls = result.names[box.cls[n].item()]
#                xmin,ymin,xmax,ymax = coords[0],coords[1],coords[2],coords[3]
#                return xmin,ymin,xmax,ymax,cls
            

def pred(img):
    # Perform prediction using the model
    results = model.predict(img, verbose=print_verbose_info, device=0, show=False, save=False)
    result = results[0]
    box = result.boxes
    
    detections = []
    
    # Check if any detections are found
    if len(box) > 0:
        for i in range(len(box)):
            conf = box.conf[i].item()
            
            # If confidence is greater than the minimum threshold
            if conf > minimum_confidence:
                coords = box.xyxy[i].tolist()
                coords = [round(x) for x in coords]
                cls = result.names[box.cls[i].item()]
                xmin, ymin, xmax, ymax = coords[0], coords[1], coords[2], coords[3]
                
                # Append the detected object's details to the list
                detections.append((xmin, ymin, xmax, ymax, cls))
                
    # Return the list of detections
    return detections
