import os
from roboflow import Roboflow

# Configuration
API_KEY = "P5W2awqHgQZgmiKPJIu7"
PROJECTS_AND_VERSIONS = [
    {"project_name": "masid", "version_number": 1},  # Residential mode model
    {"project_name": "people-detection-o4rdr", "version_number": 7},  # Commercial mode model 1
    {"project_name": "gempa", "version_number": 11},  # Commercial mode model 2
]
MINIMUM_CONFIDENCE = 0.45

# Initialize Roboflow models
rf = Roboflow(api_key=API_KEY)
models = [
    rf.workspace().project(pv["project_name"]).version(pv["version_number"]).model
    for pv in PROJECTS_AND_VERSIONS
]

# Separate models for residential and commercial modes
residential_model = models[0]
commercial_models = models[1:]

mode = 0  # Default mode is residential

def set_mode(new_mode):
    global mode
    mode = new_mode

def pred(image):
    """
    Perform prediction on the given image using the appropriate model based on the mode.

    Args:
        image (np.ndarray): The image to perform detection on.

    Returns:
        list: A list of tuples, each containing (xmin, ymin, xmax, ymax, class) for detections
              with confidence above the minimum threshold.
    """
    detections = []
    if mode == 0:
        # Residential mode using masid model
        try:
            response = residential_model.predict(image).json()
            predictions = response['predictions']
            for pred in predictions:
                conf = pred['confidence']
                if conf > MINIMUM_CONFIDENCE:
                    xmin = pred['x'] - pred['width'] / 2
                    ymin = pred['y'] - pred['height'] / 2
                    xmax = pred['x'] + pred['width'] / 2
                    ymax = pred['y'] + pred['height'] / 2
                    cls = pred['class']
                    detections.append((round(xmin), round(ymin), round(xmax), round(ymax), cls))
        except Exception as e:
            print(f"Error during prediction with residential model: {e}")
    elif mode == 1:
        # Commercial mode using both commercial models
        for model in commercial_models:
            try:
                response = model.predict(image).json()
                predictions = response['predictions']
                for pred in predictions:
                    conf = pred['confidence']
                    if conf > MINIMUM_CONFIDENCE:
                        xmin = pred['x'] - pred['width'] / 2
                        ymin = pred['y'] - pred['height'] / 2
                        xmax = pred['x'] + pred['width'] / 2
                        ymax = pred['y'] + pred['height'] / 2
                        cls = pred['class']
                        detections.append((round(xmin), round(ymin), round(xmax), round(ymax), cls))
            except Exception as e:
                print(f"Error during prediction with commercial model from project {model.project_name}, version {model.version}: {e}")
    return detections
