from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
import cv2

def inference(image):
    """
    Perform pest detection inference on an image
    
    Args:
        image: Can be PIL Image, numpy array, or bytes
    
    Returns:
        tuple: (annotated_image, classes_dict, detected_class_indices)
    """
    try:
        # Load the model with error handling
        try:
            model = YOLO('best.pt')
        except Exception as e:
            print(f"Error loading model: {e}")
            return handle_inference_error(image)
        
        # Convert input to appropriate format
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        elif isinstance(image, np.ndarray):
            # Ensure correct color format
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            image = Image.fromarray(image)
        
        # Run inference with optimized settings
        results = model(
            image, 
            conf=0.25,  # Confidence threshold
            iou=0.45,   # IoU threshold for NMS
            max_det=10  # Maximum detections
        )
        
        # Initialize return values
        infer = np.array(image)
        classes = {}
        namesInfer = []
        
        # Process results
        if results and len(results) > 0:
            result = results[0]
            
            # Get annotated image with custom settings
            infer = result.plot(
                line_width=2,
                font_size=16,
                pil=False
            )
            
            # Get class information
            classes = result.names
            
            # Get detected classes with confidence scores
            if result.boxes is not None and len(result.boxes) > 0:
                namesInfer = result.boxes.cls.tolist()
                
                # Log detections for debugging
                for i, (cls, conf) in enumerate(zip(result.boxes.cls, result.boxes.conf)):
                    class_name = classes[int(cls)]
                    confidence = float(conf)
                    print(f"Detected: {class_name} (confidence: {confidence:.2f})")
        
        return infer, classes, namesInfer
        
    except Exception as e:
        print(f"Error in inference: {e}")
        return handle_inference_error(image)

def handle_inference_error(image):
    """Handle errors by returning the original image"""
    try:
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return np.array(image), {}, []
    except:
        # Return a blank image if all else fails
        return np.zeros((480, 640, 3), dtype=np.uint8), {}, []
