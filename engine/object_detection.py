
import cv2
import numpy as np
from ultralytics import YOLO
from typing import Tuple, List
import random


class ObjectDetection:
    """
    Object Detection wrapper for YOLO models.
    Provides a clean interface for detection with pre-configured colors and class names.
    """
    
    # COCO dataset class names (80 classes)
    classes = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 
        'toothbrush'
    ]
    
    # Generate random colors for each class
    colors = {i: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) 
              for i in range(len(classes))}
    
    def __init__(self, model_path: str = "yolov8n.pt"):
        """
        Initialize the ObjectDetection with a YOLO model.
        
        Args:
            model_path: Path to the YOLO model file (.pt)
        """
        self.model_path = model_path
        self.model = YOLO(model_path)
        print(f"Model loaded: {model_path}")
    
    def detect(self, frame: np.ndarray, img_size: int = 640, conf: float = 0.25, 
               iou: float = 0.45) -> Tuple[List, List, List]:
        """
        Perform object detection on a single frame.
        
        Args:
            frame: Input image/frame (numpy array)
            img_size: Image size for inference
            conf: Confidence threshold
            iou: IoU threshold for NMS
            
        Returns:
            Tuple of (bboxes, class_ids, scores)
            - bboxes: List of bounding boxes [[x1, y1, x2, y2], ...]
            - class_ids: List of class IDs
            - scores: List of confidence scores
        """
        results = self.model(frame, imgsz=img_size, conf=conf, iou=iou, verbose=False)
        
        bboxes = []
        class_ids = []
        scores = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                bboxes.append([x1, y1, x2, y2])
                
                # Extract class ID and confidence
                class_ids.append(int(box.cls[0].cpu().numpy()))
                scores.append(float(box.conf[0].cpu().numpy()))
        
        return bboxes, class_ids, scores
    
    @classmethod
    def get_class_name(cls, class_id: int) -> str:
        """Get class name from class ID."""
        if 0 <= class_id < len(cls.classes):
            return cls.classes[class_id]
        return "Unknown"
    
    @classmethod
    def get_color(cls, class_id: int) -> Tuple[int, int, int]:
        """Get color for a specific class ID."""
        return cls.colors.get(class_id, (255, 255, 255))