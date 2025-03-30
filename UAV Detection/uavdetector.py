import cv2
import torch
import numpy as np
from ultralytics import YOLO
from collections import deque
import time

class UAVDetector:
    def __init__(self, model_path, focal_length=None, camera_matrix=None, real_wingspan=2.0, safety_margin=0.2):
        """
        Initialize UAV detector with YOLO model and depth estimation capabilities.
        
        Parameters:
        -----------
        model_path : str
            Path to YOLO model weights
        focal_length : float, optional
            Camera focal length in pixels
        camera_matrix : numpy.ndarray, optional
            3x3 camera intrinsic matrix (used instead of focal_length if provided)
        real_wingspan : float, optional
            Average UAV wingspan in meters (default 2.0m)
        safety_margin : float, optional
            Safety margin factor for depth calculation (default 0.2 or 20%)
        """
        # Initialize YOLO model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path).to(self.device)
        self.dtype = torch.float32
        
        # Camera parameters
        self.focal_length = focal_length
        self.camera_matrix = camera_matrix
        self.real_wingspan = real_wingspan
        self.safety_margin = safety_margin
        
        # Previous bounding boxes for tracking (store last 5)
        self.prev_boxes = deque(maxlen=20)
        
        # Performance tracking
        self.frame_times = []
        
        # Frame counter
        self.frame_count = 0
    
    def set_precision(self, precision="fp32"):
        """Set model precision (fp16 or fp32)"""
        self.dtype = torch.float16 if precision == "fp16" else torch.float32
    
    def calibrate_focal_length(self, known_distance, apparent_width):
        """
        Calibrate camera focal length using known distance and apparent width.
        
        Parameters:
        -----------
        known_distance : float
            Known distance to UAV in meters
        apparent_width : float
            Apparent width of UAV in image in pixels
            
        Returns:
        --------
        focal_length : float
            Calibrated focal length in pixels
        """
        self.focal_length = (apparent_width * known_distance) / self.real_wingspan
        return self.focal_length
    
    def set_camera_matrix(self, camera_matrix):
        """Set the camera intrinsic matrix"""
        self.camera_matrix = camera_matrix
    
    def detect(self, frame, conf_threshold=0.25):
        """
        Detect UAVs in a frame and estimate their depths.
        Only returns the detection with highest confidence score.
        If no detection, tries to predict using previous boxes.
        
        Parameters:
        -----------
        frame : numpy.ndarray
            Input image frame
        conf_threshold : float, optional
            Confidence threshold for detections (default 0.1)
            
        Returns:
        --------
        detection : dict or None
            Dictionary with 'bbox', 'confidence', 'depth' for the highest confidence detection,
            or None if no detection found and no previous boxes available
        """
        self.frame_count += 1
        
        # Pre-process frame
        resized_frame = cv2.resize(frame, (640, 640))
        img = torch.from_numpy(resized_frame).to(self.device).permute(2, 0, 1).unsqueeze(0).to(self.dtype) / 255.0
        
        # Measure inference time
        start_time = time.time()
        results = self.model.predict(img, imgsz=640, conf=conf_threshold)
        end_time = time.time()
        
        # Update frame times for FPS calculation
        self.frame_times.append(end_time - start_time)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        
        # Find highest confidence detection
        best_detection = None
        best_confidence = -1
        best_bbox = None
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0].item())
                
                # Check if this is the highest confidence detection so far
                if conf > best_confidence:
                    best_confidence = conf
                    best_bbox = [x1, y1, x2, y2]
        
        # If we found a detection, process it
        if best_bbox is not None:
            # Store only the best box in current_boxes
            current_boxes = [best_bbox]
            
            # Check for orientation changes if previous boxes exist
            orientation_confidence = 1.0
            
            if len(self.prev_boxes) > 0 and self.prev_boxes[-1]:
                prev_box = self.prev_boxes[-1][0] if len(self.prev_boxes[-1]) > 0 else None
                if prev_box:
                    prev_x1, prev_y1, prev_x2, prev_y2 = prev_box
                    prev_width = prev_x2 - prev_x1
                    prev_height = prev_y2 - prev_y1
                    curr_width = best_bbox[2] - best_bbox[0]
                    curr_height = best_bbox[3] - best_bbox[1]
                    
                    _, orientation_confidence = self.detect_orientation_change(
                        prev_width, prev_height, curr_width, curr_height
                    )
            
            # Estimate depth
            depth = self.estimate_depth(best_bbox, orientation_confidence)
            
            # Update previous boxes with only the best detection - THIS IS A REAL DETECTION
            self.prev_boxes.append(current_boxes)
            
            # Return only the best detection
            return {
                'bbox': best_bbox,
                'confidence': best_confidence,
                'depth': depth,
                'is_predicted': False
            }
        else:
            # No detection found above threshold
            self.prev_boxes.append([])  # Add empty list to preserve timeline
            
            # Try to predict using previous real detections
            virtual_detection = self._create_virtual_detection()
            
            # Return virtual detection if available, otherwise None
            return virtual_detection
    
    def _create_virtual_detection(self):
        """
        Create a virtual detection based on previous real detections.
        Does NOT store the virtual detection in prev_boxes.
        
        Returns:
        --------
        dict or None: Virtual detection or None if no previous real detections
        """
        # Find the most recent non-empty previous box
        most_recent_box = None
        most_recent_idx = -1
        
        for i in range(len(self.prev_boxes) - 1, -1, -1):  # Start from most recent (excluding current empty one)
            if self.prev_boxes[i] and len(self.prev_boxes[i]) > 0:
                most_recent_box = self.prev_boxes[i][0]
                most_recent_idx = i
                break
        
        # If no previous detection found, return None
        if most_recent_box is None:
            return None
        
        # Calculate how many frames have passed since the last real detection
        frames_passed = len(self.prev_boxes) - 1 - most_recent_idx
        
        # Find second most recent box for velocity calculation if available
        second_recent_box = None
        second_recent_idx = -1
        
        for i in range(most_recent_idx - 1, -1, -1):
            if self.prev_boxes[i] and len(self.prev_boxes[i]) > 0:
                second_recent_box = self.prev_boxes[i][0]
                second_recent_idx = i
                break
        
        # If we have two recent detections, calculate velocity and predict position
        if second_recent_box is not None:
            frames_between = most_recent_idx - second_recent_idx
            
            # Calculate centers and dimensions
            r_x1, r_y1, r_x2, r_y2 = most_recent_box
            s_x1, s_y1, s_x2, s_y2 = second_recent_box
            
            r_center_x = (r_x1 + r_x2) / 2
            r_center_y = (r_y1 + r_y2) / 2
            r_width = r_x2 - r_x1
            r_height = r_y2 - r_y1
            
            s_center_x = (s_x1 + s_x2) / 2
            s_center_y = (s_y1 + s_y2) / 2
            s_width = s_x2 - s_x1
            s_height = s_y2 - s_y1
            
            # Calculate velocity vectors (pixels per frame)
            vel_x = (r_center_x - s_center_x) / frames_between
            vel_y = (r_center_y - s_center_y) / frames_between
            vel_w = (r_width - s_width) / frames_between
            vel_h = (r_height - s_height) / frames_between
            
            # Predict new position
            pred_center_x = r_center_x + vel_x * frames_passed
            pred_center_y = r_center_y + vel_y * frames_passed
            pred_width = max(10, r_width + vel_w * frames_passed)  # Ensure minimum size
            pred_height = max(10, r_height + vel_h * frames_passed)
            
            # Calculate predicted bounding box
            pred_x1 = int(max(0, pred_center_x - pred_width / 2))
            pred_y1 = int(max(0, pred_center_y - pred_height / 2))
            pred_x2 = int(pred_center_x + pred_width / 2)
            pred_y2 = int(pred_center_y + pred_height / 2)
            
            pred_bbox = [pred_x1, pred_y1, pred_x2, pred_y2]
            
        else:
            # If we only have one detection, use it directly (assume no motion)
            pred_bbox = most_recent_box
        
        # Calculate depth for the predicted bbox
        pred_depth = self.estimate_depth(pred_bbox)
        
        # Decrease confidence based on how many frames have passed
        decay_factor = 0.8 ** frames_passed  # Exponential decay
        predicted_confidence = max(0.1, decay_factor)  # Minimum confidence threshold
        
        # Create virtual detection
        return {
            'bbox': pred_bbox,
            'confidence': predicted_confidence,
            'depth': pred_depth,
            'is_predicted': True  # Flag indicating this is a virtual box
        }
    
    def estimate_depth(self, bbox, orientation_confidence=1.0):
        """
        Estimate depth based on bounding box dimensions.
        
        Parameters:
        -----------
        bbox : list
            Bounding box [x1, y1, x2, y2]
        orientation_confidence : float, optional
            Confidence in orientation estimate (default 1.0)
            
        Returns:
        --------
        depth : float
            Estimated depth in meters
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        center_x = x1 + width / 2
        center_y = y1 + height / 2
        
        # Use camera matrix if available, otherwise use focal length
        if self.camera_matrix is not None:
            fx = self.camera_matrix[0, 0]
            fy = self.camera_matrix[1, 1]
            focal_length = (fx + fy) / 2
        elif self.focal_length is not None:
            focal_length = self.focal_length
        else:
            raise ValueError("Either focal_length or camera_matrix must be provided")
        
        # Calculate aspect ratio and check for unusual orientation
        aspect_ratio = width / height if height > 0 else float('inf')
        is_normal_orientation = aspect_ratio >= 1.0
        
        # Depth estimation based on similar triangles
        estimated_depth = (self.real_wingspan * focal_length) / width
        
        # Adjust confidence based on orientation
        expected_min_ratio = 1.5
        orientation_factor = min(aspect_ratio / expected_min_ratio, 1.0) if aspect_ratio < expected_min_ratio else 1.0
        adjusted_confidence = orientation_confidence * orientation_factor
        
        # Adjust depth based on orientation confidence
        adjusted_depth = estimated_depth * adjusted_confidence
        
        # Apply safety margin
        half_wingspan_distance = (self.real_wingspan / 2) * (adjusted_depth / estimated_depth)
        safe_depth = adjusted_depth - half_wingspan_distance - (self.safety_margin * adjusted_depth)
        
        # Ensure positive depth
        safe_depth = max(safe_depth, 0.1)
        
        return safe_depth
    
    def detect_orientation_change(self, prev_width, prev_height, curr_width, curr_height, threshold=0.3):
        """
        Detect significant changes in UAV orientation by comparing aspect ratios.
        
        Parameters:
        -----------
        prev_width, prev_height : float
            Previous frame's bounding box dimensions
        curr_width, curr_height : float
            Current frame's bounding box dimensions
        threshold : float, optional
            Threshold for significant change (default 0.3)
            
        Returns:
        --------
        is_changing : bool
            True if orientation is changing significantly
        confidence : float
            Confidence in orientation estimate (0.0-1.0)
        """
        # Calculate aspect ratios
        prev_ratio = prev_width / prev_height if prev_height > 0 else float('inf')
        curr_ratio = curr_width / curr_height if curr_height > 0 else float('inf')
        
        # Calculate ratio change
        if prev_ratio > curr_ratio:
            ratio_change = prev_ratio / curr_ratio - 1 if curr_ratio > 0 else float('inf')
        else:
            ratio_change = curr_ratio / prev_ratio - 1 if prev_ratio > 0 else float('inf')
        
        # Determine if change is significant
        is_changing = ratio_change > threshold
        
        # Calculate confidence (inversely proportional to ratio change)
        confidence = 1.0 / (1.0 + ratio_change)
        
        return is_changing, confidence
    
    def get_fps(self):
        """Calculate average FPS based on recent frame times"""
        avg_time = sum(self.frame_times) / len(self.frame_times) if self.frame_times else 0
        fps = 1 / avg_time if avg_time > 0 else 0
        return fps