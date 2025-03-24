import numpy as np

class DepthEstimator:
    def __init__(self, focal_length, real_wingspan=2.0, safety_margin=0.2):
        """
        focal_length: Kamera odak uzaklığı (pixel cinsinden)
        real_wingspan: UAV'ın gerçek kanat açıklığı (varsayılan 2m)
        safety_margin: Güvenlik payı oranı (default 0.2)
        """
        self.focal_length = focal_length
        self.real_wingspan = real_wingspan
        self.safety_margin = safety_margin

    def estimate_depth(self, bbox, orientation_confidence=1.0):
        """
        bbox: YOLO tarafından bulunan [x1, y1, x2, y2]
        orientation_confidence: Varsayılan 1.0, açı doğruluğu için
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        center_x = x1 + width / 2
        center_y = y1 + height / 2
        
        # Genişliği kullanarak derinlik tahmini
        apparent_size = width
        aspect_ratio = width / height if height > 0 else float('inf')
        is_normal_orientation = aspect_ratio >= 1.0
        expected_min_ratio = 1.5
        orientation_factor = min(aspect_ratio / expected_min_ratio, 1.0) if aspect_ratio < expected_min_ratio else 1.0
        adjusted_confidence = orientation_confidence * orientation_factor
        
        estimated_depth = (self.real_wingspan * self.focal_length) / apparent_size
        adjusted_depth = estimated_depth * adjusted_confidence
        
        half_wingspan_distance = (self.real_wingspan / 2) * (adjusted_depth / estimated_depth)
        safe_depth = adjusted_depth - half_wingspan_distance - (self.safety_margin * adjusted_depth)
        safe_depth = max(safe_depth, 0.1)

        return safe_depth
