import numpy as np
import cv2

def estimate_depth(center_x, center_y, width, height, focal_length, real_wingspan=2.0, safety_margin=0.2, orientation_confidence=1.0):
    """
    Estimate the depth (distance) of a UAV from the camera using center coordinates and dimensions.
    
    Parameters:
    -----------
    center_x : float
        The x-coordinate of the bounding box center.
    center_y : float
        The y-coordinate of the bounding box center.
    width : float
        The width of the bounding box in pixels.
    height : float
        The height of the bounding box in pixels.
    focal_length : float
        The focal length of the camera in pixels.
    real_wingspan : float, optional
        The average wingspan of the UAV in meters (default is 2.0 meters).
    safety_margin : float, optional
        Safety margin factor to reduce the estimated distance (default is 0.2 or 20%).
        Higher values create a larger safety buffer.
    orientation_confidence : float, optional
        Confidence factor (0.0-1.0) for orientation alignment. Use lower values when
        the target UAV might be at an angle to the camera (default is 1.0).
        
    Returns:
    --------
    depth : float
        The estimated depth (distance) of the closest part of the UAV from the camera in meters.
    """
    # For UAVs, we should use the width as the apparent wingspan
    apparent_size = width
    
    # Check if height is larger than width, which could indicate unusual orientation
    unusual_orientation = height > width
    if unusual_orientation:
        print("WARNING: UAV bounding box height is larger than width. The UAV may be at an unusual orientation.")
    
    # Calculate the aspect ratio of the bounding box to detect potential orientation issues
    aspect_ratio = width / height if height > 0 else float('inf')
    
    # Track this for logging and debugging
    is_normal_orientation = aspect_ratio >= 1.0
    
    # Using the principle of similar triangles to calculate depth
    # depth = (real_size * focal_length) / apparent_size
    estimated_depth = (real_wingspan * focal_length) / apparent_size
    
    # For a typical UAV viewed head-on or from behind, aspect ratio should be substantially > 1
    # The more the aspect ratio drops below expected values, the less confidence we have
    # Most UAVs have wingspan/length ratios between 1.5 and 3.0
    expected_min_ratio = 1.5
    orientation_factor = min(aspect_ratio / expected_min_ratio, 1.0) if aspect_ratio < expected_min_ratio else 1.0
    
    adjusted_confidence = orientation_confidence * orientation_factor
    
    # Log additional information for debugging
    if not is_normal_orientation:
        print(f"CAUTION: Unusual aspect ratio detected: {aspect_ratio:.2f}. Confidence adjusted to {adjusted_confidence:.2f}")
    
    # Adjust the depth estimate based on orientation confidence
    # Lower confidence means we might be underestimating the actual wingspan
    # When aspect ratio indicates the UAV is not presenting its full wingspan,
    # we multiply by a factor < 1 to compensate for the foreshortened appearance
    adjusted_depth = estimated_depth * adjusted_confidence
    
    # Apply safety margin to get the distance to the closest part
    # For a UAV with wingspan w at distance d, the closest point could be d-(w/2)
    half_wingspan_distance = (real_wingspan / 2) * (adjusted_depth / estimated_depth)
    
    # Additionally apply the explicit safety margin factor
    safe_depth = adjusted_depth - half_wingspan_distance - (safety_margin * adjusted_depth)
    
    # Ensure we don't return a negative distance
    safe_depth = max(safe_depth, 0.1)
    
    return safe_depth

def detect_orientation_change(prev_width, prev_height, curr_width, curr_height, threshold=0.3):
    """
    Detect significant changes in UAV orientation by comparing aspect ratios.
    
    Parameters:
    -----------
    prev_width : float
        Previous frame's bounding box width.
    prev_height : float
        Previous frame's bounding box height.
    curr_width : float
        Current frame's bounding box width.
    curr_height : float
        Current frame's bounding box height.
    threshold : float, optional
        Threshold for aspect ratio change to be considered significant (default 0.3).
        
    Returns:
    --------
    is_changing : bool
        True if significant orientation change is detected.
    confidence : float
        Orientation confidence score (0.0-1.0).
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

def calibrate_focal_length(known_distance, real_wingspan, apparent_width):
    """
    Calibrate the focal length of the camera if it's not known.
    
    Parameters:
    -----------
    known_distance : float
        A known distance to the UAV in meters.
    real_wingspan : float
        The wingspan of the UAV in meters.
    apparent_width : float
        The apparent width of the UAV in the image in pixels.
        
    Returns:
    --------
    focal_length : float
        The estimated focal length in pixels.
    """
    # Using the formula: focal_length = (apparent_width * known_distance) / real_wingspan
    focal_length = (apparent_width * known_distance) / real_wingspan
    return focal_length

def estimate_depth_with_camera_matrix(center_x, center_y, width, height, camera_matrix, 
                                      real_wingspan=2.0, safety_margin=0.2, orientation_confidence=1.0):
    """
    Estimate depth using the camera intrinsic matrix.
    This is a more accurate approach if you have the camera's calibration parameters.
    
    Parameters:
    -----------
    center_x : float
        The x-coordinate of the bounding box center.
    center_y : float
        The y-coordinate of the bounding box center.
    width : float
        The width of the bounding box in pixels.
    height : float
        The height of the bounding box in pixels.
    camera_matrix : numpy array
        The 3x3 camera intrinsic matrix.
    real_wingspan : float, optional
        The average wingspan of the UAV in meters (default is 2.0 meters).
    safety_margin : float, optional
        Safety margin factor to reduce the estimated distance (default is 0.2 or 20%).
    orientation_confidence : float, optional
        Confidence factor (0.0-1.0) for orientation alignment (default is 1.0).
        
    Returns:
    --------
    depth : float
        The estimated depth (distance) of the closest part of the UAV from the camera in meters.
    """
    # Extract focal length from the camera matrix
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    
    # Use the average of fx and fy for better accuracy
    focal_length = (fx + fy) / 2
    
    # Call the original function with the extracted focal length
    return estimate_depth(center_x, center_y, width, height, focal_length, 
                         real_wingspan, safety_margin, orientation_confidence)

# Example usage
if __name__ == "__main__":
    # Example bounding box center and dimensions
    center_x, center_y = 200, 200
    width, height = 200, 100  # Width > height for a typical UAV orientation
    
    # Example focal length (this would typically come from camera calibration)
    focal_length = 800  # in pixels
    
    # Known wingspan of the UAV
    real_wingspan = 2.0  # meters
    
    # Safety margin (20% by default)
    safety_margin = 0.2
    
    # Orientation confidence - set lower if UAVs might be at various angles
    orientation_confidence = 0.8  # 80% confidence in orientation alignment
    
    # Estimate the depth
    depth = estimate_depth(center_x, center_y, width, height, focal_length, 
                          real_wingspan, safety_margin, orientation_confidence)
    print(f"Estimated safe depth: {depth:.2f} meters")
    
    # Try with different aspect ratios to simulate orientation changes
    narrow_width, narrow_height = 40, 200  # UAV turning (narrow width, tall height)
    depth_turning = estimate_depth(center_x, center_y, narrow_width, narrow_height, focal_length, 
                                  real_wingspan, safety_margin, orientation_confidence)
    print(f"Estimated safe depth when UAV is turning: {depth_turning:.2f} meters")
    
    # Example with inverted dimensions (height > width, unusual orientation)
    unusual_width, unusual_height = 20, 200  # Height > Width
    depth_unusual = estimate_depth(center_x, center_y, unusual_width, unusual_height, focal_length, 
                                  real_wingspan, safety_margin, orientation_confidence)
    print(f"Estimated safe depth with unusual orientation: {depth_unusual:.2f} meters")
    
    # Alternative: If you have the camera's intrinsic matrix
    camera_matrix = np.array([[focal_length, 0, 640/2],
                             [0, focal_length, 480/2],
                             [0, 0, 1]])
    
    depth_with_matrix = estimate_depth_with_camera_matrix(center_x, center_y, width, height, 
                                                         camera_matrix, real_wingspan, 
                                                         safety_margin, orientation_confidence)
    print(f"Estimated safe depth (using camera matrix): {depth_with_matrix:.2f} meters")
    
    # Compare with no safety margin and perfect orientation confidence
    direct_depth = estimate_depth(center_x, center_y, width, height, focal_length, 
                                 real_wingspan, 0.0, 1.0)
    print(f"Raw estimated depth (no safety features): {direct_depth:.2f} meters")
    print(f"Safety buffer: {direct_depth - depth:.2f} meters")
    
    # Demonstrate orientation change detection
    prev_width, prev_height = 200, 100
    curr_width, curr_height = 150, 130
    is_changing, confidence = detect_orientation_change(prev_width, prev_height, 
                                                      curr_width, curr_height)
    print(f"Orientation changing: {is_changing}, Confidence: {confidence:.2f}")