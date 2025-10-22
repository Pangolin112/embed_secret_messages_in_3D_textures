import numpy as np
import cv2
from scipy.sparse import linalg as linalg
import warnings
warnings.filterwarnings('ignore')


def opencv_seamless_clone(source, target, mask):
    """
    Use OpenCV's built-in seamless cloning (fastest option)
    """
    # Ensure all images have the same size
    if source.shape[:2] != target.shape[:2] or source.shape[:2] != mask.shape[:2]:
        h, w = target.shape[:2]
        source = cv2.resize(source, (w, h))
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Ensure mask is uint8 and binary
    if mask.dtype == bool:
        mask_uint8 = (mask * 255).astype(np.uint8)
    elif mask.dtype != np.uint8:
        mask_uint8 = (mask * 255).astype(np.uint8)
    else:
        mask_uint8 = mask.copy()
    
    # Make mask binary (0 or 255) - be more aggressive about this
    mask_uint8 = np.where(mask_uint8 > 0, 255, 0).astype(np.uint8)
    
    # Find contours with different retrieval modes
    contours, hierarchy = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Ensure the bounding box is within image bounds
    img_h, img_w = target.shape[:2]
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = min(w, img_w - x)
    h = min(h, img_h - y)
    
    # Calculate center point within the bounding box
    center_x = x + w // 2
    center_y = y + h // 2
    
    # Ensure center is within image bounds
    center_x = max(1, min(center_x, img_w - 2))
    center_y = max(1, min(center_y, img_h - 2))
    
    # Verify the center point is actually in the mask
    if mask_uint8[center_y, center_x] == 0:
        # Find nearest white pixel in the contour
        mask_points = np.column_stack(np.where(mask_uint8 > 0))
        if len(mask_points) > 0:
            # Use the first white pixel
            center_y, center_x = mask_points[0]
    
    center = (center_x, center_y)
    
    # Perform seamless cloning
    # result = cv2.seamlessClone(source, target, mask_uint8, center, cv2.NORMAL_CLONE)
    result = cv2.seamlessClone(source, target, mask_uint8, center, cv2.MIXED_CLONE)
    
    return result


