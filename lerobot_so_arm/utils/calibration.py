import numpy as np
from datetime import datetime
from typing import Optional

def detect_so_calibration(
    joint_data: np.ndarray,
    creation_date: Optional[datetime] = None
) -> str:
    """
    Detect SO-100 calibration system from joint data and creation date.
    
    Thresholds:
    - Date cutoff: January 27, 2025 (new calibration PR date)
    - Joint extremes: >150° indicates old system  
    - Joint centering: <30° mean indicates new system, >90° mean indicates old
    
    Args:
        joint_data: Array (n_samples, 6) - shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
        creation_date: Dataset creation timestamp
        
    Returns:
        Robot type: 'so_old_calibration' or 'so_new_calibration'
    """
    
    old_votes = 0
    new_votes = 0
    
    # Date check
    if creation_date and creation_date < datetime(2025, 1, 27):
        old_votes += 2
    elif creation_date:
        new_votes += 2
    
    # Joint analysis
    if joint_data.shape[1] >= 6:
        shoulder_lift = joint_data[:, 1]
        elbow_flex = joint_data[:, 2]
        
        # Check for extreme positions (old system indicator)
        if np.max(np.abs(shoulder_lift)) > 150 or np.max(np.abs(elbow_flex)) > 150:
            old_votes += 1
        
        # Check centering
        shoulder_center = abs(np.mean(shoulder_lift))
        elbow_center = abs(np.mean(elbow_flex))
        
        if shoulder_center < 30 and elbow_center < 30:
            new_votes += 1
        elif shoulder_center > 90 or elbow_center > 90:
            old_votes += 1
    
    return 'so_old_calibration' if old_votes > new_votes else 'so_new_calibration'