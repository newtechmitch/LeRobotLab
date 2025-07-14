import json
import math
import numpy as np

# Your actual calibration data
CALIBRATION_DATA = {
    "shoulder_pan": {
        "id": 1,
        "drive_mode": 0,
        "homing_offset": 364,
        "range_min": 792,
        "range_max": 3410
    },
    "shoulder_lift": {
        "id": 2,
        "drive_mode": 0,
        "homing_offset": 1054,
        "range_min": 872,
        "range_max": 3211
    },
    "elbow_flex": {
        "id": 3,
        "drive_mode": 0,
        "homing_offset": -48,
        "range_min": 872,
        "range_max": 3051
    },
    "wrist_flex": {
        "id": 4,
        "drive_mode": 0,
        "homing_offset": -289,
        "range_min": 927,
        "range_max": 3176
    },
    "wrist_roll": {
        "id": 5,
        "drive_mode": 0,
        "homing_offset": -1220,
        "range_min": 93,
        "range_max": 3936
    },
    "gripper": {
        "id": 6,
        "drive_mode": 0,
        "homing_offset": -816,
        "range_min": 1582,
        "range_max": 3033
    }
}

def calculate_joint_limits_radians():
    """Calculate actual physical joint limits in radians from calibration data"""
    joint_limits = {}
    motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    
    print("Physical Joint Limits from Your Calibration:")
    print("=" * 50)
    
    for joint_name in motor_names:
        calib = CALIBRATION_DATA[joint_name]
        range_min = calib["range_min"]
        range_max = calib["range_max"]
        homing_offset = calib["homing_offset"]
        
        if joint_name != "gripper":
            # Convert raw motor positions to radians
            # Formula: radians = (raw_position - 2048) * (2 * π / 4096)
            min_radians = (range_min - 2048) * (2 * math.pi / 4096)
            max_radians = (range_max - 2048) * (2 * math.pi / 4096)
            
            joint_limits[joint_name] = [min_radians, max_radians]
            
            print(f"{joint_name:12}: [{min_radians:6.3f}, {max_radians:6.3f}] rad")
            print(f"             [{math.degrees(min_radians):6.1f}°, {math.degrees(max_radians):6.1f}°]")
            print(f"             Range: {max_radians - min_radians:.3f} rad ({math.degrees(max_radians - min_radians):.1f}°)")
        else:
            # Gripper: keep as normalized 0-1
            joint_limits[joint_name] = [0.0, 1.0]
            print(f"{joint_name:12}: [0.000, 1.000] (normalized)")
        
        print()
    
    return joint_limits

def convert_percentage_to_radians(percentages, joint_names=None):
    """
    Convert SO100/SO101 percentage values to radians using your calibration data
    
    Args:
        percentages: List/array of percentage values (0-100) for each joint
        joint_names: List of joint names (optional, defaults to standard order)
    
    Returns:
        Array of joint positions in radians
    """
    if joint_names is None:
        joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    
    radians = []
    
    for i, percentage in enumerate(percentages):
        if i >= len(joint_names):
            break
            
        joint_name = joint_names[i]
        calib = CALIBRATION_DATA[joint_name]
        
        # Convert percentage to raw motor position
        range_min = calib["range_min"]
        range_max = calib["range_max"]
        raw_position = range_min + (percentage / 100.0) * (range_max - range_min)
        
        if joint_name != "gripper":
            # Convert raw position to radians
            joint_radians = (raw_position - 2048) * (2 * math.pi / 4096)
            radians.append(joint_radians)
        else:
            # Gripper: normalize to 0-1
            radians.append(percentage / 100.0)
    
    return np.array(radians)

def convert_radians_to_percentage(radians, joint_names=None):
    """
    Convert radians back to percentages using your calibration data
    
    Args:
        radians: List/array of joint positions in radians
        joint_names: List of joint names (optional, defaults to standard order)
    
    Returns:
        Array of percentage values (0-100)
    """
    if joint_names is None:
        joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    
    percentages = []
    
    for i, rad_value in enumerate(radians):
        if i >= len(joint_names):
            break
            
        joint_name = joint_names[i]
        calib = CALIBRATION_DATA[joint_name]
        range_min = calib["range_min"]
        range_max = calib["range_max"]
        
        if joint_name != "gripper":
            # Convert radians to raw motor position
            raw_position = (rad_value / (2 * math.pi / 4096)) + 2048
        else:
            # Gripper: convert from 0-1 to raw position
            raw_position = range_min + rad_value * (range_max - range_min)
        
        # Convert raw position to percentage
        percentage = ((raw_position - range_min) / (range_max - range_min)) * 100.0
        percentages.append(np.clip(percentage, 0, 100))  # Ensure 0-100 range
    
    return np.array(percentages)

def analyze_joint_ranges():
    """Analyze the physical joint ranges"""
    joint_limits = calculate_joint_limits_radians()
    
    print("\nJoint Range Analysis:")
    print("=" * 30)
    
    motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    
    for joint_name in motor_names:
        min_rad, max_rad = joint_limits[joint_name]
        range_rad = max_rad - min_rad
        range_deg = math.degrees(range_rad)
        
        print(f"{joint_name:12}: {range_deg:5.1f}° range")
    
    return joint_limits

def convert_dataset_observations(dataset_sample):
    """Example: Convert a dataset sample from percentages to radians"""
    
    # Extract observation state (assuming it's in percentage)
    obs_percentages = dataset_sample['observation.state']
    action_percentages = dataset_sample['action']
    
    # Convert to radians
    obs_radians = convert_percentage_to_radians(obs_percentages)
    action_radians = convert_percentage_to_radians(action_percentages)
    
    return {
        'observation_radians': obs_radians,
        'action_radians': action_radians,
        'original_obs_percentages': obs_percentages,
        'original_action_percentages': action_percentages
    }

# Example usage and testing
if __name__ == "__main__":
    print("SO100/SO101 Calibration-Based Converter")
    print("=" * 40)
    
    # Calculate and display joint limits
    joint_limits = analyze_joint_ranges()
    
    print("\n" + "=" * 50)
    print("Example Conversion:")
    print("=" * 20)
    
    # Test conversion with middle position (50% for all joints)
    test_percentages = [50, 50, 50, 50, 50, 50]
    test_radians = convert_percentage_to_radians(test_percentages)
    
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    
    print("50% position for all joints:")
    for i, (joint, perc, rad) in enumerate(zip(joint_names, test_percentages, test_radians)):
        if joint != "gripper":
            print(f"{joint:12}: {perc:5.1f}% -> {rad:6.3f} rad ({math.degrees(rad):6.1f}°)")
        else:
            print(f"{joint:12}: {perc:5.1f}% -> {rad:6.3f} (normalized)")
    
    # Test round-trip conversion
    print("\nRound-trip test (should return to original):")
    back_to_percentages = convert_radians_to_percentage(test_radians)
    print("Original:", test_percentages)
    print("Round-trip:", [f"{p:.1f}" for p in back_to_percentages])
