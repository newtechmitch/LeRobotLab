import numpy as np
from lerobot.common.model.kinematics import RobotKinematics
from scipy.spatial.transform import Rotation

robot_kinematics = RobotKinematics("so_old_calibration")


def compute_cartesian_positions(joint_states: np.ndarray):
    """Convert joint positions to cartesian positions using forward kinematics."""
    cartesian_positions = np.zeros((6))
    gripper_position = 0
    ee_pose = robot_kinematics.forward_kinematics(joint_states[:5], frame="gripper_tip")
    cartesian_positions[:3] = ee_pose[:3, 3]
    rpy = Rotation.from_matrix(ee_pose[:3, :3]).as_euler('zyx', degrees=False)
    cartesian_positions[3:] = rpy
    gripper_position = joint_states[5] / 100.0
    return cartesian_positions, gripper_position
