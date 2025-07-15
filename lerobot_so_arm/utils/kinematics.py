# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#
# This is a copy of the original code from https://github.com/huggingface/lerobot/blob/main/lerobot/common/model/kinematics.py
# We will use it to compute the forward and inverse kinematics of the SO100/SO101 robot.

#
# Notes from my research:
# Joint Position Units:
#
# Rotational Joints (first 5):
#"shoulder_pan.pos" - degrees (range: [-180, 180])
#"shoulder_lift.pos" - degrees (range: [-180, 180])
#"elbow_flex.pos" - degrees (range: [-180, 180])
#"wrist_flex.pos" - degrees (range: [-180, 180])
#"wrist_roll.pos" - degrees (range: [-180, 180])

# Linear Joint (gripper):
#"gripper.pos" - percentage (range: [0, 100])

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation


def skew_symmetric(w: NDArray[np.float32]) -> NDArray[np.float32]:
    """Creates the skew-symmetric matrix from a 3D vector."""
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def rodrigues_rotation(w: NDArray[np.float32], theta: float) -> NDArray[np.float32]:
    """Computes the rotation matrix using Rodrigues' formula."""
    w_hat = skew_symmetric(w)
    return np.eye(3) + np.sin(theta) * w_hat + (1 - np.cos(theta)) * w_hat @ w_hat


def screw_axis_to_transform(s: NDArray[np.float32], theta: float) -> NDArray[np.float32]:
    """Converts a screw axis to a 4x4 transformation matrix."""
    screw_axis_rot = s[:3]
    screw_axis_trans = s[3:]

    # Pure translation
    if np.allclose(screw_axis_rot, 0) and np.linalg.norm(screw_axis_trans) == 1:
        transform = np.eye(4)
        transform[:3, 3] = screw_axis_trans * theta

    # Rotation (and potentially translation)
    elif np.linalg.norm(screw_axis_rot) == 1:
        w_hat = skew_symmetric(screw_axis_rot)
        rot_mat = np.eye(3) + np.sin(theta) * w_hat + (1 - np.cos(theta)) * w_hat @ w_hat
        t = (
            np.eye(3) * theta + (1 - np.cos(theta)) * w_hat + (theta - np.sin(theta)) * w_hat @ w_hat
        ) @ screw_axis_trans
        transform = np.eye(4)
        transform[:3, :3] = rot_mat
        transform[:3, 3] = t
    else:
        raise ValueError("Invalid screw axis parameters")
    return transform


def pose_difference_se3(pose1: NDArray[np.float32], pose2: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Calculates the SE(3) difference between two 4x4 homogeneous transformation matrices.
    SE(3) (Special Euclidean Group) represents rigid body transformations in 3D space,
    combining rotation (SO(3)) and translation.

    Each 4x4 matrix has the following structure:
    [R11 R12 R13 tx]
    [R21 R22 R23 ty]
    [R31 R32 R33 tz]
    [ 0   0   0   1]

    where R is the 3x3 rotation matrix and [tx,ty,tz] is the translation vector.

    Args:
        pose1: A 4x4 numpy array representing the first pose.
        pose2: A 4x4 numpy array representing the second pose.

    Returns:
        A 6D numpy array concatenating translation and rotation differences.
        First 3 elements are the translational difference (position).
        Last 3 elements are the rotational difference in axis-angle representation.
    """
    rot1 = pose1[:3, :3]
    rot2 = pose2[:3, :3]

    translation_diff = pose1[:3, 3] - pose2[:3, 3]

    # Calculate rotational difference using scipy's Rotation library
    rot_diff = Rotation.from_matrix(rot1 @ rot2.T)
    rotation_diff = rot_diff.as_rotvec()  # Axis-angle representation

    return np.concatenate([translation_diff, rotation_diff])


def se3_error(target_pose: NDArray[np.float32], current_pose: NDArray[np.float32]) -> NDArray[np.float32]:
    pos_error = target_pose[:3, 3] - current_pose[:3, 3]

    rot_target = target_pose[:3, :3]
    rot_current = current_pose[:3, :3]
    rot_error_mat = rot_target @ rot_current.T
    rot_error = Rotation.from_matrix(rot_error_mat).as_rotvec()

    return np.concatenate([pos_error, rot_error])


class RobotKinematics:
    """Robot kinematics class supporting multiple robot models."""

    # Robot measurements dictionary
    ROBOT_MEASUREMENTS = {
        "koch": {
            "gripper": [0.239, -0.001, 0.024],
            "wrist": [0.209, 0, 0.024],
            "forearm": [0.108, 0, 0.02],
            "humerus": [0, 0, 0.036],
            "shoulder": [0, 0, 0],
            "base": [0, 0, 0.02],
        },
        "moss": {
            "gripper": [0.246, 0.013, 0.111],
            "wrist": [0.245, 0.002, 0.064],
            "forearm": [0.122, 0, 0.064],
            "humerus": [0.001, 0.001, 0.063],
            "shoulder": [0, 0, 0],
            "base": [0, 0, 0.02],
        },
        "so_old_calibration": {
            "gripper": [0.320, 0, 0.050],
            "wrist": [0.278, 0, 0.050],
            "forearm": [0.143, 0, 0.044],
            "humerus": [0.031, 0, 0.072],
            "shoulder": [0, 0, 0],
            "base": [0, 0, 0.02],
        },
        "so_new_calibration": {
            "gripper": [0.33, 0.0, 0.285],
            "wrist": [0.30, 0.0, 0.267],
            "forearm": [0.25, 0.0, 0.266],
            "humerus": [0.06, 0.0, 0.264],
            "shoulder": [0.0, 0.0, 0.238],
            "base": [0.0, 0.0, 0.12],
        },
    }

    def __init__(self, robot_type: str = "so100"):
        """Initialize kinematics for the specified robot type.

        Args:
            robot_type: String specifying the robot model ("koch", "so100", or "moss")
        """
        if robot_type not in self.ROBOT_MEASUREMENTS:
            raise ValueError(
                f"Unknown robot type: {robot_type}. Available types: {list(self.ROBOT_MEASUREMENTS.keys())}"
            )

        self.robot_type = robot_type
        self.measurements = self.ROBOT_MEASUREMENTS[robot_type]

        # Initialize all transformation matrices and screw axes
        self._setup_transforms()

    def _create_translation_matrix(
        self, x: float = 0.0, y: float = 0.0, z: float = 0.0
    ) -> NDArray[np.float32]:
        """Create a 4x4 translation matrix."""
        return np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])

    def _setup_transforms(self):
        """Setup all transformation matrices and screw axes for the robot."""
        # Set up rotation matrices (constant across robot types)

        # Gripper orientation
        self.gripper_X0 = np.array(
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, -1, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

        # Wrist orientation
        self.wrist_X0 = np.array(
            [
                [0, -1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

        # Base orientation
        self.base_X0 = np.array(
            [
                [0, 0, 1, 0],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

        # Gripper
        # Screw axis of gripper frame wrt base frame
        self.S_BG = np.array(
            [
                1,
                0,
                0,
                0,
                self.measurements["gripper"][2],
                -self.measurements["gripper"][1],
            ],
            dtype=np.float32,
        )

        # Gripper origin to centroid transform
        self.X_GoGc = self._create_translation_matrix(x=0.07)

        # Gripper origin to tip transform
        self.X_GoGt = self._create_translation_matrix(x=0.12)

        # 0-position gripper frame pose wrt base
        self.X_BoGo = self._create_translation_matrix(
            x=self.measurements["gripper"][0],
            y=self.measurements["gripper"][1],
            z=self.measurements["gripper"][2],
        )

        # Wrist
        # Screw axis of wrist frame wrt base frame
        self.S_BR = np.array(
            [0, 1, 0, -self.measurements["wrist"][2], 0, self.measurements["wrist"][0]], dtype=np.float32
        )

        # 0-position wrist frame pose wrt base
        self.X_BoRo = self._create_translation_matrix(
            x=self.measurements["wrist"][0],
            y=self.measurements["wrist"][1],
            z=self.measurements["wrist"][2],
        ) @ self.wrist_X0

        # Forearm
        # Screw axis of forearm frame wrt base frame
        self.S_BF = np.array(
            [0, 0, 1, -self.measurements["forearm"][1], self.measurements["forearm"][0], 0],
            dtype=np.float32,
        )

        # 0-position forearm frame pose wrt base
        self.X_BoFo = self._create_translation_matrix(
            x=self.measurements["forearm"][0],
            y=self.measurements["forearm"][1],
            z=self.measurements["forearm"][2],
        )

        # Humerus
        # Screw axis of humerus frame wrt base frame
        self.S_BH = np.array(
            [0, 1, 0, -self.measurements["humerus"][2], 0, self.measurements["humerus"][0]],
            dtype=np.float32,
        )

        # 0-position humerus frame pose wrt base
        self.X_BoHo = self._create_translation_matrix(
            x=self.measurements["humerus"][0],
            y=self.measurements["humerus"][1],
            z=self.measurements["humerus"][2],
        )

        # Shoulder
        # Screw axis of shoulder frame wrt base frame
        self.S_BS = np.array([0, 0, 1, 0, 0, 0], dtype=np.float32)

        # 0-position shoulder frame pose wrt base
        self.X_BoSo = self._create_translation_matrix(
            x=self.measurements["shoulder"][0],
            y=self.measurements["shoulder"][1],
            z=self.measurements["shoulder"][2],
        )

        # Base
        # 0-position base frame pose wrt world
        self.X_WBo = self._create_translation_matrix(
            x=self.measurements["base"][0],
            y=self.measurements["base"][1],
            z=self.measurements["base"][2],
        ) @ self.base_X0

    def forward_kinematics(
        self,
        robot_pos_deg: NDArray[np.float32],
        frame: str = "gripper_tip",
    ) -> NDArray[np.float32]:
        """
        Compute forward kinematics for the robot.

        Args:
            robot_pos_deg: Array of joint positions in degrees [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll]
            frame: Target frame to compute ("base", "shoulder", "humerus", "forearm", "wrist", "gripper", "gripper_centroid", "gripper_tip")

        Returns:
            4x4 homogeneous transformation matrix representing the pose of the target frame
        """
        # Convert joint angles from degrees to radians
        robot_pos_rad = np.deg2rad(robot_pos_deg)

        # Compute transforms for each joint
        # Base to shoulder
        X_BS = screw_axis_to_transform(self.S_BS, robot_pos_rad[0])

        # Base to humerus
        X_BH = X_BS @ screw_axis_to_transform(self.S_BH, robot_pos_rad[1])

        # Base to forearm
        X_BF = X_BH @ screw_axis_to_transform(self.S_BF, robot_pos_rad[2])

        # Base to wrist
        X_BR = X_BF @ screw_axis_to_transform(self.S_BR, robot_pos_rad[3])

        # Base to gripper
        X_BG = X_BR @ screw_axis_to_transform(self.S_BG, robot_pos_rad[4])

        # Return the appropriate transform based on the requested frame
        if frame == "base":
            return self.X_WBo
        elif frame == "shoulder":
            return self.X_WBo @ self.X_BoSo @ X_BS
        elif frame == "humerus":
            return self.X_WBo @ self.X_BoHo @ X_BH
        elif frame == "forearm":
            return self.X_WBo @ self.X_BoFo @ X_BF
        elif frame == "wrist":
            return self.X_WBo @ self.X_BoRo @ X_BR
        elif frame == "gripper":
            return self.X_WBo @ self.X_BoGo @ X_BG
        elif frame == "gripper_centroid":
            return self.X_WBo @ self.X_BoGo @ X_BG @ self.X_GoGc
        elif frame == "gripper_tip":
            return self.X_WBo @ self.X_BoGo @ X_BG @ self.X_GoGt
        else:
            raise ValueError(f"Unknown frame: {frame}")

    def compute_jacobian(
        self, robot_pos_deg: NDArray[np.float32], frame: str = "gripper_tip"
    ) -> NDArray[np.float32]:
        """
        Compute the Jacobian matrix for the robot at the given configuration.

        Args:
            robot_pos_deg: Array of joint positions in degrees [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll]
            frame: Target frame to compute the Jacobian for

        Returns:
            6xN Jacobian matrix (N is number of joints)
        """
        # Small perturbation for numerical differentiation
        delta = 0.01  # radians
        
        # Get current end-effector pose
        current_pose = self.forward_kinematics(robot_pos_deg, frame=frame)
        
        # Initialize Jacobian matrix (6 x number of joints)
        jacobian = np.zeros((6, len(robot_pos_deg)))
        
        # Compute Jacobian columns through numerical differentiation
        for i in range(len(robot_pos_deg)):
            # Perturb joint position
            perturbed_pos = robot_pos_deg.copy()
            perturbed_pos[i] += np.rad2deg(delta)
            
            # Get perturbed end-effector pose
            perturbed_pose = self.forward_kinematics(perturbed_pos, frame=frame)
            
            # Compute pose difference
            pose_diff = pose_difference_se3(perturbed_pose, current_pose)
            
            # Jacobian column is the pose difference divided by the perturbation
            jacobian[:, i] = pose_diff / delta
        
        return jacobian

    def compute_positional_jacobian(
        self, robot_pos_deg: NDArray[np.float32], frame: str = "gripper_tip"
    ) -> NDArray[np.float32]:
        """
        Compute the positional part of the Jacobian matrix (for position-only control).

        Args:
            robot_pos_deg: Array of joint positions in degrees [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll]
            frame: Target frame to compute the Jacobian for

        Returns:
            3xN Jacobian matrix for position only (N is number of joints)
        """
        # Compute full Jacobian
        full_jacobian = self.compute_jacobian(robot_pos_deg, frame=frame)
        
        # Extract positional part (first 3 rows)
        positional_jacobian = full_jacobian[:3, :]
        
        return positional_jacobian

    def ik(
        self,
        current_joint_pos: NDArray[np.float32],
        desired_ee_pose: NDArray[np.float32],
        position_only: bool = True,
        frame: str = "gripper_tip",
        max_iterations: int = 5,
        learning_rate: float = 1,
    ) -> NDArray[np.float32]:
        """
        Compute inverse kinematics using the Jacobian pseudo-inverse method.

        Args:
            current_joint_pos: Current joint positions in degrees [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll]
            desired_ee_pose: Desired end-effector pose as a 4x4 homogeneous transformation matrix
            position_only: If True, only consider position error (not orientation)
            frame: Target frame for IK
            max_iterations: Maximum number of iterations for the IK solver
            learning_rate: Learning rate for the IK solver

        Returns:
            Array of joint positions in degrees
        """
        joint_pos = current_joint_pos.copy()
        
        for _ in range(max_iterations):
            # Get current end-effector pose
            current_ee_pose = self.forward_kinematics(joint_pos, frame=frame)
            
            # Compute error
            if position_only:
                error = desired_ee_pose[:3, 3] - current_ee_pose[:3, 3]
                jacobian = self.compute_positional_jacobian(joint_pos, frame=frame)
            else:
                error = se3_error(desired_ee_pose, current_ee_pose)
                jacobian = self.compute_jacobian(joint_pos, frame=frame)
            
            # Compute pseudo-inverse of Jacobian
            jacobian_pinv = np.linalg.pinv(jacobian)
            
            # Update joint positions
            delta_theta = jacobian_pinv @ error
            joint_pos += np.rad2deg(learning_rate * delta_theta)
        
        return joint_pos 