import numpy as np
from lerobot.common.teleoperators.so101_leader import SO101LeaderConfig, SO101Leader
from lerobot.common.robots.so101_follower import SO101FollowerConfig, SO101Follower
from utils import compute_cartesian_positions

robot_config = SO101FollowerConfig(  # follower arm
    port="/dev/tty.usbmodem5A460828321",
    id="michel_follower_arm",
)

teleop_config = SO101LeaderConfig(  # leader arm
    port="/dev/tty.usbmodem5A460823761",
    id="michel_leader_arm",
)

robot = SO101Follower(robot_config)
teleop_device = SO101Leader(teleop_config)
robot.connect()
teleop_device.connect()

# Counter for printing
loop_counter = 0

while True:
    action_received = teleop_device.get_action()
    action_to_send = action_received.copy()
    robot.send_action(action_to_send)

    # Print  joint observations once every 10 loops
    loop_counter += 1
    if loop_counter % 10 == 0:
        observations = robot.get_observation()

        print("Action received:")
        print(f"  Shoulder Pan: {observations['shoulder_pan.pos']}")
        print(f"  Shoulder Lift: {observations['shoulder_lift.pos']}")
        print(f"  Elbow Flex: {observations['elbow_flex.pos']}")
        print(f"  Wrist Flex: {observations['wrist_flex.pos']}")
        print(f"  Wrist Roll: {observations['wrist_roll.pos']}")
        print(f"  Gripper: {observations['gripper.pos']}")  
        # Create joint_states from observations
        joint_states = np.zeros(6)  
        joint_states[0] = observations['shoulder_pan.pos']
        joint_states[1] = observations['shoulder_lift.pos']
        joint_states[2] = observations['elbow_flex.pos']
        joint_states[3] = observations['wrist_flex.pos']
        joint_states[4] = observations['wrist_roll.pos']
        joint_states[5] = observations['gripper.pos']
        # Compute cartesian positions and gripper positions
        cartesian_positions, gripper_positions = compute_cartesian_positions(joint_states)
        # Print cartesian positions and gripper positions
        print("\nCartesian positions:")
        print(f"  X: {cartesian_positions[0]:.4f}")
        print(f"  Y: {cartesian_positions[1]:.4f}")
        print(f"  Z: {cartesian_positions[2]:.4f}")
        print("\nGripper positions:")
        print(f"  Position: {gripper_positions:.4f}")
