from lerobot.common.teleoperators.so101_leader import SO101LeaderConfig, SO101Leader
from lerobot.common.robots.so101_follower import SO101FollowerConfig, SO101Follower
from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig


camera_config = {
    "front": OpenCVCameraConfig(index_or_path=0, width=1920, height=1080, fps=30),
    "top": OpenCVCameraConfig(index_or_path=2, width=1920, height=1080, fps=30)
}

robot_config = SO101FollowerConfig(  # follower arm
    port="/dev/tty.usbmodem5A460828321",
    id="michel_follower_arm",
    cameras=camera_config
)

teleop_config = SO101LeaderConfig(  # leader arm
    port="/dev/tty.usbmodem5A460823761",
    id="michel_leader_arm",
)


robot = SO101Follower(robot_config)
teleop_device = SO101Leader(teleop_config)
robot.connect()
teleop_device.connect()

while True:
    action_received = teleop_device.get_action()
    print(f"Action received: {action_received['elbow_flex.pos']}")
    action_to_send = action_received.copy()
    robot.send_action(action_to_send)
