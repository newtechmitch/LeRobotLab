import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import json
import os
import h5py
from typing import List, Tuple, Dict
from lerobot_so_arm.config import get_path
from lerobot_so_arm.utils.kinematics import RobotKinematics
from lerobot_so_arm.utils.calibration import detect_so_calibration
from utils.data_loading import load_so_episode
    
SOURCE_FOLDER = get_path('datasets') + '/output'
OUTPUT_FOLDER = get_path('datasets') + '/output_extrinsic'
CALIBRATION_FRAMES_FOLDER = get_path('datasets') + '/calibration_frames'
CALIBRATION_DATA_FOLDER = get_path('datasets') + '/calibration_data'


class DatasetExtrinsicCalibrator:
    """Calculate camera extrinsics from robot dataset using PnP."""
    
    def __init__(self, robot_kinematics, camera_matrix=None, dist_coeffs=None):
        self.K = camera_matrix
        self.dist = dist_coeffs if dist_coeffs is not None else np.zeros(5)
        self.robot_kinematics = robot_kinematics
        
    def get_robot_keypoints_3d(self, joint_angles_deg: np.ndarray) -> Dict[str, np.ndarray]:
        """Get 3D coordinates of robot keypoints using forward kinematics."""
        keypoints_3d = {}
        frames = ["base", "gripper", "gripper_tip"]  # Use base, gripper, and gripper tip
        
        for frame in frames:
            try:
                T = self.robot_kinematics.forward_kinematics(joint_angles_deg, frame) # add check of the arm_callibration
                keypoints_3d[frame] = T[:3, 3]
            except Exception:
                continue
                
        return keypoints_3d
    
    def annotate_robot_keypoints(self, image: np.ndarray, keypoints_3d: Dict[str, np.ndarray], 
                                save_path: str = None) -> Dict[str, np.ndarray]:
        """Interactive tool to annotate robot keypoints in image."""
        keypoints_2d = {}
        current_keypoint = 0
        keypoint_names = list(keypoints_3d.keys())
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal current_keypoint, keypoints_2d
            
            if event == cv2.EVENT_LBUTTONDOWN and current_keypoint < len(keypoint_names):
                keypoint_name = keypoint_names[current_keypoint]
                keypoints_2d[keypoint_name] = np.array([x, y])
                
                cv2.circle(image_display, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(image_display, keypoint_name, (x+10, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                current_keypoint += 1
                
                if current_keypoint < len(keypoint_names):
                    print(f"Click on: {keypoint_names[current_keypoint]}")
                else:
                    print("Complete! Press 's' to save, 'q' to quit")
        
        image_display = image.copy()
        print(f"Click on: {keypoint_names[current_keypoint]}")
        
        cv2.namedWindow('Annotation', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Annotation', mouse_callback)
        
        while True:
            cv2.imshow('Annotation', image_display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s') and save_path:
                self.save_annotations(keypoints_2d, save_path)
                print(f"Saved: {save_path}")
                break
            elif key == ord('r'):
                image_display = image.copy()
                keypoints_2d = {}
                current_keypoint = 0
                print(f"Reset. Click on: {keypoint_names[current_keypoint]}")
        
        cv2.destroyAllWindows()
        return keypoints_2d
    
    def save_annotations(self, keypoints_2d: Dict[str, np.ndarray], filepath: str):
        """Save 2D annotations to JSON file."""
        annotations = {name: point.tolist() for name, point in keypoints_2d.items()}
        with open(filepath, 'w') as f:
            json.dump(annotations, f, indent=2)
    
    def load_annotations(self, filepath: str) -> Dict[str, np.ndarray]:
        """Load 2D annotations from JSON file."""
        with open(filepath, 'r') as f:
            annotations = json.load(f)
        return {name: np.array(point) for name, point in annotations.items()}
    
    def simple_intrinsic_estimation(self, image_width: int, image_height: int, 
                                   fov_degrees: float = 60) -> np.ndarray:
        """Simple camera intrinsic estimation based on field of view."""
        fov_rad = np.deg2rad(fov_degrees)
        focal_length = image_width / (2 * np.tan(fov_rad / 2))
        cx, cy = image_width / 2, image_height / 2
        
        return np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float32)
    
    def display_video_frame_selector(self, video_path: str, joint_states: np.ndarray, 
                                     episode_name: str) -> List[Dict]:
        """Interactive frame selector for choosing calibration frames from video."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        current_frame = 0
        selected_frames = []
        
        print("\nInteractive Frame Selection")
        print(f"Video: {total_frames} frames, {fps:.1f} FPS")
        print("Controls:")
        print("  Space: Select current frame")
        print("  Left/Right arrows: Navigate frames")
        print("  'j'/'k': Jump by 10 frames")
        print("  's': Show selected frames")
        print("  'q': Quit selection")
        print("  'c': Clear all selections")
        
        def display_frame(frame_idx):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Add frame info overlay
                display_frame = frame.copy()
                text = f"Frame: {frame_idx}/{total_frames-1} | Time: {frame_idx/fps:.2f}s"
                if frame_idx < len(joint_states):
                    joint_text = f"Joints: {joint_states[frame_idx][:5]}"
                    cv2.putText(display_frame, joint_text, (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.putText(display_frame, text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Mark if frame is selected
                if frame_idx in [f['frame_index'] for f in selected_frames]:
                    cv2.putText(display_frame, "SELECTED", (display_frame.shape[1]-150, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.imshow('Frame Selection', display_frame)
                return frame
            return None
        
        cv2.namedWindow('Frame Selection', cv2.WINDOW_NORMAL)
        current_frame_img = display_frame(current_frame)
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space to select frame
                if current_frame < len(joint_states):
                    # Check if already selected
                    if current_frame not in [f['frame_index'] for f in selected_frames]:
                        # Create calibration data directory structure  
                        episode_data_dir = os.path.join(CALIBRATION_DATA_FOLDER, episode_name)
                        os.makedirs(episode_data_dir, exist_ok=True)
                        
                        # Save current frame with sample number in filename
                        frame_filename = f"sample_{len(selected_frames):03d}_frame_{current_frame:06d}.jpg"
                        temp_frame_path = os.path.join(episode_data_dir, frame_filename)
                        cv2.imwrite(temp_frame_path, current_frame_img)
                        
                        selected_frames.append({
                            'image_path': temp_frame_path,
                            'joint_angles': joint_states[current_frame][:5],
                            'timestamp': current_frame,
                            'episode': episode_name,
                            'frame_index': current_frame
                        })
                        print(f"Selected frame {current_frame} ({len(selected_frames)} total)")
                    else:
                        print(f"Frame {current_frame} already selected")
                current_frame_img = display_frame(current_frame)
                
            elif key == 83:  # Right arrow
                current_frame = min(current_frame + 1, total_frames - 1)
                current_frame_img = display_frame(current_frame)
            elif key == 81:  # Left arrow
                current_frame = max(current_frame - 1, 0)
                current_frame_img = display_frame(current_frame)
            elif key == ord('j'):  # Jump forward 10 frames
                current_frame = min(current_frame + 10, total_frames - 1)
                current_frame_img = display_frame(current_frame)
            elif key == ord('k'):  # Jump backward 10 frames
                current_frame = max(current_frame - 10, 0)
                current_frame_img = display_frame(current_frame)
            elif key == ord('s'):  # Show selected frames
                print(f"\nSelected frames: {[f['frame_index'] for f in selected_frames]}")
            elif key == ord('c'):  # Clear selections
                for frame_data in selected_frames:
                    if os.path.exists(frame_data['image_path']):
                        os.remove(frame_data['image_path'])
                selected_frames = []
                print("Cleared all selections")
                current_frame_img = display_frame(current_frame)
        
        cv2.destroyAllWindows()
        cap.release()
        
        print(f"\nFinal selection: {len(selected_frames)} frames")
        return selected_frames
    
    def calibrate_single_image(self, joint_angles_deg: np.ndarray, 
                               keypoints_2d: Dict[str, np.ndarray]) -> np.ndarray:
        """Calibrate camera extrinsics from single image."""
        keypoints_3d = self.get_robot_keypoints_3d(joint_angles_deg)
        
        points_3d = []
        points_2d = []
        
        for name in keypoints_2d.keys():
            if name in keypoints_3d:
                points_3d.append(keypoints_3d[name])
                points_2d.append(keypoints_2d[name])
        
        if len(points_3d) < 3:
            raise ValueError(f"Need at least 3 keypoints, got {len(points_3d)}")
        
        points_3d = np.array(points_3d, dtype=np.float32)
        points_2d = np.array(points_2d, dtype=np.float32)
        
        # Use SQPNP method which works well with 3+ points
        success, rvec, tvec = cv2.solvePnP(
            points_3d, points_2d, self.K, self.dist, flags=cv2.SOLVEPNP_SQPNP
        )
        
        if not success:
            raise ValueError("PnP solution failed")
        
        R, _ = cv2.Rodrigues(rvec)
        T_robot_to_camera = np.eye(4)
        T_robot_to_camera[:3, :3] = R
        T_robot_to_camera[:3, 3] = tvec.flatten()
        
        return np.linalg.inv(T_robot_to_camera)
    
    def calibrate_multiple_images(self, dataset_samples: List[Dict]) -> np.ndarray:
        """Calibrate using multiple images for better accuracy."""
        transformations = []
        
        for i, sample in enumerate(dataset_samples):
            try:
                T = self.calibrate_single_image(sample['joint_angles'], sample['keypoints_2d'])
                transformations.append(T)
                print(f"Calibrated image {i+1}/{len(dataset_samples)}")
            except Exception as e:
                print(f"Failed image {i+1}: {e}")
                continue
        
        if not transformations:
            raise ValueError("No successful calibrations")
        
        # Average transformations
        positions = np.array([T[:3, 3] for T in transformations])
        avg_position = np.mean(positions, axis=0)
        
        quaternions = []
        for T in transformations:
            r = Rotation.from_matrix(T[:3, :3])
            quaternions.append(r.as_quat())
        
        quaternions = np.array(quaternions)
        avg_quat = np.mean(quaternions, axis=0)
        avg_quat = avg_quat / np.linalg.norm(avg_quat)
        avg_rotation = Rotation.from_quat(avg_quat).as_matrix()
        
        T_avg = np.eye(4)
        T_avg[:3, :3] = avg_rotation
        T_avg[:3, 3] = avg_position
        
        return T_avg
    
    def calibrate_without_intrinsics(self, dataset_samples: List[Dict], 
                                    image_width: int, image_height: int, 
                                    fov_degrees: float = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Complete calibration pipeline without knowing camera intrinsics."""
        print("Estimating camera intrinsics...")
        camera_matrix = self.simple_intrinsic_estimation(image_width, image_height, fov_degrees)
        self.K = camera_matrix
        self.dist = np.zeros(5)
        
        print("Calibrating extrinsics...")
        T_camera_to_robot = self.calibrate_multiple_images(dataset_samples)
        
        return camera_matrix, T_camera_to_robot
    
    def verify_calibration(self, T_camera_to_robot: np.ndarray, 
                          joint_angles_deg: np.ndarray, 
                          keypoints_2d: Dict[str, np.ndarray]) -> float:
        """Verify calibration by computing reprojection error."""
        keypoints_3d = self.get_robot_keypoints_3d(joint_angles_deg)
        
        points_3d = []
        points_2d = []
        
        for name in keypoints_2d.keys():
            if name in keypoints_3d:
                points_3d.append(keypoints_3d[name])
                points_2d.append(keypoints_2d[name])
        
        points_3d = np.array(points_3d, dtype=np.float32)
        points_2d = np.array(points_2d, dtype=np.float32)
        
        T_robot_to_camera = np.linalg.inv(T_camera_to_robot)
        points_3d_homogeneous = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
        points_3d_camera = (T_robot_to_camera @ points_3d_homogeneous.T).T[:, :3]
        
        projected_2d, _ = cv2.projectPoints(
            points_3d_camera, np.zeros(3), np.zeros(3), self.K, self.dist
        )
        projected_2d = projected_2d.reshape(-1, 2)
        
        errors = np.linalg.norm(projected_2d - points_2d, axis=1)
        return np.mean(errors)


def save_episode_with_extrinsics(episode_path: str, T_camera_to_robot: np.ndarray, output_folder: str):
    """Save episode with camera extrinsics added to the HDF5 file."""
    import shutil
    
    episode_name = os.path.basename(episode_path)
    output_episode_path = os.path.join(output_folder, episode_name)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Copy entire episode directory to output folder
    if os.path.exists(output_episode_path):
        shutil.rmtree(output_episode_path)
    shutil.copytree(episode_path, output_episode_path)
    
    # Get camera name from MP4 file
    episode_data = load_so_episode(episode_path)
    if not episode_data['video_path']:
        raise ValueError("No video file found in episode")
    
    mp4_filename = os.path.basename(episode_data['video_path'])
    camera_name = mp4_filename.split('.')[0]  # Remove .mp4 extension
    
    # Convert transformation matrix to pose format (translation + XYZ euler angles)
    translation = T_camera_to_robot[:3, 3]
    rotation_matrix = T_camera_to_robot[:3, :3]
    euler_angles = Rotation.from_matrix(rotation_matrix).as_euler('xyz', degrees=False)
    
    # Create pose: [x, y, z, rx, ry, rz] in radians
    camera_extrinsics_pose = np.concatenate([translation, euler_angles])
    
    # Update the HDF5 file with camera extrinsics
    h5_path = os.path.join(output_episode_path, 'trajectory.h5')
    
    with h5py.File(h5_path, 'r+') as f:
        # Create camera_extrinsics group if it doesn't exist
        if 'observation' not in f:
            f.create_group('observation')
        
        if 'camera_extrinsics' not in f['observation']:
            f['observation'].create_group('camera_extrinsics')
        
        # Save camera extrinsics
        if camera_name in f['observation']['camera_extrinsics']:
            del f['observation']['camera_extrinsics'][camera_name]
        
        f['observation']['camera_extrinsics'].create_dataset(
            camera_name, 
            data=camera_extrinsics_pose,
            dtype=np.float64
        )
    
    print(f"Episode saved with camera extrinsics to: {output_episode_path}")
    print(f"Camera '{camera_name}' extrinsics: {camera_extrinsics_pose}")
    print(f"  Translation (m): [{translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}]")
    print(f"  Rotation (rad): [{euler_angles[0]:.3f}, {euler_angles[1]:.3f}, {euler_angles[2]:.3f}]")


def calibrate_single_episode(episode_path: str):
    """Calibrate camera extrinsics from a single SO100/SO101 episode with interactive frame selection."""
    
    print(f"Loading episode: {os.path.basename(episode_path)}")
    episode_data = load_so_episode(episode_path)
    
    if not episode_data['trajectory_data'] or not episode_data['video_path']:
        raise ValueError(f"Episode missing trajectory data or video: {episode_path}")
    
    joint_states = episode_data['trajectory_data']['joint_states']
    video_path = episode_data['video_path']
    episode_name = episode_data['name']
   
    if joint_states is None:
        raise ValueError(f"Episode has no joint state data: {episode_path}")
    
    print(f"Episode loaded: {len(joint_states)} frames")
    
    # Auto-detect calibration type
    creation_date = episode_data['metadata'].get('creation_date') if episode_data['metadata'] else None
    calibration_name = detect_so_calibration(joint_states, creation_date)
    print(f"Auto-detected calibration: {calibration_name}")
    
    # Initialize robot kinematics with detected calibration
    robot_kinematics = RobotKinematics(calibration_name)
    
    # Initialize calibrator
    calibrator = DatasetExtrinsicCalibrator(robot_kinematics)
    
    # Interactive frame selection
    print("Starting interactive frame selection...")
    selected_samples = calibrator.display_video_frame_selector(video_path, joint_states, episode_name)
    
    if not selected_samples:
        raise ValueError("No frames selected for calibration")
    
    print(f"Selected {len(selected_samples)} frames for calibration")
    
    # Annotate keypoints
    print("Starting keypoint annotation...")
    annotated_samples = []
    
    for i, sample in enumerate(selected_samples):
        print(f"\n[{i + 1}/{len(selected_samples)}] Processing: {os.path.basename(sample['image_path'])}")
        
        image = cv2.imread(sample['image_path'])
        if image is None:
            print(f"ERROR: Could not load image: {sample['image_path']}")
            continue
        
        keypoints_3d = calibrator.get_robot_keypoints_3d(sample['joint_angles'])
        print(f"Required keypoints: {list(keypoints_3d.keys())}")
        
        # Check for existing annotation file in calibration data folder
        episode_data_dir = os.path.join(CALIBRATION_DATA_FOLDER, episode_name)
        annotation_filename = f"annotations_episode_{sample['episode']}_{i:03d}.json"
        annotation_file = os.path.join(episode_data_dir, annotation_filename)
        
        if os.path.exists(annotation_file):
            print(f"Loading existing annotations from: {annotation_file}")
            keypoints_2d = calibrator.load_annotations(annotation_file)
            # Check if existing annotations have all required keypoints
            required_keypoints = set(keypoints_3d.keys())
            existing_keypoints = set(keypoints_2d.keys())
            if not required_keypoints.issubset(existing_keypoints):
                missing = required_keypoints - existing_keypoints
                print(f"Existing annotations incomplete - missing: {list(missing)}")
                print(f"Re-annotating image...")
                os.makedirs(episode_data_dir, exist_ok=True)
                keypoints_2d = calibrator.annotate_robot_keypoints(image, keypoints_3d, annotation_file)
        else:
            print(f"Starting new annotation...")
            os.makedirs(episode_data_dir, exist_ok=True)
            keypoints_2d = calibrator.annotate_robot_keypoints(image, keypoints_3d, annotation_file)
        
        print(f"Annotated keypoints: {list(keypoints_2d.keys())} (count: {len(keypoints_2d)})")
        print(f"Required count: 3, Actual count: {len(keypoints_2d)}")
        
        if len(keypoints_2d) >= 3:
            annotated_samples.append({
                'joint_angles': sample['joint_angles'],
                'keypoints_2d': keypoints_2d,
                'image_path': sample['image_path']
            })
            print("ACCEPTED: Image added to calibration set")
        else:
            print(f"REJECTED: Not enough keypoints ({len(keypoints_2d)}/3)")
            required_set = set(keypoints_3d.keys())
            annotated_set = set(keypoints_2d.keys())
            missing = required_set - annotated_set
            if missing:
                print(f"Missing keypoints: {list(missing)}")
    
    print(f"\nSUMMARY: Using {len(annotated_samples)} out of {len(selected_samples)} selected images")
    for i, sample in enumerate(annotated_samples):
        keypoint_names = list(sample['keypoints_2d'].keys())
        print(f"  {i+1}. {os.path.basename(sample['image_path'])} -> {keypoint_names}")
    
    if len(annotated_samples) < 1:
        raise ValueError(f"Need at least 1 annotated image, got {len(annotated_samples)}")
        
    print(f"\nProceeding with calibration using {len(annotated_samples)} images...")
    
    # Calibrate camera
    print(f"Calibrating camera with {len(annotated_samples)} images...")
    
    sample_image = cv2.imread(annotated_samples[0]['image_path'])
    image_height, image_width = sample_image.shape[:2]
    
    # Handle single vs multiple images
    if len(annotated_samples) == 1:
        camera_matrix = calibrator.simple_intrinsic_estimation(image_width, image_height, fov_degrees=65)
        calibrator.K = camera_matrix
        calibrator.dist = np.zeros(5)
        T_camera_to_robot = calibrator.calibrate_single_image(
            annotated_samples[0]['joint_angles'], 
            annotated_samples[0]['keypoints_2d']
        )
    else:
        camera_matrix, T_camera_to_robot = calibrator.calibrate_without_intrinsics(
            annotated_samples, image_width, image_height, fov_degrees=65
        )
    
    # Verify calibration
    print("Verification Results:")
    print("=" * 40)
    total_error = 0
    for i, sample in enumerate(annotated_samples):
        error = calibrator.verify_calibration(
            T_camera_to_robot, sample['joint_angles'], sample['keypoints_2d']
        )
        print(f"Image {i + 1}: {error:.2f} pixels")
        total_error += error
    
    avg_error = total_error / len(annotated_samples)
    print(f"Average error: {avg_error:.2f} pixels")
    
    if avg_error < 5:
        print("Good calibration!")
    elif avg_error < 10:
        print("Acceptable calibration")
    else:
        print("Poor calibration - consider adding more images or better keypoint annotations")
    
    # Save results in calibration data folder
    episode_name = os.path.basename(episode_path)
    episode_data_dir = os.path.join(CALIBRATION_DATA_FOLDER, episode_name)
    os.makedirs(episode_data_dir, exist_ok=True)
    
    results_file = os.path.join(episode_data_dir, f"camera_calibration_{episode_name}.json")
    
    results = {
        'episode_path': episode_path,
        'episode_name': episode_name,
        'camera_matrix': camera_matrix.tolist(),
        'camera_to_robot_transform': T_camera_to_robot.tolist(),
        'average_reprojection_error': avg_error,
        'annotated_images': [s['image_path'] for s in annotated_samples],
        'num_samples_used': len(annotated_samples),
        'keypoints_used': ['base', 'gripper', 'gripper_tip']
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to '{results_file}'")
    
    # Save updated episode with camera extrinsics to OUTPUT_FOLDER
    print("\nSaving episode with camera extrinsics...")
    save_episode_with_extrinsics(episode_path, T_camera_to_robot, OUTPUT_FOLDER)
    
    # Note: Calibration data (frames and JSON files) are kept in CALIBRATION_DATA_FOLDER for future reference
    # They are organized by episode name and include sample numbers in filenames
    print(f"Calibration data saved in: {CALIBRATION_DATA_FOLDER}/{episode_name}")
    print(f"Frame files: {[os.path.basename(s['image_path']) for s in annotated_samples]}")
    
    return camera_matrix, T_camera_to_robot

# Usage example
if __name__ == "__main__":
    print("Camera Extrinsic Calibration from SO100/SO101 Episode")
    print("=" * 50)
   
    episode_path = SOURCE_FOLDER + '/smanni+train_so100_all-episode_001'
    camera_matrix, T_camera_to_robot = calibrate_single_episode(episode_path)