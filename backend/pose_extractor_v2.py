"""
Pose Extraction Module for NeuroCombat - V2 FIXED
==================================================

STABLE PLAYER TRACKING - Players maintain consistent IDs throughout video

Features:
- Dual-fighter detection using MediaPipe Pose
- Rock-solid player ID assignment with position anchoring
- Hungarian algorithm with strict validation
- Position history smoothing
- Handles occlusions, crossovers, and rotations
"""

import cv2
import mediapipe as mp
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from collections import deque


@dataclass
class Pose:
    keypoints: List[List[float]]
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    confidence: float
    player_id: Optional[int] = None


class PoseExtractor:
    POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS

    def __init__(self, confidence_threshold: float = 0.5, verbose_logging: bool = False):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.confidence_threshold = confidence_threshold
        self.verbose_logging = verbose_logging

        self.detector = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=confidence_threshold,
            min_tracking_confidence=confidence_threshold,
            enable_segmentation=False
        )

        self.frame_count = 0
        
        # ENHANCED TRACKING STATE
        self.player_positions = {1: None, 2: None}  # Current positions
        self.position_history = {1: deque(maxlen=10), 2: deque(maxlen=10)}  # Position history
        self.anchor_positions = {1: None, 2: None}  # Initial anchor positions
        self.lost_frames = {1: 0, 2: 0}  # Frames since last detection
        self.players_initialized = False
        
        # Tracking parameters
        self.MAX_DISTANCE_THRESHOLD = 200  # Stricter than before
        self.MAX_LOST_FRAMES = 30  # Reset tracking after this many lost frames
        self.MIN_POSITION_CHANGE = 150  # Minimum distance for player swap validation

    def extract_poses_from_video(
        self,
        video_path: str,
        output_json: str = None,
        overlay_video: str = None,
        display: bool = False
    ) -> Dict:

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.verbose_logging:
            print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")

        poses_output = {
            "metadata": {
                "video_name": video_path.name,
                "resolution": [width, height],
                "fps": fps,
                "total_frames": total_frames
            },
            "frames": {}
        }

        writer = None
        if overlay_video:
            overlay_path = Path(overlay_video)
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            
            codecs_to_try = [
                ('avc1', '.mp4'),
                ('H264', '.mp4'),
                ('X264', '.mp4'),
                ('mp4v', '.mp4'),
            ]
            
            for codec, ext in codecs_to_try:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    temp_writer = cv2.VideoWriter(
                        str(overlay_path), 
                        fourcc, 
                        fps, 
                        (width, height)
                    )
                    
                    if temp_writer.isOpened():
                        writer = temp_writer
                        if self.verbose_logging:
                            print(f"✓ Using video codec: {codec}")
                        break
                    else:
                        temp_writer.release()
                except Exception as e:
                    if self.verbose_logging:
                        print(f"  Codec {codec} failed: {e}")
                    continue
            
            if writer is None:
                print("⚠️ Warning: Could not initialize video writer")

        frame_id = 0
        with tqdm(total=total_frames, desc="Extracting poses", unit="frame") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                self.frame_count += 1
                frame_id += 1

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                poses = self._detect_all_poses(rgb, frame.shape)
                tracked = self._track_and_assign_players_stable(poses)

                frame_key = f"frame_{frame_id:06d}"
                poses_output["frames"][frame_key] = self._frame_to_json(tracked)

                overlay = self._draw_overlay(frame, tracked)

                if writer and writer.isOpened():
                    writer.write(overlay)

                if display:
                    cv2.imshow("Overlay", overlay)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                pbar.update(1)

        cap.release()
        
        if writer:
            writer.release()
            import time
            time.sleep(0.5)
            
        if display:
            cv2.destroyAllWindows()

        if output_json:
            with open(output_json, "w") as f:
                json.dump(poses_output, f, indent=2)

        if self.verbose_logging:
            print(f"✓ Processed {frame_id} frames")
            print(f"✓ JSON saved to: {output_json}")
            if overlay_video:
                print(f"✓ Overlay saved to: {overlay_video}")

        return poses_output

    def _detect_all_poses(self, rgb_frame, frame_shape) -> List[Pose]:
        h, w = frame_shape[:2]

        result = self.detector.process(rgb_frame)
        
        poses = []
        
        if result.pose_landmarks:
            pose1 = self._landmarks_to_pose(result.pose_landmarks.landmark, w, h)
            if pose1:
                poses.append(pose1)

        if len(poses) == 1 and result.pose_landmarks:
            mask = np.ones((h, w, 3), dtype=np.uint8) * 255
            x, y, bw, bh = poses[0].bbox
            
            mask_padding = 50
            x1 = max(0, x - mask_padding)
            y1 = max(0, y - mask_padding)
            x2 = min(w, x + bw + mask_padding)
            y2 = min(h, y + bh + mask_padding)
            
            mask[y1:y2, x1:x2] = 0
            
            masked_frame = cv2.bitwise_and(rgb_frame, mask)
            
            result2 = self.detector.process(masked_frame)
            
            if result2.pose_landmarks:
                pose2 = self._landmarks_to_pose(result2.pose_landmarks.landmark, w, h)
                if pose2:
                    c1 = poses[0].centroid
                    c2 = pose2.centroid
                    distance = np.linalg.norm(np.array(c1) - np.array(c2))
                    
                    if distance > 100:
                        poses.append(pose2)

        return poses

    def _landmarks_to_pose(self, landmarks, w: int, h: int) -> Optional[Pose]:
        keypoints = []
        for p in landmarks:
            keypoints.append([
                int(p.x * w),
                int(p.y * h),
                float(p.visibility)
            ])

        avg_visibility = np.mean([p.visibility for p in landmarks])
        if avg_visibility < self.confidence_threshold:
            return None

        bbox = self._bbox_from_keypoints(keypoints, w, h)
        centroid = self._centroid_from_keypoints(keypoints)
        conf = float(avg_visibility)

        return Pose(keypoints, bbox, centroid, conf, player_id=None)

    def _track_and_assign_players_stable(self, poses: List[Pose]) -> Dict[int, Optional[Pose]]:
        """
        STABLE TRACKING: Players maintain consistent IDs throughout video
        """
        result = {1: None, 2: None}

        if len(poses) == 0:
            # Increment lost frame counters
            self.lost_frames[1] += 1
            self.lost_frames[2] += 1
            
            # Reset tracking if lost for too long
            if self.lost_frames[1] > self.MAX_LOST_FRAMES:
                self.player_positions[1] = None
            if self.lost_frames[2] > self.MAX_LOST_FRAMES:
                self.player_positions[2] = None
                
            return result

        # INITIALIZATION: First frame with poses
        if not self.players_initialized:
            if len(poses) == 1:
                poses[0].player_id = 1
                result[1] = poses[0]
                self.player_positions[1] = poses[0].centroid
                self.anchor_positions[1] = poses[0].centroid
                self.position_history[1].append(poses[0].centroid)
                self.lost_frames[1] = 0
            elif len(poses) >= 2:
                # LEFT = Player 1, RIGHT = Player 2
                sorted_poses = sorted(poses, key=lambda p: p.centroid[0])
                
                sorted_poses[0].player_id = 1
                sorted_poses[1].player_id = 2
                
                result[1] = sorted_poses[0]
                result[2] = sorted_poses[1]
                
                self.player_positions[1] = sorted_poses[0].centroid
                self.player_positions[2] = sorted_poses[1].centroid
                
                self.anchor_positions[1] = sorted_poses[0].centroid
                self.anchor_positions[2] = sorted_poses[1].centroid
                
                self.position_history[1].append(sorted_poses[0].centroid)
                self.position_history[2].append(sorted_poses[1].centroid)
                
                self.lost_frames[1] = 0
                self.lost_frames[2] = 0
            
            self.players_initialized = True
            return result

        # TRACKING: Subsequent frames with STABILITY VALIDATION
        active_players = [pid for pid in [1, 2] if self.player_positions[pid] is not None]
        
        if not active_players:
            # Reinitialize if all players lost
            self.players_initialized = False
            return self._track_and_assign_players_stable(poses)
        
        # Use smoothed positions for better tracking
        prev_centroids = [self._get_smoothed_position(pid) for pid in active_players]
        curr_centroids = [p.centroid for p in poses]

        # Build cost matrix
        cost = np.zeros((len(active_players), len(curr_centroids)))
        for i, prev_c in enumerate(prev_centroids):
            for j, curr_c in enumerate(curr_centroids):
                cost[i, j] = np.linalg.norm(np.array(prev_c) - np.array(curr_c))

        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost)

        # STRICT VALIDATION: Check for unrealistic assignments
        validated_assignments = []
        for prev_idx, curr_idx in zip(row_ind, col_ind):
            distance = cost[prev_idx, curr_idx]
            player_id = active_players[prev_idx]
            
            # Validate assignment
            if distance < self.MAX_DISTANCE_THRESHOLD:
                # Additional check: Prevent sudden position swaps
                if self._validate_position_continuity(player_id, poses[curr_idx].centroid):
                    validated_assignments.append((player_id, curr_idx))
                    
                    # Update tracking state
                    poses[curr_idx].player_id = player_id
                    result[player_id] = poses[curr_idx]
                    self.player_positions[player_id] = poses[curr_idx].centroid
                    self.position_history[player_id].append(poses[curr_idx].centroid)
                    self.lost_frames[player_id] = 0

        # Handle unmatched players (increase lost counter)
        matched_players = {pid for pid, _ in validated_assignments}
        for pid in active_players:
            if pid not in matched_players:
                self.lost_frames[pid] += 1
                if self.lost_frames[pid] > self.MAX_LOST_FRAMES:
                    self.player_positions[pid] = None

        # Handle unmatched detections (new fighter or reappearance)
        matched_curr = {curr_idx for _, curr_idx in validated_assignments}
        for idx, pose in enumerate(poses):
            if idx not in matched_curr:
                # Try to assign to empty slot or lost player
                for pid in [1, 2]:
                    if result[pid] is None:
                        # Validate against anchor position if available
                        if self._validate_against_anchor(pid, pose.centroid):
                            pose.player_id = pid
                            result[pid] = pose
                            self.player_positions[pid] = pose.centroid
                            self.position_history[pid].append(pose.centroid)
                            self.lost_frames[pid] = 0
                            break

        return result

    def _get_smoothed_position(self, player_id: int) -> Tuple[float, float]:
        """Get smoothed position from history"""
        history = self.position_history[player_id]
        if len(history) == 0:
            return self.player_positions[player_id]
        
        # Use median of recent positions for robustness
        recent = list(history)[-5:]
        x_vals = [p[0] for p in recent]
        y_vals = [p[1] for p in recent]
        
        return (np.median(x_vals), np.median(y_vals))

    def _validate_position_continuity(self, player_id: int, new_position: Tuple[float, float]) -> bool:
        """Validate that new position makes sense given history"""
        if len(self.position_history[player_id]) < 3:
            return True  # Not enough history
        
        # Check if position change is too extreme
        smoothed = self._get_smoothed_position(player_id)
        distance = np.linalg.norm(np.array(smoothed) - np.array(new_position))
        
        return distance < self.MIN_POSITION_CHANGE

    def _validate_against_anchor(self, player_id: int, position: Tuple[float, float]) -> bool:
        """Validate position against initial anchor"""
        if self.anchor_positions[player_id] is None:
            return True
        
        anchor = self.anchor_positions[player_id]
        distance = np.linalg.norm(np.array(anchor) - np.array(position))
        
        # More lenient for reappearances
        return distance < self.MIN_POSITION_CHANGE * 2

    def _draw_overlay(self, frame, tracked_poses: Dict[int, Optional[Pose]]):
        overlay = frame.copy()
        
        colors = {
            1: (0, 0, 255),    # RED for Player 1
            2: (255, 0, 0)     # BLUE for Player 2
        }
        
        labels = {
            1: "Player 1",
            2: "Player 2"
        }

        for player_id, pose in tracked_poses.items():
            if pose is None:
                continue

            color = colors[player_id]
            label = labels[player_id]
            kp = pose.keypoints

            # Draw skeleton
            for connection in self.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                if (kp[start_idx][2] > 0.3 and kp[end_idx][2] > 0.3):
                    start_point = (kp[start_idx][0], kp[start_idx][1])
                    end_point = (kp[end_idx][0], kp[end_idx][1])
                    cv2.line(overlay, start_point, end_point, color, 3, cv2.LINE_AA)

            # Draw keypoints
            for x, y, visibility in kp:
                if visibility > 0.3:
                    cv2.circle(overlay, (x, y), 5, color, -1)
                    cv2.circle(overlay, (x, y), 6, (255, 255, 255), 1)

            # Draw bounding box
            x, y, w, h = pose.bbox
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 3)

            # Draw player label with background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            font_thickness = 3
            
            (text_w, text_h), baseline = cv2.getTextSize(
                label, font, font_scale, font_thickness
            )
            
            label_x = x
            label_y = y - 15
            
            cv2.rectangle(
                overlay,
                (label_x, label_y - text_h - 10),
                (label_x + text_w + 10, label_y + 5),
                color,
                -1
            )
            
            cv2.putText(
                overlay,
                label,
                (label_x + 5, label_y),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness,
                cv2.LINE_AA
            )
            
            # Confidence
            conf_text = f"{pose.confidence*100:.0f}%"
            cv2.putText(
                overlay,
                conf_text,
                (x, y + h + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA
            )

        return overlay

    def _bbox_from_keypoints(self, kp, w, h):
        visible = [(x, y) for x, y, v in kp if v > 0.3]
        if not visible:
            return (0, 0, 0, 0)

        xs = [x for x, _ in visible]
        ys = [y for _, y in visible]

        x1, x2 = max(0, min(xs)), min(w, max(xs))
        y1, y2 = max(0, min(ys)), min(h, max(ys))

        padding = 30
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        return (x1, y1, x2 - x1, y2 - y1)

    def _centroid_from_keypoints(self, kp):
        visible = [(x, y) for x, y, v in kp if v > 0.3]
        if not visible:
            return (0.0, 0.0)
        xs = [x for x, _ in visible]
        ys = [y for _, y in visible]
        return (float(np.mean(xs)), float(np.mean(ys)))

    def _frame_to_json(self, tracked: Dict[int, Optional[Pose]]) -> Dict:
        out = {}
        
        for player_id, pose in tracked.items():
            if pose is None:
                continue
            
            key = f"player_{player_id}"
            out[key] = {
                "keypoints": pose.keypoints,
                "bbox": list(pose.bbox),
                "centroid": list(pose.centroid),
                "confidence": float(pose.confidence),
                "player_id": player_id
            }
        
        return out