"""
Pose Extractor Module (MediaPipe 0.10+ Compatible)
==================================================
Fixed version that:
- Draws skeleton properly
- Uses correct protobuf classes (landmark_pb2)
- Supports tracking pipeline
- Safe for Streamlit video writing
"""

import cv2
import numpy as np
import mediapipe as mp

from dataclasses import dataclass
from typing import List, Tuple
from mediapipe.framework.formats import landmark_pb2

from .utils import setup_logging

logger = setup_logging(__name__)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


@dataclass
class PoseLandmarks:
    landmarks: np.ndarray          # (33, 3)
    visibility: np.ndarray         # (33,)
    bbox: Tuple[int, int, int, int]
    player_id: int
    timestamp: float


class PoseExtractor:
    def __init__(
        self,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
    ):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        logger.info("[PoseExtractor] Initialized (MediaPipe 0.10 compatible)")

    # -------------------------------------------------------------
    def extract_poses_from_frame(self, frame, timestamp):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        poses = []

        if results.pose_landmarks:
            pts = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
            vis = np.array([lm.visibility for lm in results.pose_landmarks.landmark])

            bbox = self._calculate_bbox(results.pose_landmarks, frame.shape)

            poses.append(
                PoseLandmarks(
                    landmarks=pts,
                    visibility=vis,
                    bbox=bbox,
                    player_id=0,
                    timestamp=timestamp,
                )
            )

        return poses

    # -------------------------------------------------------------
    def draw_poses_on_frame(self, frame, poses):
        annotated = frame.copy()

        for pose in poses:
            x, y, w, h = pose.bbox
            color = (0, 255, 0) if pose.player_id == 1 else (255, 0, 0)

            # Draw box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            # Label
            cv2.putText(
                annotated,
                f"Player {pose.player_id}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

            # Create landmark list in protobuf format
            landmark_list = landmark_pb2.NormalizedLandmarkList(
                landmark=[
                    landmark_pb2.NormalizedLandmark(
                        x=float(lm[0]),
                        y=float(lm[1]),
                        z=float(lm[2]),
                        visibility=float(vis)
                    )
                    for lm, vis in zip(pose.landmarks, pose.visibility)
                ]
            )

            # Draw skeleton using MediaPipe utilities
            mp_drawing.draw_landmarks(
                annotated,
                landmark_list,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
            )

        return annotated

    # -------------------------------------------------------------
    def _calculate_bbox(self, landmarks, shape):
        h, w = shape[:2]
        xs = [lm.x * w for lm in landmarks.landmark]
        ys = [lm.y * h for lm in landmarks.landmark]

        x_min, x_max = int(min(xs)), int(max(xs))
        y_min, y_max = int(min(ys)), int(max(ys))

        pad = 15
        x_min = max(0, x_min - pad)
        y_min = max(0, y_min - pad)
        x_max = min(w, x_max + pad)
        y_max = min(h, y_max + pad)

        return (x_min, y_min, x_max - x_min, y_max - y_min)

    # -------------------------------------------------------------
    def close(self):
        self.pose.close()
        logger.info("[PoseExtractor] Closed")

    def __enter__(self):
        return self

    def __exit__(self, t, v, tb):
        self.close()
