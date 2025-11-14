"""
Move Classification Module for NeuroCombat - V2 FIXED
=====================================================

TEMPORAL SMOOTHING - No duplicate moves, clean output

Features:
- Extracts features from keypoints
- Classifies MMA moves with temporal context
- ELIMINATES duplicate consecutive moves
- Outputs only distinct move sequences with start/end times
"""

import json
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm


@dataclass
class Move:
    move_type: str
    confidence: float
    frame_start: int
    frame_end: int
    player_id: int
    keypoints: List[List[float]]
    duration_frames: int = 0

    def __post_init__(self):
        self.duration_frames = self.frame_end - self.frame_start


class MoveClassifier:
    MOVE_TYPES = [
        "jab", "cross", "hook", "uppercut",
        "front_kick", "roundhouse_kick", "side_kick",
        "takedown", "guard", "clinch", "neutral"
    ]

    def __init__(
        self,
        model_path: str = "models/move_classifier.pkl",
        confidence_threshold: float = 0.6,
        window_size: int = 5,
        min_move_duration: int = 3,
        move_cooldown: int = 8
    ):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.window_size = window_size
        self.min_move_duration = min_move_duration
        self.move_cooldown = move_cooldown

        self.model = None
        if self.model_path.exists():
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            print(f"✓ Loaded model from {self.model_path}")
        else:
            print(f"⚠️ Model not found at {self.model_path}")
            print("   Using rule-based fallback")

    # ----------------------------------------------------
    #                PUBLIC FUNCTION
    # ----------------------------------------------------
    def classify_from_json(self, pose_json_path: str, output_path: str = None) -> List[Move]:
        pose_json_path = Path(pose_json_path)

        if not pose_json_path.exists():
            raise FileNotFoundError(f"Pose JSON not found: {pose_json_path}")

        with open(pose_json_path, "r") as f:
            pose_data = json.load(f)

        metadata = pose_data.get("metadata", {})
        frames = pose_data.get("frames", {})

        print(f"Classifying moves from {len(frames)} frames...")

        all_moves = []

        for player_id in [1, 2]:
            player_moves = self._classify_player_moves_smoothed(frames, player_id, metadata)
            all_moves.extend(player_moves)

        all_moves.sort(key=lambda m: m.frame_start)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            output_data = {
                "metadata": metadata,
                "moves": [asdict(m) for m in all_moves]
            }

            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2)

            print(f"✓ Saved {len(all_moves)} cleaned moves to {output_path}")

        return all_moves

    # ----------------------------------------------------
    #                SMOOTHING + PIPELINE
    # ----------------------------------------------------
    def _classify_player_moves_smoothed(self, frames, player_id, metadata):

        player_key = f"player_{player_id}"
        frame_keys = sorted(frames.keys())

        frame_classifications = []

        for i in tqdm(range(len(frame_keys)), desc=f"Classifying Player {player_id}"):
            frame_key = frame_keys[i]
            frame_data = frames[frame_key]

            if player_key not in frame_data:
                frame_classifications.append(None)
                continue

            player_data = frame_data[player_key]
            keypoints = player_data.get("keypoints", [])

            if not keypoints:
                frame_classifications.append(None)
                continue

            features = self._extract_features(keypoints)

            context_features = []
            for j in range(max(0, i - self.window_size), i):
                ctx_key = frame_keys[j]
                if player_key in frames[ctx_key]:
                    ctx_kp = frames[ctx_key][player_key].get("keypoints", [])
                    if ctx_kp:
                        context_features.append(self._extract_features(ctx_kp))

            move_type, confidence = self._classify_frame(features, context_features)

            frame_classifications.append({
                'move_type': move_type,
                'confidence': confidence,
                'frame_num': int(frame_key.split("_")[1]),
                'keypoints': keypoints
            })

        smoothed = self._apply_temporal_smoothing(frame_classifications)
        moves = self._extract_move_sequences(smoothed, player_id, self.min_move_duration)
        moves = self._apply_move_cooldown(moves)

        return moves

    # ----------------------------------------------------
    #            TEMPORAL SMOOTHING
    # ----------------------------------------------------
    def _apply_temporal_smoothing(self, classifications, window_size=5):
        smoothed = []

        for i in range(len(classifications)):
            cls = classifications[i]
            if cls is None:
                smoothed.append(None)
                continue

            start = max(0, i - window_size // 2)
            end = min(len(classifications), i + window_size // 2 + 1)

            window = [c for c in classifications[start:end] if c is not None]

            if not window:
                smoothed.append(None)
                continue

            move_counts = {}
            for c in window:
                m = c['move_type']
                if m != 'neutral':
                    move_counts[m] = move_counts.get(m, 0) + 1

            if not move_counts:
                smoothed.append(cls)
                continue

            best = max(move_counts, key=move_counts.get)
            vote_strength = move_counts[best] / len(window)

            if vote_strength >= 0.4:
                new_c = cls.copy()
                new_c['move_type'] = best
                new_c['confidence'] *= vote_strength
                smoothed.append(new_c)
            else:
                smoothed.append(cls)

        return smoothed

    # ----------------------------------------------------
    #     FIXED VERSION OF _extract_move_sequences
    # ----------------------------------------------------
    def _extract_move_sequences(self, classifications, player_id, min_duration=3):

        moves = []

        current_move = None
        current_start = None

        for i, cls in enumerate(classifications):

            if cls is None:

                if current_move is not None:
                    duration = i - current_start
                    if duration >= min_duration and current_move['move_type'] != 'neutral':
                        moves.append(Move(
                            move_type=current_move['move_type'],
                            confidence=current_move['confidence'],
                            frame_start=current_start,
                            frame_end=i,
                            player_id=player_id,
                            keypoints=current_move['keypoints']
                        ))

                current_move = None
                current_start = None
                continue

            move_type = cls.get('move_type')
            confidence = cls.get('confidence', 0)

            if move_type == 'neutral':
                if current_move is not None:
                    duration = i - current_start
                    if duration >= min_duration:
                        moves.append(Move(
                            move_type=current_move['move_type'],
                            confidence=current_move['confidence'],
                            frame_start=current_start,
                            frame_end=i,
                            player_id=player_id,
                            keypoints=current_move['keypoints']
                        ))
                current_move = None
                current_start = None
                continue

            if confidence < self.confidence_threshold:
                continue

            if current_move is None:
                current_move = cls
                current_start = cls['frame_num']

            elif current_move['move_type'] == move_type:
                current_move['confidence'] = max(current_move['confidence'], confidence)

            else:
                duration = i - current_start
                if duration >= min_duration:
                    moves.append(Move(
                        move_type=current_move['move_type'],
                        confidence=current_move['confidence'],
                        frame_start=current_start,
                        frame_end=cls['frame_num'],
                        player_id=player_id,
                        keypoints=current_move['keypoints']
                    ))

                current_move = cls
                current_start = cls['frame_num']

        if current_move is not None:
            duration = len(classifications) - current_start
            if duration >= min_duration and current_move['move_type'] != 'neutral':
                moves.append(Move(
                    move_type=current_move['move_type'],
                    confidence=current_move['confidence'],
                    frame_start=current_start,
                    frame_end=len(classifications),
                    player_id=player_id,
                    keypoints=current_move['keypoints']
                ))

        return moves

    # ----------------------------------------------------
    #                COOLDOWN LOGIC
    # ----------------------------------------------------
    def _apply_move_cooldown(self, moves):
        if not moves:
            return moves

        filtered = [moves[0]]

        for mv in moves[1:]:
            last = filtered[-1]
            if mv.move_type == last.move_type:
                frames_since = mv.frame_start - last.frame_end

                if frames_since < self.move_cooldown:
                    last.frame_end = mv.frame_end
                    last.confidence = max(last.confidence, mv.confidence)
                    last.duration_frames = last.frame_end - last.frame_start
                else:
                    filtered.append(mv)
            else:
                filtered.append(mv)

        return filtered

    # ----------------------------------------------------
    #                FEATURE EXTRACTION
    # ----------------------------------------------------
    def _extract_features(self, keypoints):

        if not keypoints or len(keypoints) < 33:
            return np.zeros(50)

        kp = np.array(keypoints)
        features = []

        left_shoulder = kp[11][:2]
        right_shoulder = kp[12][:2]
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
        features.append(shoulder_width)

        left_wrist = kp[15][:2]
        right_wrist = kp[16][:2]

        left_arm_ext = np.linalg.norm(left_shoulder - left_wrist) / (shoulder_width + 1e-6)
        right_arm_ext = np.linalg.norm(right_shoulder - right_wrist) / (shoulder_width + 1e-6)
        features.extend([left_arm_ext, right_arm_ext])

        left_ankle = kp[27][:2]
        right_ankle = kp[28][:2]
        left_leg_ext = np.linalg.norm(kp[23][:2] - left_ankle) / (shoulder_width + 1e-6)
        right_leg_ext = np.linalg.norm(kp[24][:2] - right_ankle) / (shoulder_width + 1e-6)
        features.extend([left_leg_ext, right_leg_ext])

        left_elbow_angle = self._calculate_angle(kp[11][:2], kp[13][:2], kp[15][:2])
        right_elbow_angle = self._calculate_angle(kp[12][:2], kp[14][:2], kp[16][:2])
        features.extend([left_elbow_angle, right_elbow_angle])

        left_knee_angle = self._calculate_angle(kp[23][:2], kp[25][:2], kp[27][:2])
        right_knee_angle = self._calculate_angle(kp[24][:2], kp[26][:2], kp[28][:2])
        features.extend([left_knee_angle, right_knee_angle])

        visible = kp[kp[:, 2] > 0.3][:, :2]
        if len(visible) > 0:
            com = np.mean(visible, axis=0)
            features.extend(com.tolist())
        else:
            features.extend([0, 0])

        while len(features) < 50:
            features.append(0)

        return np.array(features[:50])

    def _calculate_angle(self, p1, p2, p3):
        v1 = p1 - p2
        v2 = p3 - p2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

    # ----------------------------------------------------
    #                FRAME CLASSIFICATION
    # ----------------------------------------------------
    def _classify_frame(self, features, context_features):

        if self.model is not None:
            try:
                features_2d = features.reshape(1, -1)
                prediction = self.model.predict(features_2d)[0]

                if hasattr(self.model, 'predict_proba'):
                    probs = self.model.predict_proba(features_2d)[0]
                    conf = float(np.max(probs))
                else:
                    conf = 0.7

                return prediction, conf

            except:
                pass

        return self._rule_based_classification(features, context_features)

    # ----------------------------------------------------
    #                RULE BASED BACKUP
    # ----------------------------------------------------
    def _rule_based_classification(self, features, context_features):

        if len(features) < 11:
            return "neutral", 0.5

        shoulder = features[0]
        la, ra = features[1], features[2]
        ll, rl = features[3], features[4]
        lea, rea = features[5], features[6]
        lka, rka = features[7], features[8]

        if (la > 1.5 or ra > 1.5) and (lea > 160 or rea > 160):
            return "jab", 0.75

        if (la > 1.2 or ra > 1.2) and (80 < lea < 120 or 80 < rea < 120):
            return "hook", 0.70

        if (ll > 1.8 or rl > 1.8) and (lka > 140 or rka > 140):
            return "front_kick", 0.72

        if (la < 1.0 and ra < 1.0) and (100 < lka < 170 and 100 < rka < 170):
            return "guard", 0.65

        return "neutral", 0.6
