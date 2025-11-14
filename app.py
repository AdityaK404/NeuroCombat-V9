"""
NeuroCombat - AI-Powered MMA Fight Commentary System
=====================================================
Streamlit UI for real-time fight analysis and commentary generation.

Run with: streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

# Import backend modules (unchanged)
from backend.pose_extractor import PoseExtractor
from backend.tracker import PlayerTracker
from backend.move_classifier import MoveClassifier
from backend.commentary_engine import CommentaryEngine
from backend.utils import setup_logging, format_timestamp

# Optional: import mediapipe only to get POSE_CONNECTIONS for drawing connections.
# If mediapipe import exists in venv this will work; otherwise connections can be hard-coded.
try:
    import mediapipe as mp
    POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS
except Exception:
    # Minimal fallback connections (pairs of indices) - common MediaPipe pose connections
    POSE_CONNECTIONS = [
        (11, 13),(13,15),(12,14),(14,16),(11,12),(23,24),(23,25),(24,26),
        (25,27),(26,28),(27,29),(28,30),(29,31),(30,32),(11,23),(12,24)
    ]

logger = setup_logging(__name__)


# Page configuration
st.set_page_config(
    page_title="NeuroCombat - AI Fight Commentary",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .commentary-box {
        background-color: #1E1E1E;
        border-left: 4px solid #FF6B6B;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        color: #FFFFFF;
    }
    .stats-box {
        background-color: #2E2E2E;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'processed_video' not in st.session_state:
        st.session_state.processed_video = None
    if 'commentary_events' not in st.session_state:
        st.session_state.commentary_events = []
    if 'fight_stats' not in st.session_state:
        st.session_state.fight_stats = {}
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False


# ---------- Helper drawing (replaces backend proto-based drawing) ----------
def draw_poses_simple(frame: np.ndarray, poses: List, color_player1: Tuple[int,int,int]=(0,0,255), color_player2: Tuple[int,int,int]=(255,0,0)) -> np.ndarray:
    """
    Draw bounding boxes, keypoints and skeleton lines on frame.
    - 'poses' is a list of PoseLandmarks-like objects (with .landmarks (N x 3 normalized), .bbox, .player_id)
    """
    h, w = frame.shape[:2]
    out = frame.copy()

    for pose in poses:
        # Player color
        color = color_player1 if getattr(pose, "player_id", 0) == 1 else color_player2

        # Draw bbox if available
        try:
            x, y, bw, bh = pose.bbox
            cv2.rectangle(out, (int(x), int(y)), (int(x + bw), int(y + bh)), color, 2)
            label = f"Player {pose.player_id}" if getattr(pose, "player_id", 0) else "Untracked"
            cv2.putText(out, label, (int(x), max(12, int(y) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        except Exception:
            pass

        # Landmarks: expect normalized coordinates in pose.landmarks shape (33, 3)
        landmarks = getattr(pose, "landmarks", None)
        if landmarks is None:
            continue

        # If landmarks are in normalized [0..1], convert to pixels.
        # Some implementations already multiply by width/height; handle both.
        kpts = []
        for lm in landmarks:
            lx, ly = lm[0], lm[1]
            # If small floats -> assume normalized
            if 0.0 <= lx <= 1.0 and 0.0 <= ly <= 1.0:
                px = int(lx * w)
                py = int(ly * h)
            else:
                px = int(lx)
                py = int(ly)
            kpts.append((px, py, float(lm[2]) if len(lm) > 2 else 1.0))

        # Draw skeleton connections
        for (start_idx, end_idx) in POSE_CONNECTIONS:
            if start_idx < len(kpts) and end_idx < len(kpts):
                s = kpts[start_idx]
                e = kpts[end_idx]
                if s[2] > 0.3 and e[2] > 0.3:
                    cv2.line(out, (s[0], s[1]), (e[0], e[1]), color, 2)

        # Draw keypoints
        for (px, py, vis) in kpts:
            if vis > 0.3:
                cv2.circle(out, (px, py), 3, color, -1)

    return out
# -------------------------------------------------------------------------


def process_video(
    video_path: str,
    progress_bar,
    status_text,
) -> tuple:
    """
    Process uploaded video through the full pipeline.

    Returns:
        Tuple of (commentary_events, fight_stats, annotated_video_path)
    """
    # Initialize pipeline components
    status_text.text("🔧 Initializing AI components...")

    pose_extractor = PoseExtractor(
        min_detection_confidence=0.5,
        model_complexity=1,
    )
    tracker = PlayerTracker(
        iou_threshold=0.3,
        max_missing_frames=30,
    )
    move_classifier = MoveClassifier(
        window_size=15,
        confidence_threshold=0.6,
        use_mock=True,  # demo
    )
    commentary_engine = CommentaryEngine(
        min_time_between_comments=2.0,
    )

    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    # Create output video writer (mp4)
    output_fd, output_path = tempfile.mkstemp(suffix='.mp4')
    os.close(output_fd)  # we'll open via cv2 writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))

    # Process frames
    commentary_events = []
    frame_idx = 0

    status_text.text("🎥 Processing video frames...")

    # guard against total_frames 0 (some containers)
    if total_frames == 0:
        # we will still iterate until cap returns False
        total_frames = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Update progress
        progress = min(1.0, (frame_idx + 1) / total_frames)
        progress_bar.progress(progress)

        timestamp = frame_idx / (fps or 1.0)

        # Extract poses (pose_extractor expects BGR; it does its own conversion)
        poses = pose_extractor.extract_poses_from_frame(frame, timestamp)

        # Track players (tracker.update expected to return list of pose objects)
        tracked_poses = tracker.update(poses, frame_idx)

        # Classify moves & generate commentary
        for pose in tracked_poses:
            classification = move_classifier.classify_move(pose)
            if classification:
                commentary = commentary_engine.generate_commentary(classification)
                if commentary:
                    commentary_events.append(commentary)

        # Annotate frame: use local safe drawer to avoid MediaPipe proto usage
        annotated_frame = draw_poses_simple(frame, tracked_poses)

        # Add commentary overlay (latest)
        if commentary_events:
            latest = commentary_events[-1]
            cv2.putText(
                annotated_frame,
                latest.text,
                (50, max(50, height - 50)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        out.write(annotated_frame)
        frame_idx += 1

        # Update status
        if frame_idx % 30 == 0:
            status_text.text(f"🎥 Processed {frame_idx}/{total_frames} frames...")

    # Cleanup
    cap.release()
    out.release()
    pose_extractor.close()

    # Wait a tiny bit and ensure file is flushed to disk (Windows)
    time.sleep(0.2)

    # Try to ensure file is readable (small retry loop)
    tries = 0
    while tries < 10:
        try:
            if os.path.exists(output_path) and os.path.getsize(output_path) > 512:
                break
        except Exception:
            pass
        time.sleep(0.15)
        tries += 1

    # Get fight statistics
    fight_stats = commentary_engine.get_fight_summary()
    status_text.text("✅ Processing complete!")

    return commentary_events, fight_stats, output_path


def render_sidebar():
    """Render sidebar with configuration options."""
    st.sidebar.markdown("## ⚙️ Configuration")

    st.sidebar.markdown("### Detection Settings")
    detection_confidence = st.sidebar.slider(
        "Detection Confidence",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence for pose detection"
    )

    st.sidebar.markdown("### Classifier Settings")
    classifier_mode = st.sidebar.selectbox(
        "Classifier Mode",
        options=["Mock (Rule-based)", "ML Model"],
        index=0,
        help="Use mock predictions or trained ML model"
    )

    st.sidebar.markdown("### Commentary Settings")
    commentary_speed = st.sidebar.slider(
        "Commentary Frequency",
        min_value=1.0,
        max_value=5.0,
        value=2.0,
        step=0.5,
        help="Minimum seconds between commentary lines"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 About")
    st.sidebar.info(
        "**NeuroCombat** uses AI to analyze MMA fights in real-time, "
        "detecting fighter poses, classifying moves, and generating "
        "dynamic commentary."
    )

    return {
        "detection_confidence": detection_confidence,
        "classifier_mode": classifier_mode,
        "commentary_speed": commentary_speed,
    }


def render_main_content():
    """Render main content area."""
    # Header
    st.markdown('<h1 class="main-header">🥊 NeuroCombat 🥊</h1>', unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; font-size: 1.2rem; color: #888;'>"
        "AI-Powered Real-Time MMA Fight Commentary System"
        "</p>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    # Video upload section
    st.markdown("## 📤 Upload Fight Video")
    uploaded_file = st.file_uploader(
        "Choose an MMA fight video",
        type=["mp4", "avi", "mov", "mkv"],
        help="Upload a video file of an MMA fight"
    )

    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name

        st.success(f"✅ Uploaded: {uploaded_file.name}")

        # Preview original video
        st.markdown("### 🎬 Original Video")
        st.video(video_path)

        # Process button
        if st.button("🚀 Analyze Fight & Generate Commentary", type="primary"):
            with st.spinner("Processing video... This may take a few minutes."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    # Process video
                    commentary_events, fight_stats, output_path = process_video(
                        video_path,
                        progress_bar,
                        status_text,
                    )

                    # Store in session state
                    st.session_state.commentary_events = commentary_events
                    st.session_state.fight_stats = fight_stats
                    st.session_state.processed_video = output_path
                    st.session_state.processing_complete = True

                    st.success("🎉 Fight analysis complete!")

                    # Immediately show results area below without forcing full rerun
                    # (rerun can also be used but we simply call render_results)
                    render_results()

                except Exception as e:
                    st.error(f"❌ Error processing video: {str(e)}")
                    logger.error(f"Processing error: {e}", exc_info=True)

    # Display results if processing is complete and not already shown
    if st.session_state.processing_complete and not st.session_state.get("_results_shown", False):
        render_results()
        st.session_state["_results_shown"] = True


def safe_st_video_from_file(path: str, max_wait: float = 5.0):
    """
    Safely load a video file into Streamlit by reading bytes after writer closed.
    Waits briefly until file appears non-empty (to avoid black video on Windows).
    """
    start = time.time()
    while time.time() - start < max_wait:
        try:
            if os.path.exists(path) and os.path.getsize(path) > 512:
                with open(path, "rb") as f:
                    data = f.read()
                st.video(data)
                return True
        except Exception:
            pass
        time.sleep(0.15)
    # Fallback: give Streamlit the path string (may show black on some Windows setups)
    try:
        st.video(path)
        return True
    except Exception:
        st.error("Unable to load processed video for preview.")
        return False


def render_results():
    """Render analysis results."""
    st.markdown("---")
    st.markdown("## 📊 Fight Analysis Results")

    # Create two columns
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 🎥 Analyzed Video")
        if st.session_state.processed_video:
            safe_st_video_from_file(st.session_state.processed_video)

    with col2:
        st.markdown("### 📈 Fight Statistics")
        stats = st.session_state.fight_stats or {}

        st.markdown('<div class="stats-box">', unsafe_allow_html=True)
        st.metric("Total Commentary Lines", stats.get("total_commentary_lines", len(st.session_state.commentary_events)))
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="stats-box">', unsafe_allow_html=True)
        st.metric("Player 1 Moves", stats.get("total_moves", {}).get(1, 0))
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="stats-box">', unsafe_allow_html=True)
        st.metric("Player 2 Moves", stats.get("total_moves", {}).get(2, 0))
        st.markdown('</div>', unsafe_allow_html=True)

    # Commentary timeline
    st.markdown("### 💬 Live Commentary")

    if st.session_state.commentary_events:
        for event in st.session_state.commentary_events:
            timestamp_str = format_timestamp(event.timestamp)
            st.markdown(
                f'<div class="commentary-box">'
                f'<strong>[{timestamp_str}]</strong> {event.text}'
                f'</div>',
                unsafe_allow_html=True
            )
    else:
        st.info("No commentary generated. Try adjusting detection settings.")

    # Download options
    st.markdown("---")
    st.markdown("### 💾 Export Options")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📥 Download Annotated Video", key="download_video_button"):
            st.info("Video download will be implemented in production version.")

    with col2:
        if st.button("📄 Download Commentary Transcript", key="download_transcript_button"):
            st.info("Commentary export will be implemented in production version.")


def main():
    """Main application entry point."""
    initialize_session_state()

    # Render sidebar
    _ = render_sidebar()

    # Render main content
    render_main_content()


if __name__ == "__main__":
    main()
