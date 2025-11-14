"""
NeuroCombat V2 - Premium AI MMA Fight Commentary System
========================================================

Modern, sleek, production-ready Streamlit UI

Run with:
    streamlit run app_v2.py
"""

import streamlit as st
import cv2
import json
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import asdict
import plotly.graph_objects as go
import plotly.express as px

# Backend imports
try:
    from backend.pose_extractor_v2 import PoseExtractor
    from backend.move_classifier_v2 import MoveClassifier
    from backend.commentary_engine_v2 import CommentaryEngine, CommentaryLine
except ImportError:
    st.error("❌ Backend modules missing. Ensure backend/*.py exist.")
    st.stop()


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="NeuroCombat - AI Fight Analyst",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PREMIUM CSS THEME
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

/* Dark theme background */
.stApp {
    background: linear-gradient(135deg, #0a0e27 0%, #1a1a2e 50%, #16213e 100%);
}

/* Hide default Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Main header styling */
.main-header {
    font-size: 4rem;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(135deg, #ff4b4b 0%, #ff8c00 50%, #ffd700 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 30px rgba(255, 75, 75, 0.3);
    margin-bottom: 0.5rem;
    letter-spacing: -2px;
}

.subtitle {
    text-align: center;
    font-size: 1.3rem;
    color: #8b92a8;
    font-weight: 500;
    margin-bottom: 3rem;
    letter-spacing: 1px;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 24px;
    background-color: rgba(26, 26, 46, 0.6);
    padding: 12px;
    border-radius: 16px;
    backdrop-filter: blur(10px);
}

.stTabs [data-baseweb="tab"] {
    height: 50px;
    padding: 0 24px;
    background-color: transparent;
    border-radius: 12px;
    color: #8b92a8;
    font-weight: 600;
    font-size: 1rem;
    border: 2px solid transparent;
    transition: all 0.3s ease;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #ff4b4b 0%, #ff6b35 100%);
    color: white !important;
    box-shadow: 0 4px 20px rgba(255, 75, 75, 0.4);
}

/* Card components */
.card {
    background: rgba(26, 26, 46, 0.8);
    border-radius: 20px;
    padding: 2rem;
    margin: 1.5rem 0;
    border: 1px solid rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
}

/* Commentary card */
.commentary-card {
    background: linear-gradient(135deg, rgba(26, 26, 46, 0.9) 0%, rgba(22, 33, 62, 0.9) 100%);
    padding: 1.2rem 1.5rem;
    border-radius: 16px;
    margin-bottom: 1rem;
    border-left: 6px solid;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    transition: all 0.3s ease;
    animation: slideIn 0.4s ease;
}

@keyframes slideIn {
    from {
        opacity: 0;
        transform: translateX(-20px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}

.commentary-card:hover {
    transform: translateX(8px);
    box-shadow: 0 6px 24px rgba(0, 0, 0, 0.3);
}

.commentary-text {
    font-size: 1.1rem;
    font-weight: 600;
    color: #ffffff;
    margin-bottom: 0.5rem;
}

.commentary-meta {
    font-size: 0.85rem;
    color: #8b92a8;
    font-weight: 500;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, rgba(255, 75, 75, 0.15) 0%, rgba(255, 140, 0, 0.15) 100%);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    border: 2px solid rgba(255, 75, 75, 0.3);
    transition: all 0.3s ease;
}

.metric-card:hover {
    transform: scale(1.05);
    border-color: rgba(255, 75, 75, 0.6);
    box-shadow: 0 8px 32px rgba(255, 75, 75, 0.3);
}

.metric-value {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ff4b4b 0%, #ffd700 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.metric-label {
    font-size: 0.9rem;
    color: #8b92a8;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.5rem;
}

/* Button styling */
.stButton > button {
    background: linear-gradient(135deg, #ff4b4b 0%, #ff6b35 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.8rem 2rem;
    font-size: 1.1rem;
    font-weight: 700;
    transition: all 0.3s ease;
    box-shadow: 0 4px 16px rgba(255, 75, 75, 0.3);
    letter-spacing: 0.5px;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 24px rgba(255, 75, 75, 0.5);
    background: linear-gradient(135deg, #ff6b35 0%, #ff4b4b 100%);
}

/* Progress bar */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #ff4b4b 0%, #ffd700 100%);
}

/* File uploader */
.stFileUploader {
    background: rgba(26, 26, 46, 0.6);
    border-radius: 16px;
    padding: 2rem;
    border: 2px dashed rgba(255, 75, 75, 0.3);
    transition: all 0.3s ease;
}

.stFileUploader:hover {
    border-color: rgba(255, 75, 75, 0.6);
    background: rgba(26, 26, 46, 0.8);
}

/* Video player */
video {
    border-radius: 16px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
}

/* Info/Success boxes */
.stSuccess, .stInfo {
    background: rgba(26, 26, 46, 0.8);
    border-radius: 12px;
    border-left: 4px solid #00ff88;
    backdrop-filter: blur(10px);
}

/* Section headers */
h2 {
    color: #ffffff;
    font-weight: 700;
    margin-top: 2rem;
    margin-bottom: 1rem;
    font-size: 2rem;
}

h3 {
    color: #ffd700;
    font-weight: 600;
    font-size: 1.5rem;
}

/* Sidebar */
.css-1d391kg {
    background: rgba(10, 14, 39, 0.95);
}

/* Expander */
.streamlit-expanderHeader {
    background: rgba(26, 26, 46, 0.6);
    border-radius: 12px;
    font-weight: 600;
}

/* Badge/Tag */
.player-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 8px;
    font-size: 0.85rem;
    font-weight: 700;
    margin-right: 8px;
}

.player-1-badge {
    background: rgba(255, 75, 75, 0.2);
    color: #ff4b4b;
    border: 2px solid #ff4b4b;
}

.player-2-badge {
    background: rgba(75, 155, 255, 0.2);
    color: #4b9bff;
    border: 2px solid #4b9bff;
}

.general-badge {
    background: rgba(255, 215, 0, 0.2);
    color: #ffd700;
    border: 2px solid #ffd700;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================
def init_state():
    defaults = {
        "uploaded_video_path": None,
        "pose_data_path": None,
        "moves_data_path": None,
        "commentary_data": None,
        "overlay_video_path": None,
        "processing_stage": "idle",
        "video_metadata": {},
        "single_fighter_mode": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def save_uploaded_file(uploaded_file) -> str:
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    file_path = temp_dir / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(file_path)


def get_video_metadata(path: str) -> Dict:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total / fps if fps > 0 else 0
    cap.release()
    return {"fps": fps, "total_frames": total, "w": w, "h": h, "duration": duration}


def format_time(seconds: float):
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


# ============================================================
# COMMENTARY RENDER
# ============================================================
def render_commentary_line(line: CommentaryLine, index: int):
    if line.player == 1:
        color = "#ff4b4b"
        emoji = "🔴"
        badge_class = "player-1-badge"
        badge_text = "P1"
    elif line.player == 2:
        color = "#4b9bff"
        emoji = "🔵"
        badge_class = "player-2-badge"
        badge_text = "P2"
    else:
        color = "#ffd700"
        emoji = "⚡"
        badge_class = "general-badge"
        badge_text = "GEN"

    st.markdown(f"""
    <div class="commentary-card" style="border-left-color: {color}; animation-delay: {index * 0.05}s;">
        <div class="commentary-text">
            {emoji} {line.text}
        </div>
        <div class="commentary-meta">
            <span class="player-badge {badge_class}">{badge_text}</span>
            {format_time(line.timestamp)} • Confidence: {line.confidence*100:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# METRIC CARD
# ============================================================
def render_metric_card(label: str, value: str, col):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# PIPELINE STAGES
# ============================================================
def run_pose_extraction(video_path: str, pb, status):
    status.text("🔍 Extracting pose data with AI...")

    extractor = PoseExtractor(confidence_threshold=0.5, verbose_logging=True)

    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)

    name = Path(video_path).stem
    out_json = out_dir / f"poses_{name}.json"
    out_overlay = out_dir / f"poses_{name}_overlay.mp4"

    result = extractor.extract_poses_from_video(
        video_path=str(video_path),
        output_json=str(out_json),
        overlay_video=str(out_overlay),
        display=False
    )

    pb.progress(1.0)
    status.text("✅ Pose extraction complete!")

    st.session_state.overlay_video_path = str(out_overlay)
    st.session_state.video_metadata = result["metadata"]

    return str(out_json)


def run_move_classification(pose_json, pb, status):
    status.text("🥋 Classifying MMA moves...")

    classifier = MoveClassifier(
        model_path="models/move_classifier.pkl",
        confidence_threshold=0.55,
        min_move_duration=3,
        move_cooldown=8
    )

    out_dir = Path("artifacts")
    name = Path(pose_json).stem.replace("poses_", "")
    out_json = out_dir / f"moves_{name}.json"

    classifier.classify_from_json(
        pose_json_path=str(pose_json),
        output_path=str(out_json)
    )

    pb.progress(1.0)
    status.text("✅ Move classification complete!")
    return str(out_json)


def run_commentary(moves_json, fps, pb, status):
    status.text("🎙️ Generating AI commentary...")

    engine = CommentaryEngine(fps=fps, enable_tts=False)
    lines = engine.generate_commentary(
        moves_json_path=str(moves_json),
        output_path="artifacts/commentary_output"
    )

    pb.progress(1.0)
    status.text("✅ Commentary generation complete!")

    return lines, getattr(engine, "single_fighter_mode", False)


def run_full_pipeline(video_path: str):
    meta = get_video_metadata(video_path)
    st.session_state.video_metadata = meta

    st.markdown("### 🚀 AI Pipeline Running")

    # Stage 1
    with st.container():
        st.markdown("#### Stage 1: Pose Extraction")
        pb1 = st.progress(0)
        st1 = st.empty()
        pose_json = run_pose_extraction(video_path, pb1, st1)
        st.session_state.pose_data_path = pose_json
        st.session_state.processing_stage = "pose"
        time.sleep(0.5)

    # Stage 2
    with st.container():
        st.markdown("#### Stage 2: Move Classification")
        pb2 = st.progress(0)
        st2 = st.empty()
        moves_json = run_move_classification(pose_json, pb2, st2)
        st.session_state.moves_data_path = moves_json
        st.session_state.processing_stage = "classify"
        time.sleep(0.5)

    # Stage 3
    with st.container():
        st.markdown("#### Stage 3: Commentary Generation")
        pb3 = st.progress(0)
        st3 = st.empty()
        commentary, single_flag = run_commentary(moves_json, meta["fps"], pb3, st3)

        st.session_state.commentary_data = commentary
        st.session_state.single_fighter_mode = single_flag
        st.session_state.processing_stage = "complete"

    st.balloons()
    st.success("🎉 Pipeline Complete! Head to the Results tab.")


# ============================================================
# MAIN UI
# ============================================================
def main():
    init_state()

    # Header
    st.markdown('<h1 class="main-header">⚡ NEUROCOMBAT</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-Powered MMA Fight Analysis & Commentary</p>', unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "📤 Upload & Process",
        "🎬 Results & Playback",
        "📊 Fight Statistics"
    ])

    # ==================== TAB 1: UPLOAD ====================
    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("## 📹 Upload Your Fight Video")
        
        uploaded = st.file_uploader(
            "Choose an MMA fight video",
            type=["mp4", "mov", "avi", "mkv"],
            help="Upload a video file containing MMA fight footage"
        )

        if uploaded:
            if st.session_state.uploaded_video_path is None:
                with st.spinner("Saving video..."):
                    path = save_uploaded_file(uploaded)
                    st.session_state.uploaded_video_path = path
                    st.session_state.video_metadata = get_video_metadata(path)

            meta = st.session_state.video_metadata

            st.markdown("### 📊 Video Information")
            
            col1, col2, col3, col4 = st.columns(4)
            
            render_metric_card("FPS", f"{meta.get('fps', 0):.0f}", col1)
            render_metric_card("Frames", str(meta.get('total_frames', 'N/A')), col2)
            render_metric_card("Duration", format_time(meta.get('duration', 0)), col3)
            render_metric_card("Resolution", f"{meta.get('w', 0)}x{meta.get('h', 0)}", col4)

            st.markdown("---")

            if st.session_state.processing_stage == "idle":
                if st.button("🚀 Start AI Analysis", use_container_width=True, type="primary"):
                    run_full_pipeline(st.session_state.uploaded_video_path)
                    st.rerun()
            else:
                st.success("✅ Video already processed! Check the Results tab.")
                
                if st.button("🔄 Process New Video", use_container_width=True):
                    for k in list(st.session_state.keys()):
                        del st.session_state[k]
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

    # ==================== TAB 2: RESULTS ====================
    with tab2:
        if st.session_state.processing_stage != "complete":
            st.info("📤 Please upload and process a video in the 'Upload & Process' tab first.")
        else:
            # Video Section
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("## 🎬 Pose Detection Overlay")

            overlay_path = st.session_state.overlay_video_path
            
            if overlay_path and Path(overlay_path).exists():
                file_size = Path(overlay_path).stat().st_size
                
                if file_size < 1000:
                    st.error(f"❌ Video file error ({file_size} bytes). Please reprocess.")
                else:
                    st.video(overlay_path)
                    
                    col1, col2 = st.columns([3, 1])
                    with col2:
                        with open(overlay_path, "rb") as f:
                            st.download_button(
                                label="⬇️ Download Video",
                                data=f,
                                file_name=f"neurocombat_overlay.mp4",
                                mime="video/mp4",
                                use_container_width=True
                            )
            else:
                st.error("❌ Overlay video not found. Please reprocess.")
            
            st.markdown('</div>', unsafe_allow_html=True)

            # Commentary Section
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("## 🎙️ AI Commentary Feed")

            if st.session_state.commentary_data:
                lines = st.session_state.commentary_data
                
                # Filter options
                col1, col2 = st.columns([2, 1])
                with col1:
                    filter_option = st.selectbox(
                        "Filter by:",
                        ["All Commentary", "Player 1 Only", "Player 2 Only", "General Comments"]
                    )
                
                # Apply filter
                if filter_option == "Player 1 Only":
                    filtered_lines = [l for l in lines if l.player == 1]
                elif filter_option == "Player 2 Only":
                    filtered_lines = [l for l in lines if l.player == 2]
                elif filter_option == "General Comments":
                    filtered_lines = [l for l in lines if l.player is None]
                else:
                    filtered_lines = lines

                st.markdown(f"**Showing {len(filtered_lines)} of {len(lines)} commentary lines**")
                st.markdown("---")

                for idx, line in enumerate(filtered_lines):
                    render_commentary_line(line, idx)
            else:
                st.info("No commentary generated.")
            
            st.markdown('</div>', unsafe_allow_html=True)

    # ==================== TAB 3: STATS ====================
    with tab3:
        if st.session_state.processing_stage != "complete":
            st.info("📊 Process a video to see detailed statistics.")
        else:
            lines = st.session_state.commentary_data

            if lines:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("## 📈 Fight Analytics Dashboard")

                # Summary metrics
                total = len(lines)
                p1 = sum(1 for l in lines if l.player == 1)
                p2 = sum(1 for l in lines if l.player == 2)
                gen = sum(1 for l in lines if l.player is None)

                col1, col2, col3, col4 = st.columns(4)
                render_metric_card("Total Events", str(total), col1)
                render_metric_card("Player 1", str(p1), col2)
                render_metric_card("Player 2", str(p2), col3)
                render_metric_card("General", str(gen), col4)

                st.markdown("---")

                # Move type distribution
                st.markdown("### 🥊 Move Distribution")
                
                move_counts = {}
                for line in lines:
                    if line.move_type and line.move_type != 'neutral':
                        move_counts[line.move_type] = move_counts.get(line.move_type, 0) + 1

                if move_counts:
                    fig = px.bar(
                        x=list(move_counts.keys()),
                        y=list(move_counts.values()),
                        labels={'x': 'Move Type', 'y': 'Count'},
                        color=list(move_counts.values()),
                        color_continuous_scale='Reds'
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown("---")

                # Timeline
                st.markdown("### ⏱️ Action Timeline")
                
                timeline_data = []
                for line in lines:
                    if line.player is not None:
                        timeline_data.append({
                            'time': line.timestamp,
                            'player': f'Player {line.player}',
                            'move': line.move_type or 'action'
                        })

                if timeline_data:
                    fig = px.scatter(
                        timeline_data,
                        x='time',
                        y='player',
                        color='player',
                        labels={'time': 'Time (seconds)', 'player': ''},
                        color_discrete_map={'Player 1': '#ff4b4b', 'Player 2': '#4b9bff'}
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown('</div>', unsafe_allow_html=True)

                # Raw data
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### 📋 Raw Commentary Data")
                
                with st.expander("View JSON output"):
                    st.json([asdict(l) for l in lines])
                
                st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()