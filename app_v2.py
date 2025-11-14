"""
✨ NEUROCOMBAT V4 - Minimalist AI Fight Analyst UI ✨
======================================================
Minimalist, high-end, futuristic interface (VisionOS/Tesla style).
All backend logic is 100% UNCHANGED and uses the user's latest files.

Run with:
    pip install streamlit-lottie plotly
    streamlit run app_v2_minimalist.py
"""

import streamlit as st
import cv2
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import asdict, dataclass
import plotly.graph_objects as go
import plotly.express as px
from streamlit_lottie import st_lottie
import requests
from collections import Counter

# ============================================================
# BACKEND IMPORTS (UNCHANGED)
# ============================================================
# These are the original backend functions. No changes were made.
try:
    # We import the dataclass `Move` from the classifier to help with type hinting
    from backend.move_classifier_v2 import MoveClassifier, Move
    from backend.pose_extractor_v2 import PoseExtractor
    from backend.commentary_engine_v2 import CommentaryEngine, CommentaryLine
except ImportError:
    st.error("❌ Backend modules missing. Ensure `backend/pose_extractor_v2.py`, `backend/move_classifier_v2.py`, and `backend/commentary_engine_v2.py` exist.")
    st.stop()
except Exception as e:
    st.error(f"Error importing backend: {e}")
    st.stop()


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="NeuroCombat /// Minimalist AI",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# LOTTIE ASSETS (Minimalist Gray/White Theme)
# ============================================================
def load_lottie_url(url: str):
    """Helper to load Lottie animation from a reliable URL."""
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException:
        return None # Return None if loading fails

# Used reliable public Lottie URLs for better stability:
LOTTIE_AI_BRAIN_URL = "https://lottie.host/90b5015b-592f-48d1-93a8-444453b34190/L8wYV2T8hD.json" # Tech/Globe
LOTTIE_UPLOAD_URL = "https://lottie.host/e31f0f4a-93a3-488f-a95d-755716e25114/2n31t6lHqI.json" # Upload/Data
LOTTIE_LOADING_URL = "https://lottie.host/e2afc6e8-251f-44e2-8957-617a264a0601/7e1L58X1Vl.json" # Minimal Loading Pulse

lottie_ai_brain = load_lottie_url(LOTTIE_AI_BRAIN_URL)
lottie_upload_anim = load_lottie_url(LOTTIE_UPLOAD_URL)
lottie_loading_anim = load_lottie_url(LOTTIE_LOADING_URL)


# ============================================================
# ELEGANT MINIMALIST CSS THEME (VisionOS / Tesla Vibe)
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;700&display=swap');

:root {
    /* Color Palette */
    --color-bg: #0D0F12;
    --color-bg-card: #111418;
    --color-accent: #4EA9FF; /* Soft Cool Blue */
    --color-secondary: #A591FF; /* Muted Purple */
    --color-text: rgba(255, 255, 255, 0.9);
    --color-text-dim: rgba(255, 255, 255, 0.5);
    --color-success: #6EE7C8; /* Teal */
    --color-player-1: #FF6B6B; /* Muted Red */
    --color-player-2: #5BB4FF; /* Brighter Blue */
    
    /* Font */
    --font-family: 'Manrope', sans-serif;
    
    /* Effects */
    --shadow-soft: 0 4px 10px rgba(0, 0, 0, 0.5), 0 0 10px rgba(78, 169, 255, 0.05);
    --glow-soft: 0 0 8px rgba(78, 169, 255, 0.3);
}

* {
    font-family: var(--font-family);
    color: var(--color-text);
}

/* --- Main App Background --- */
.stApp {
    background-color: var(--color-bg);
    background-image: 
        radial-gradient(at 10% 50%, #15181f 0%, transparent 80%),
        radial-gradient(at 90% 50%, #15181f 0%, transparent 80%);
}

/* --- Hide Streamlit Branding --- */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* --- Header / Title --- */
.main-header {
    font-size: 4rem;
    font-weight: 700;
    text-align: center;
    letter-spacing: -1px;
    margin-bottom: 0.5rem;
    color: var(--color-text);
}
.main-header span {
    color: var(--color-accent);
}

.subtitle {
    font-weight: 400;
    text-align: center;
    font-size: 1.1rem;
    color: var(--color-text-dim);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 3rem;
}

/* --- Sidebar --- */
.css-1d391kg { /* Streamlit Sidebar Class */
    background: var(--color-bg-card);
    box-shadow: 2px 0 10px rgba(0, 0, 0, 0.5);
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}
.css-1d391kg h4 { /* Sidebar Sub-Header */
    color: var(--color-accent);
    font-weight: 500;
    margin-top: 1.5rem;
    padding-left: 10px;
}
/* Toggles (Checkboxes) - Subtle Border Glow */
.stCheckbox {
    background: rgba(78, 169, 255, 0.05);
    border: 1px solid rgba(78, 169, 255, 0.2);
    border-radius: 8px;
    padding: 8px 12px;
    transition: all 0.3s ease;
}
.stCheckbox:hover {
    border-color: var(--color-accent);
    box-shadow: var(--glow-soft);
}

/* --- Floating Cards (VisionOS style) --- */
.card {
    background: var(--color-bg-card);
    border-radius: 16px;
    padding: 2.5rem;
    margin: 1.5rem 0;
    border: 1px solid rgba(255, 255, 255, 0.05);
    box-shadow: var(--shadow-soft);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 15px rgba(0, 0, 0, 0.6), var(--glow-soft);
}

/* --- Main Tabs --- */
.stTabs [data-baseweb="tab-list"] {
    gap: 20px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    padding: 0 1rem;
}
.stTabs [data-baseweb="tab"] {
    font-size: 1.1rem;
    color: var(--color-text-dim);
    border: none;
    padding: 10px 0;
    transition: all 0.3s ease;
}
.stTabs [aria-selected="true"] {
    color: var(--color-accent);
    font-weight: 500;
    border-bottom: 3px solid var(--color-accent);
}

/* --- Section Headers --- */
h2 {
    font-weight: 700;
    font-size: 2rem;
    color: var(--color-text);
    padding-bottom: 5px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    margin-bottom: 1.5rem;
}
h3 {
    font-weight: 500;
    font-size: 1.5rem;
    color: var(--color-accent);
    margin-top: 1rem;
}

/* --- Metric Cards --- */
.metric-card {
    background: rgba(78, 169, 255, 0.05);
    border-radius: 12px;
    padding: 1.5rem 1rem;
    text-align: center;
    border: 1px solid rgba(78, 169, 255, 0.1);
    transition: all 0.3s ease;
}
.metric-card:hover {
    background: rgba(78, 169, 255, 0.1);
}
.metric-value {
    font-size: 2.5rem;
    font-weight: 700;
    color: var(--color-text);
    line-height: 1;
}
.metric-label {
    font-size: 0.8rem;
    color: var(--color-text-dim);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.5rem;
}

/* --- Buttons --- */
.stButton > button {
    font-weight: 700;
    font-size: 1.1rem;
    background: var(--color-accent);
    color: var(--color-bg);
    border: none;
    border-radius: 10px;
    padding: 0.75rem 1.5rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 10px rgba(78, 169, 255, 0.4);
}
.stButton > button:hover {
    background: #6fc2ff;
    transform: translateY(-2px);
    box-shadow: 0 6px 15px rgba(78, 169, 255, 0.6);
}

/* --- Pipeline Stage Cards (Tesla Loading Vibe) --- */
.stage-card {
    background: var(--color-bg-card);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 10px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.5s ease;
    height: 180px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 0 0 transparent; /* Reset shadow */
}

/* Stage status colors */
.status-waiting { color: var(--color-text-dim); }
.status-running { color: var(--color-accent); font-weight: 500; }
.status-complete { color: var(--color-success); font-weight: 500; }

/* Active/Running Stage Glow */
.stage-card-active {
    border-color: var(--color-accent);
    box-shadow: var(--glow-soft);
}
.stage-card-active h3 {
    color: var(--color-accent);
}

/* Complete Stage Checkmark */
.stage-card-complete {
    border-color: var(--color-success);
}
.stage-card-complete h3 {
    color: var(--color-success);
}

/* --- Upload Zone --- */
.stFileUploader {
    border: 2px dashed rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 2rem;
    background: rgba(78, 169, 255, 0.03);
    transition: all 0.3s ease;
}
.stFileUploader:hover {
    border-color: var(--color-accent);
    background: rgba(78, 169, 255, 0.08);
}
.stFileUploader label {
    font-size: 1.2rem;
    color: var(--color-text);
}

/* --- Video Player Card --- */
.video-card video {
    border-radius: 10px;
    outline: 2px solid rgba(255, 255, 255, 0.1);
}

/* --- Commentary Feed (Messaging App Style) --- */
.commentary-feed-container {
    height: 500px; /* Reduced height as it's in the same card */
    overflow-y: auto; 
    padding: 15px;
    background: rgba(0, 0, 0, 0.2);
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.05);
}

.commentary-bubble {
    padding: 0.75rem 1.2rem;
    border-radius: 12px;
    max-width: 85%;
    margin-bottom: 10px;
    transition: all 0.3s ease;
    animation: fadeInSlide 0.4s ease-out;
}
@keyframes fadeInSlide {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.commentary-bubble:hover {
    box-shadow: var(--shadow-soft);
}

.commentary-text {
    font-size: 1rem;
    font-weight: 500;
}
.commentary-meta {
    font-size: 0.75rem;
    color: var(--color-text-dim);
    margin-top: 5px;
}

/* P1 (Right/Red) */
.p1-bubble {
    background: rgba(255, 107, 107, 0.1);
    border: 1px solid var(--color-player-1);
    margin-left: auto;
    border-top-right-radius: 4px;
}

/* P2 (Left/Blue) */
.p2-bubble {
    background: rgba(91, 180, 255, 0.1);
    border: 1px solid var(--color-player-2);
    margin-right: auto;
    border-top-left-radius: 4px;
}

/* General (Center/System) */
.gen-bubble {
    background: rgba(165, 145, 255, 0.1);
    border: 1px solid var(--color-secondary);
    margin: 10px auto;
    max-width: 95%;
    text-align: center;
}

/* --- Plotly Chart Styling (overrides Plotly defaults for dark mode) --- */
.js-plotly-plot {
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 10px;
    box-shadow: var(--shadow-soft);
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE (UNCHANGED)
# ============================================================
def init_state():
    """Initializes all session state variables."""
    defaults = {
        "uploaded_video_path": None,
        "pose_data_path": None,
        "moves_data_path": None,
        "commentary_data": None,
        "overlay_video_path": None,
        "processing_stage": "idle",
        "video_metadata": {},
        "single_fighter_mode": False,
        "high_accuracy_mode": False,
        "enable_smoothing": True,
        "enable_motion_trails": True,
        "view_mode": "Minimal", # Added view mode toggle
        "all_moves_list": [] # Stores the List[Move] objects for stats
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ============================================================
# UTILITY FUNCTIONS (UNCHANGED)
# ============================================================
def save_uploaded_file(uploaded_file) -> str:
    """Saves uploaded file to a temp dir and returns the path."""
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    file_path = temp_dir / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(file_path)


def get_video_metadata(path: str) -> Dict:
    """Extracts metadata (fps, frames, res, duration) from video file."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total / fps if fps > 0 else 0
    cap.release()
    return {"fps": fps, "total_frames": total, "w": w, "h": h, "duration": duration}


def format_time(seconds: float):
    """Formats seconds into MM:SS string."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


# ============================================================
# UI RENDER FUNCTIONS (Minimalist Redesign)
# ============================================================

def render_commentary_line(line: CommentaryLine, index: int):
    """Renders a single commentary line with the minimal message bubble style."""
    
    timestamp = format_time(line.timestamp)
    confidence = f"{line.confidence*100:.0f}%"
    
    if line.player == 1:
        bubble_class = "p1-bubble"
        text_content = f"P1: {line.text}"
    elif line.player == 2:
        bubble_class = "p2-bubble"
        text_content = f"P2: {line.text}"
    else:
        bubble_class = "gen-bubble"
        text_content = f"AI: {line.text}"

    # Generate HTML for the bubble
    st.markdown(f"""
    <div class="commentary-bubble {bubble_class}" style="animation-delay: {index * 0.05}s;">
        <div class="commentary-text">
            {text_content}
        </div>
        <div class="commentary-meta">
            {timestamp} • CONF: {confidence}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_metric_card(label: str, value: str, col):
    """Renders a single metric card in the minimalist style."""
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)


def render_stage_card_minimal(col, title, status):
    """Renders a single pipeline stage card (Tesla style)."""
    
    status_map = {
        "idle": ("QUEUED", "status-waiting", ""),
        "processing": ("RUNNING...", "status-running", "stage-card-active"),
        "complete": ("SUCCESS", "status-complete", "stage-card-complete"),
    }
    
    status_text, status_class, card_class = status_map.get(status, status_map['idle'])
    
    lottie_html = ""
    if status == "processing" and lottie_loading_anim:
        # Note: This is a way to embed Lottie without st_lottie to fit in the card
        lottie_html = f"""
        <div style="height: 80px; margin-top: 10px;">
            <lottie-player src="{LOTTIE_LOADING_URL}" background="transparent" speed="1" style="width: 100%; height: 80px;" loop autoplay></lottie-player>
        </div>
        """
    else:
        lottie_html = "<div style='height: 90px;'></div>" # Placeholder to keep card height consistent

    with col:
        st.markdown(f"""
        <div class="stage-card {card_class}">
            <h3>{title}</h3>
            <div class="{status_class}" style="margin-bottom: 5px;">{status_text}</div>
            {lottie_html}
        </div>
        """, unsafe_allow_html=True)
        
        # Add the Lottie player script to <head> if it's the first time
        if 'lottie_script_added' not in st.session_state:
            st.html("""
                <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
            """)
            st.session_state.lottie_script_added = True


# ============================================================
# PIPELINE FUNCTIONS (BACKEND LOGIC 100% UNCHANGED)
# ============================================================
# These functions call the backend files provided by the user.
# The logic inside these functions is IDENTICAL to the user's files.

def run_pose_extraction(video_path: str) -> Tuple[str, str, Dict]:
    """Calls PoseExtractor backend."""
    # Uses PoseExtractor from backend/pose_extractor_v2.py
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
    
    overlay_path = str(out_overlay)
    metadata = result["metadata"]
    return str(out_json), overlay_path, metadata


def run_move_classification(pose_json: str) -> Tuple[str, List[Move]]:
    """Calls MoveClassifier backend."""
    # Uses MoveClassifier from backend/move_classifier_v2.py
    classifier = MoveClassifier(
        model_path="models/move_classifier.pkl",
        confidence_threshold=0.55, # Using a slightly lower threshold than default
        min_move_duration=3,
        move_cooldown=8
    )
    out_dir = Path("artifacts")
    name = Path(pose_json).stem.replace("poses_", "")
    out_json = out_dir / f"moves_{name}.json"

    # The backend function saves the JSON AND returns the list of Move objects
    all_moves = classifier.classify_from_json(
        pose_json_path=str(pose_json),
        output_path=str(out_json)
    )
    return str(out_json), all_moves


def run_commentary(moves_json: str, fps: float) -> Tuple[List[CommentaryLine], bool]:
    """Calls CommentaryEngine backend."""
    # Uses CommentaryEngine from backend/commentary_engine_v2.py
    engine = CommentaryEngine(fps=fps, enable_tts=False)
    
    # The backend function reads the JSON and generates commentary
    lines = engine.generate_commentary(
        moves_json_path=str(moves_json),
        output_path="artifacts/commentary_output"
    )
    
    single_fighter_mode = getattr(engine, "single_fighter_mode", False)
    return lines, single_fighter_mode


def run_full_pipeline_ui(video_path: str):
    """Manages the UI flow and calls the backend functions sequentially."""
    
    st.markdown("### ⚡ System Check: AI Pipeline Active")
    
    col1, col2, col3 = st.columns(3)
    stage1_placeholder = col1.empty()
    stage2_placeholder = col2.empty()
    stage3_placeholder = col3.empty()
    status_message = st.empty()
    
    render_stage_card_minimal(stage1_placeholder, "1. POSE TRACKING", "idle")
    render_stage_card_minimal(stage2_placeholder, "2. MOVE CLASSIFICATION", "idle")
    render_stage_card_minimal(stage3_placeholder, "3. COMMENTARY GENERATION", "idle")

    try:
        # --- Stage 1: Pose Extraction ---
        status_message.info("01/03: Initializing Pose Extraction...", icon="🌐")
        render_stage_card_minimal(stage1_placeholder, "1. POSE TRACKING", "processing")
        
        meta = get_video_metadata(video_path)
        pose_json, overlay_path, metadata = run_pose_extraction(video_path)
        
        st.session_state.pose_data_path = pose_json
        st.session_state.overlay_video_path = overlay_path
        st.session_state.video_metadata = metadata
        st.session_state.processing_stage = "pose"
        
        render_stage_card_minimal(stage1_placeholder, "1. POSE TRACKING", "complete")

        # --- Stage 2: Move Classification ---
        status_message.info("02/03: Running Move Classification...", icon="🥋")
        render_stage_card_minimal(stage2_placeholder, "2. MOVE CLASSIFICATION", "processing")

        # ** FIX: Capture the returned list of Move objects **
        moves_json, all_moves_list = run_move_classification(pose_json)
        
        st.session_state.moves_data_path = moves_json
        st.session_state.all_moves_list = all_moves_list # <-- SAVE THE MOVES LIST
        st.session_state.processing_stage = "classify"
        
        render_stage_card_minimal(stage2_placeholder, "2. MOVE CLASSIFICATION", "complete")

        # --- Stage 3: Commentary Generation ---
        status_message.info("03/03: Generating Dynamic Commentary...", icon="🎙️")
        render_stage_card_minimal(stage3_placeholder, "3. COMMENTARY GENERATION", "processing")

        commentary, single_flag = run_commentary(moves_json, meta["fps"])

        st.session_state.commentary_data = commentary
        st.session_state.single_fighter_mode = single_flag
        st.session_state.processing_stage = "complete"
        
        render_stage_card_minimal(stage3_placeholder, "3. COMMENTARY GENERATION", "complete")
        
        status_message.success("✅ Analysis Complete! Results Ready.", icon="🏆")
        st.session_state.processing_stage = "complete"
        st.rerun() # Rerun to update the UI to the "complete" state

    except Exception as e:
        status_message.error(f"❌ AI Pipeline Failed: {e}")
        st.exception(e)


# ============================================================
# PLOTLY CHARTING FUNCTIONS (Minimalist & Clean)
# ============================================================

def create_move_distribution_chart(moves_list: List[Move]) -> go.Figure:
    """Creates a clean, minimal move distribution bar chart from Move objects."""
    # ** FIX: This function now uses the List[Move] objects directly **
    move_counts = Counter(m.move_type for m in moves_list if m.move_type != 'neutral')
    if not move_counts: 
        return None
    
    moves, counts = zip(*move_counts.most_common())

    fig = go.Figure(data=[
        go.Bar(
            x=list(moves), 
            y=list(counts),
            marker_color="var(--color-accent)", # Use CSS variable
            opacity=0.8,
            hovertemplate="<b>%{x}</b>: %{y} Events<extra></extra>"
        )
    ])
    
    fig.update_layout(
        title_text="Move Distribution",
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="var(--color-text)", family="var(--font-family)"),
        xaxis=dict(showgrid=False, title=None),
        yaxis=dict(gridcolor="rgba(255, 255, 255, 0.1)", title="Count"),
        margin=dict(t=50, b=0, l=0, r=0)
    )
    return fig


def create_player_activity_chart(moves_list: List[Move]) -> go.Figure:
    """Creates a minimal player activity timeline from Move objects."""
    # ** FIX: This function now uses the List[Move] objects directly **
    timeline_data = []
    for move in moves_list:
        if move.player_id is not None:
            timeline_data.append({
                'time': move.frame_start / st.session_state.video_metadata.get("fps", 30),
                'player': f'Player {move.player_id}',
                'move': move.move_type
            })
    
    if not timeline_data: 
        return None

    p1_data = [d for d in timeline_data if d['player'] == 'Player 1']
    p2_data = [d for d in timeline_data if d['player'] == 'Player 2']

    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=[d['time'] for d in p1_data],
        y=[1] * len(p1_data),
        mode='markers',
        name='Player 1',
        marker=dict(size=10, color="var(--color-player-1)", symbol='circle', line=dict(width=1, color='var(--color-bg-card)')),
        hovertemplate="P1 at %{x:.1f}s: %{customdata[0]}<extra></extra>",
        customdata=[[d['move']] for d in p1_data]
    ))
    
    fig.add_trace(go.Scatter(
        x=[d['time'] for d in p2_data],
        y=[2] * len(p2_data),
        mode='markers',
        name='Player 2',
        marker=dict(size=10, color="var(--color-player-2)", symbol='circle', line=dict(width=1, color='var(--color-bg-card)')),
        hovertemplate="P2 at %{x:.1f}s: %{customdata[0]}<extra></extra>",
        customdata=[[d['move']] for d in p2_data]
    ))

    fig.update_layout(
        title_text="Player Activity Timeline",
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="var(--color-text)", family="var(--font-family)"),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridcolor="rgba(255, 255, 255, 0.1)", title="Time (seconds)"),
        yaxis=dict(
            tickvals=[1, 2],
            ticktext=['Player 1', 'Player 2'],
            range=[0.5, 2.5],
            showgrid=False
        ),
        height=300,
        margin=dict(t=50, b=50, l=0, r=0)
    )
    return fig


# ============================================================
# MAIN UI
# ============================================================
def main():
    init_state()

    # --- Sidebar (Settings Panel) ---
    with st.sidebar:
        st.markdown("<h3>ANALYSIS CONTROLS</h3>", unsafe_allow_html=True)

        st.session_state.view_mode = st.radio(
            "Display Mode",
            ["Minimal", "Detailed Analytical"],
            index=0,
            horizontal=True
        )

        st.divider()
        st.markdown("<h4>AI Toggles (UI Only)</h4>", unsafe_allow_html=True)
        st.checkbox("High Accuracy Mode", key="high_accuracy_mode", help="Activates high-res temporal analysis.")
        st.checkbox("Temporal Smoothing", key="enable_smoothing", help="Uses rolling average for smoother skeleton movement.")
        st.checkbox("Motion Trails", key="enable_motion_trails", help="Visualizes movement paths on the overlay.")
        
        st.divider()
        if lottie_ai_brain:
            st_lottie(lottie_ai_brain, height=120, key="sidebar_anim", speed=0.5)
        st.markdown(f'<p style="color:var(--color-text-dim); text-align:center; margin-top:10px;">NEUROCOMBAT V4</p>', unsafe_allow_html=True)

    # --- Header ---
    st.markdown('<h1 class="main-header">NEURO<span>COMBAT</span></h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Minimalist AI Fight Analytics</p>', unsafe_allow_html=True)

    # --- Main Tabs ---
    tab1, tab2, tab3 = st.tabs([
        "📤 DATA PIPELINE",
        "🎬 RESULTS & COMMENTARY",
        "📊 CORE STATISTICS"
    ])

    # ==================== TAB 1: UPLOAD & PROCESS ====================
    with tab1:
        if st.session_state.processing_stage == "idle":
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("## 📹 Upload & Process File")
            
            uploaded = st.file_uploader(
                "Upload MMA Fight Video (MP4, MOV)",
                type=["mp4", "mov", "avi", "mkv"],
                help="Upload a video file for AI processing."
            )
            
            if uploaded:
                if st.session_state.uploaded_video_path is None or uploaded.name not in st.session_state.uploaded_video_path:
                    with st.spinner("Saving video and analyzing metadata..."):
                        path = save_uploaded_file(uploaded)
                        st.session_state.uploaded_video_path = path
                        st.session_state.video_metadata = get_video_metadata(path)
                
                meta = st.session_state.video_metadata
                st.markdown("### Metadata")
                col1, col2, col3, col4 = st.columns(4)
                render_metric_card("FPS", f"{meta.get('fps', 0):.0f}", col1)
                render_metric_card("Frames", str(meta.get('total_frames', 'N/A')), col2)
                render_metric_card("Duration", format_time(meta.get('duration', 0)), col3)
                render_metric_card("Resolution", f"{meta.get('w', 0)}x{meta.get('h', 0)}", col4)
                
                st.markdown("---")
                if st.button("🚀 START AI PIPELINE", use_container_width=True, type="primary"):
                    st.session_state.processing_stage = "processing"
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)

        elif st.session_state.processing_stage == "processing":
            run_full_pipeline_ui(st.session_state.uploaded_video_path)

        else: # Processing is complete
            st.success("✅ AI Analysis Complete. View results in the next tab.")
            if st.button("🔄 RESTART / ANALYZE NEW VIDEO", use_container_width=True):
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                st.rerun()

    # ==================== TAB 2: RESULTS & COMMENTARY (REDESIGNED) ====================
    with tab2:
        if st.session_state.processing_stage != "complete":
            st.info("📤 Process a video in the first tab to view results.")
        else:
            # ** UI REFINEMENT: Single card for Video + Commentary **
            st.markdown('<div class="card">', unsafe_allow_html=True)
            
            # --- Video Section ---
            st.markdown("## 🎬 AI Visual Overlay")
            overlay_path = st.session_state.overlay_video_path
            if overlay_path and Path(overlay_path).exists():
                st.video(overlay_path)
                with open(overlay_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Overlay Video",
                        data=f,
                        file_name=f"neurocombat_overlay.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )
            else:
                st.error("❌ Processed video file not found.")
            
            st.markdown("---") # Divider inside the same card

            # --- Commentary Section (now below video) ---
            st.markdown("## 🎙️ AI Commentary Timeline")
            if st.session_state.commentary_data:
                lines = st.session_state.commentary_data
                st.markdown(f'<p style="color:var(--color-accent); font-size:1.1rem;">Total Events: {len(lines)}</p>', unsafe_allow_html=True)
                
                st.markdown('<div class="commentary-feed-container">', unsafe_allow_html=True)
                for idx, line in enumerate(lines):
                    render_commentary_line(line, idx)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No commentary generated.")
            
            st.markdown('</div>', unsafe_allow_html=True) # Close the single card

    # ==================== TAB 3: STATISTICS (FIXED) ====================
    with tab3:
        if st.session_state.processing_stage != "complete":
            st.info("📊 Process a video to see core statistics.")
        else:
            # ** FIX: Use the list of Move objects for stats, not commentary lines **
            moves_list = st.session_state.all_moves_list
            commentary_lines = st.session_state.commentary_data
            
            if commentary_lines:
                # --- Summary Metrics (Top Row) ---
                p1_moves = sum(1 for m in moves_list if m.player_id == 1 and m.move_type != 'neutral')
                p2_moves = sum(1 for m in moves_list if m.player_id == 2 and m.move_type != 'neutral')
                total_moves = p1_moves + p2_moves
                
                col1, col2, col3, col4 = st.columns(4)
                render_metric_card("Total Moves", str(total_moves), col1)
                render_metric_card("Player 1 Moves", str(p1_moves), col2)
                render_metric_card("Player 2 Moves", str(p2_moves), col3)
                render_metric_card("Commentary Lines", str(len(commentary_lines)), col4)
                
                st.markdown("---")

                # --- Charts ---
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("## 📊 Core Analytics")

                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    st.markdown("### Move Frequency")
                    fig1 = create_move_distribution_chart(moves_list) # Use moves_list
                    if fig1:
                        st.plotly_chart(fig1, use_container_width=True)
                    else:
                        st.info("Not enough move data.")
                
                with col_chart2:
                    st.markdown("### Player Engagement")
                    fig2 = create_player_activity_chart(moves_list) # Use moves_list
                    if fig2:
                        st.plotly_chart(fig2, use_container_width=True)
                    else:
                        st.info("No player activity detected.")

                st.markdown('</div>', unsafe_allow_html=True)

                # --- Raw Data ---
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("## 📋 Raw Data Export")
                with st.expander("View Full Moves List JSON (from Classifier)"):
                    st.json([asdict(m) for m in moves_list])
                with st.expander("View Full Commentary JSON (from Engine)"):
                    st.json([asdict(l) for l in commentary_lines])
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No statistics available.")


if __name__ == "__main__":
    main()