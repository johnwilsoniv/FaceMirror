"""
S2.5 Frame Curator — configuration.

Sits between S2 (action coding) and S3 (analysis). The researcher curates which
frames within each (patient, action) are genuine, representative task performance,
producing clean input for the characteristic-window aggregator.

Paths point at the main checkout's data (the worktree's gitignored data dirs are
not populated; the real per-frame CSVs + videos live in the main tree).
"""
from pathlib import Path

# --- Data locations (absolute, main checkout) ---
S3_DATA = Path("/Users/johnwilsoniv/Documents/SplitFace Open3/S3 Data Analysis")
PER_FRAME_DIR = S3_DATA / "recoded_rerun_dual_v1316"
AUTO_WINDOW_CSV = S3_DATA / "pilot15_window_frame_indices.csv"
INSET_CSV = S3_DATA / "pilot15_inset_aus.csv"
CONTROL_DIR = Path("/Users/johnwilsoniv/Documents/SplitFace Open3/"
                   "S Data/Normal Cohort")
SOURCE_VIDEO_DIR = Path("/Users/johnwilsoniv/Documents/SplitFace/S2O Coded Files")
MIRRORED_VIDEO_DIR = Path("/Users/johnwilsoniv/Documents/SplitFace/"
                          "S1O Processed Files/Face Mirror 1.0 Output")

# --- Curation state output ---
# Follows the SplitFace stage-output convention (S1O Processed / S2O Coded /
# S25 Curated). Working state (JSON) + analysis exports (csv/xlsx) all live here.
S25_DIR = Path("/Users/johnwilsoniv/Documents/SplitFace/S25 Curated Files")
CURATION_JSON = S25_DIR / "s25_curation.json"
CURATED_CSV = S25_DIR / "s25_curated_frames.csv"
CURATED_XLSX = S25_DIR / "s25_curated_frames.xlsx"   # combined long-format export

# --- S2 re-score handoff (launch S2 from source on a Python with its deps) ---
S2_DIR = Path("/Users/johnwilsoniv/Documents/SplitFace Open3/S2 Action Coder")
S2_PYTHON = "/Library/Frameworks/Python.framework/Versions/3.10/bin/python3.10"
# ^ Python 3.10 is S2's BUILD-TARGET version (the app bundle is cpython-310): it
# has the full stack incl. tkinter (Homebrew 3.13/3.14 lack _tkinter -> splash
# crash). Verified S2 launches from source on it.
# S2's input = the {pid}_source.MOV in S1O 'Combined Data' (paired with the
# left/right _mirrored.csv AU files there). NOT the mirrored mp4s.
S1_COMBINED_DIR = Path("/Users/johnwilsoniv/Documents/SplitFace/"
                       "S1O Processed Files/Combined Data")

# --- Auto-curator: validated per-action selection rules (s25_auto_curator.py) ---
AUTO_PARAMS_JSON = S3_DATA / "s25_auto_params.json"
# Authoritative FACS task-AU set per action used by the auto-curator rules.
# The 8 analyzed actions reuse the validated in-set; the 6 others get their
# defining AU where it exists in the 17-AU panel (PL/BC/LT have none -> []).
FACS_TASK_AUS = {
    'BS': ['AU06', 'AU07', 'AU10', 'AU12', 'AU14', 'AU23', 'AU25'],
    'SS': ['AU06', 'AU07', 'AU10', 'AU12', 'AU14', 'AU23', 'AU25', 'AU45'],
    'RE': ['AU01', 'AU02', 'AU05'],
    'ES': ['AU07', 'AU45'],
    'ET': ['AU04', 'AU06', 'AU07', 'AU09', 'AU10', 'AU14', 'AU23', 'AU26', 'AU45'],
    'SE': ['AU06', 'AU07', 'AU10', 'AU12', 'AU14', 'AU25', 'AU26'],
    'SO': ['AU01', 'AU04', 'AU06', 'AU07', 'AU10', 'AU12', 'AU14', 'AU25', 'AU26'],
    'WN': ['AU09'], 'FR': ['AU04'], 'BK': ['AU45'],
    'PL': [], 'BC': [], 'LT': [],
}
EYE_TASKS = {'ES', 'ET', 'BK'}            # high AU45 is the TARGET, not a blink
NO_PANEL_ACTIONS = {'PL', 'LT'}           # no measurable task AU -> position rule
# (BC left this set: its CV-confirmed AU12+AU17 proxy beats position-only, so it
#  routes through the frac-of-peak branch using signal_aus from the fitted params.)
STRONGER_SIDE_ACTIONS = {'BK', 'BS', 'SS'}  # read higher-key-AU-peak hemiface

# --- AUs ---
AU_ORDER = ['AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 'AU09',
            'AU10', 'AU12', 'AU14', 'AU15', 'AU17', 'AU20', 'AU23',
            'AU25', 'AU26', 'AU45']

# --- Batch filter ---
# When non-empty, the curator shows ONLY these patients, in THIS order. Set to
# None/[] to show the full roster. Curation for patients NOT in the list is still
# preserved in the JSON — the filter only changes what's visible this session.
#
# Full 30-patient set = BATCH 1 (the 10 already curated, kept for reference/edits)
# followed by BATCH 2 (20 NEW patients to refine the auto-curator, prioritizing
# the data-starved actions: FR 1->7, SO/SE 5->12, BK 4->15, etc., plus 4 fresh
# controls and all 7 LCA clusters). The done ones show with checkmarks; the new
# ones are todo.
BATCH_1_DONE = [
    "IMG_0422", "IMG_0428", "20240723_175947000_iOS", "20240723_185024000_iOS",
    "IMG_0492", "20240731_181857000_iOS", "IMG_0504", "IMG_0495",
    "20240730_172902000_iOS", "20240731_192357000_iOS",
]
BATCH_2_NEW = [
    # 6 FR-bearers (FR is the scarce action) — also carry SO/LT/SE/WN/BK
    "20250107_204041000_iOS",   # case LCA1 +FR
    "20250204_174155000_iOS",   # case LCA3 +FR
    "20250213_160905000_iOS",   # case LCA3 +FR
    "20250213_193056000_iOS",   # case LCA3 +FR
    "20250402_133347000_iOS",   # case LCA3 +FR
    "20250409_124159000_iOS",   # case LCA1 +FR
    # 4 fresh controls (normal reference)
    "IMG_0433", "IMG_0434", "IMG_0435", "IMG_0438",
    # one case per LCA cluster 0..6
    "20240903_145723000_iOS",   # case LCA0
    "20240730_134811000_iOS",   # case LCA1
    "IMG_2259",                 # case LCA2
    "20240806_140516000_iOS",   # case LCA3
    "IMG_0592",                 # case LCA4
    "IMG_0861",                 # case LCA5
    "20240820_173737000_iOS",   # case LCA6
    # fill: weak-action coverage (SO/LT/SE/BK/WN/BC/PL/ET)
    "20240820_141306000_iOS",   # case LCA3
    "20250312_124816000_iOS",   # case LCA1
    "20250326_184059000_iOS",   # case LCA1
]
BATCH_PIDS = BATCH_1_DONE + BATCH_2_NEW

# --- Canonical action order (shown in the action map; absent ones disabled) ---
# The analyzed set (feeds the framework) is marked ANALYZED_ACTIONS.
CANONICAL_ACTIONS = ['BL', 'BS', 'SS', 'RE', 'FR', 'ES', 'ET', 'BK',
                     'WN', 'SE', 'SO', 'PL', 'BC', 'LT']
ANALYZED_ACTIONS = ['BL', 'BS', 'SS', 'RE', 'ES', 'ET', 'SE', 'SO']
ACTION_NAMES = {
    'BL': 'Baseline (rest)', 'BS': 'Big Smile', 'SS': 'Soft Smile',
    'RE': 'Raise Eyebrows', 'FR': 'Frown', 'ES': 'Eyes Soft',
    'ET': 'Eyes Tight', 'BK': 'Blink', 'WN': 'Wrinkle Nose',
    'SE': 'Say E', 'SO': 'Say O', 'PL': 'Pucker Lips',
    'BC': 'Blow Cheeks', 'LT': 'Lower Teeth',
}

# --- Exclusion reason taxonomy (hover buttons) ---
# Kept short so they fit in a hover overlay; "Other" catches the rest.
EXCLUSION_REASONS = ['Blink', 'Eyes open', 'Transition', 'Non-compliant',
                     'Blur', 'Other']
# Short labels for the compact hover buttons
EXCLUSION_SHORT = {
    'Blink': 'Blink', 'Eyes open': 'EyesOpen', 'Transition': 'Trans',
    'Non-compliant': 'NonComp', 'Blur': 'Blur', 'Other': 'Other',
}

# --- Display (face-cropped thumbnails, portrait) ---
# Larger tiles: the researcher needs facial detail to judge each frame. The
# upper area is kept compact (see curator_window) to give these room.
THUMB_W = 176
THUMB_H = 212
THUMB_PAD = 6
GRID_COLS = 7   # fallback; actual columns computed from available width

# --- Auto-keep (plateau) selection ---
# Keep frames where the action is sustained near its peak; reject onset/offset
# edges. A frame is auto-kept if its in-set AU sum >= AUTO_KEEP_FRAC * the
# patient's own peak in-set sum for that action (and above a small floor).
AUTO_KEEP_FRAC = 0.62
AUTO_KEEP_FLOOR = 0.5   # absolute in-set-sum floor so near-zero peaks don't keep noise

# --- Face-crop parameters (source frames are 1080x1920 portrait; face ~35%) ---
FACE_DETECT_SAMPLES = 6      # frames sampled across video for robust bbox
FACE_EXPAND_TOP = 0.38       # forehead/brows margin (fraction of face height)
FACE_EXPAND_BOTTOM = 0.30    # chin margin
FACE_EXPAND_SIDE = 0.12      # cheek margin (fraction of face width)

# --- Status / state colors (tuned for the S2 light theme) ---
COLOR_INCLUDED = '#2e8b2e'    # green border = in curated good set
COLOR_EXCLUDED = '#c0392b'    # red = excluded with reason
COLOR_AVAILABLE = '#bbbbbb'   # gray = in action but not selected
COLOR_CURRENT = '#4682B4'     # S2 highlight blue = current action / patient
COLOR_DONE = '#2e8b2e'
COLOR_TODO = '#bbbbbb'
COLOR_PARTIAL = '#e0a020'

# --- S2.5 mockup palette (flat, warm off-white, semantic green/keep) ---
THEME = {
    'app_bg':        '#f4f2ec',   # warm off-white app background
    'card_bg':       '#ffffff',
    'card_border':   '#e3e0d8',
    'text':          '#2c2c2c',
    'text_muted':    '#8a857c',
    'keep_border':   '#2e6b3e',   # representative
    'keep_fill':     '#e9f1e3',
    'keep_check':    '#2e6b3e',
    'range_block':   '#2e6b3e',   # R1/R2 blocks on the clip timeline
    'reject_border': '#d0cdc4',
    'reject_fill':   '#f0eee9',
    'reject_text':   '#a8a39a',
    'review_dot':    '#9c6b1f',   # low-confidence marker
    'review_text':   '#9c6b1f',
    'review_card':   '#f6ecd9',
    'primary':       '#5663c0',   # 'save & next action'
    'primary_hover': '#6571cf',
    'timeline_bg':   '#eceae3',
}

# --- S2-matched UI theme (light) — kept for back-compat references ---
UI_COLORS = {
    'section_bg': '#f5f5f5',
    'section_border': '#cccccc',
    'highlight': '#4682B4',
    'text_normal': '#333333',
    'text_inactive': '#999999',
    'window_bg': '#fafafa',
}
STANDARD_BUTTON_STYLE = (
    "QPushButton { background-color: #f0f0f0; color: #333333; "
    "border: 1px solid #cccccc; border-radius: 3px; padding: 4px 8px; "
    "min-height: 22px; } QPushButton:hover { background-color: #e0e0e0; } "
    "QPushButton:pressed { background-color: #d0d0d0; border: 1px solid #999999; }")
PRIMARY_BUTTON_STYLE = (
    "QPushButton { background-color: #4682B4; color: white; "
    "border: 1px solid #3a6a94; border-radius: 3px; padding: 4px 10px; "
    "min-height: 30px; font-weight: bold; } "
    "QPushButton:hover { background-color: #5692c4; } "
    "QPushButton:pressed { background-color: #3a6a94; }")
GROUP_BOX_STYLE = (
    "QGroupBox { font-family: Arial; font-weight: bold; "
    "border: 1px solid #cccccc; border-radius: 3px; margin-top: 12px; "
    "padding-top: 8px; } QGroupBox::title { subcontrol-origin: margin; "
    "subcontrol-position: top left; left: 10px; padding: 0 3px; }")

# Per-frame AU label: which AUs to show under each thumbnail (in-set is dynamic;
# we always also show AU45 to make blinks visible).
ALWAYS_SHOW_AU = ['AU45']

# --- Per-action categorical colors (timeline context blocks) ---
# S2 uses a single highlight blue for all coded ranges; this is a distinct-per-
# action palette so the timeline reads as a sequence. Greens/grays/ambers are
# deliberately avoided — those are reserved for the keep/reject/flag defrag.
ACTION_COLORS = {
    'BL': '#90a4ae',  # blue-gray (baseline)
    'RE': '#5b8def',  # blue
    'FR': '#7e57c2',  # purple
    'ES': '#26c6da',  # cyan
    'ET': '#0097a7',  # dark cyan
    'BK': '#ab47bc',  # magenta
    'WN': '#ec407a',  # pink
    'SS': '#ff8a65',  # light orange
    'BS': '#f4511e',  # deep orange
    'SE': '#8d6e63',  # brown
    'SO': '#6d4c41',  # dark brown
    'PL': '#5c6bc0',  # indigo
    'BC': '#42a5f5',  # light blue
    'LT': '#78909c',  # slate
}
# Defrag (per-frame status) colors for the current action's window(s)
DEFRAG_KEEP = '#2e8b2e'    # representative (green)
DEFRAG_REJECT = '#c4c0b6'  # rejected (gray)
DEFRAG_FLAG = '#EF6C00'    # flagged for review (deep orange; distinct from gold)
ACTIVE_GOLD = '#FFC400'    # active-window border (bright gold, not neon yellow)
