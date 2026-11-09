"""All tunable thresholds in one place. Override any key with --config my.json."""
import json

DEFAULTS = {
    # analysis resolution / speed
    "analysis_short_side": 480,   # frames are analysed with their SHORTER side at this many px
                                  # (360 px wide was fine for vertical video but left faces ~30 px
                                  # tall in landscape clips: too small for MediaPipe)

    # --- technical: video ---
    "black_min_dur": 0.10,        # s, shortest black run reported
    "black_pix_th": 0.10,         # ffmpeg blackdetect pixel threshold
    "black_blocker_dur": 0.30,    # s, black longer than this (unscripted) = blocker
    "freeze_noise_db": -72,       # ffmpeg freezedetect noise tolerance (strict: a talking head on a static set
                                  # changes very few pixels, -60 dB calls it frozen)
    "freeze_min_dur": 0.8,        # s, shorter holds are treated as intentional
    "freeze_blocker_dur": 1.5,

    # --- technical: audio ---
    "silence_db": -50,            # dBFS, below this counts as silence
    "silence_min_dur": 1.0,       # s
    "silence_no_script_major": 3.0,  # s, unscripted silence longer than this = major
    "clip_level": 0.95,           # |sample| at/above this counts as clipped (AAC decoding rings below 1.0)
    "clip_ratio": 0.005,          # >0.5% clipped samples in a 100 ms window = distorted
    "clip_merge_gap": 0.5,        # s, join clipped windows separated by short gaps (pauses between words)
    "clip_min_dur": 0.3,          # s of consecutive distorted windows
    "spike_db": 10,               # 100 ms window this many dB above the episode's 95th-pct loudness = spike

    # --- visual continuity ---
    "cut_threshold": 0.45,        # Bhattacharyya distance between frames = shot cut
    "min_shot_dur": 0.4,
    "bg_cluster_dist": 0.35,      # background hist distance to be "same location"
    "torso_cluster_dist": 0.50,   # torso hist distance to be "same outfit" (real lighting varies)
    "torso_min_face_h": 0.10,     # only judge clothing under a face at least this tall (share of frame height)
    "torso_min_samples": 2,       # a new outfit must be seen in >= this many samples (one odd frame is noise)
    "min_face_frac": 0.004,       # face box area / frame area
    "glitch_max_dur": 6.0,        # s, longest single "wrong background" shot we call a glitch insert
    "dark_shot_v": 30,            # shots darker than this (0-255) are skipped by continuity (black/fades)

    # --- lip sync ---
    "lip_min_window": 1.5,        # s of dialogue needed to judge
    "lip_min_face_cov": 0.6,      # face visible in >= 60% of window frames
    "lip_max_lag": 0.5,           # s, search range for A/V offset
    "lip_min_corr": 0.30,         # below this at every lag = mouth doesn't follow speech
    "lip_offset_flag": 0.20,      # s, |best lag| above this = A/V offset
    "lip_static_std": 0.004,      # mouth signal std below this = mouth not moving

    # script-aware tolerance around timestamps
    "script_tol": 0.75,

    # optional vision-LLM verifier for continuity flags
    "llm_provider": None,         # None | "gemini" | "openai_compatible"
    "llm_model": None,
}


def load_config(path=None, overrides=None):
    cfg = dict(DEFAULTS)
    if path:
        with open(path, encoding="utf-8") as f:
            cfg.update(json.load(f))
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg
