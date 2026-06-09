"""
S2 <-> S2.5 re-score handoff (shared protocol).

File-based contract in a shared dir. S2.5 writes request.json to ask S2 to
re-score one patient's video; S2 claims it (rename to .inflight), loads the file
bypassing its dialog, and on save/cancel writes response.json. S2.5 watches for
the response and offers to reload.

Both apps import this module (S2 from its own dir; S2.5 has its own copy with the
same logic). Pure stdlib so it works in either Python.
"""
import json
import os
import time
from pathlib import Path

HANDOFF_DIR = Path("/Users/johnwilsoniv/Documents/SplitFace/"
                   "S25 Curated Files/.s2_handoff")
REQUEST = HANDOFF_DIR / "request.json"
INFLIGHT = HANDOFF_DIR / "request.inflight"
RESPONSE = HANDOFF_DIR / "response.json"

# A request is only honored if it is newer than this many seconds (so S2 doesn't
# pick up a stale request from a previous session on a normal launch).
FRESH_WINDOW_S = 120


def _ensure_dir():
    HANDOFF_DIR.mkdir(parents=True, exist_ok=True)


def write_request(patient_id, input_videos, input_dir):
    """S2.5 -> S2. Clear any old response first."""
    _ensure_dir()
    try:
        RESPONSE.unlink()
    except OSError:
        pass
    payload = {
        "patient_id": patient_id,
        "input_videos": [str(v) for v in input_videos],
        "input_dir": str(input_dir),
        "return_to": "S2.5",
        "ts": time.time(),
    }
    tmp = REQUEST.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, REQUEST)
    return payload


def claim_request(max_age_s=FRESH_WINDOW_S):
    """S2 side: if a FRESH request exists, atomically claim it (rename to
    .inflight) and return its payload; else None. Renaming prevents double-pickup
    and stops a normal (non-handoff) launch from consuming an old request."""
    if not REQUEST.exists():
        return None
    try:
        payload = json.loads(REQUEST.read_text())
    except Exception:
        return None
    if (time.time() - float(payload.get("ts", 0))) > max_age_s:
        return None
    try:
        os.replace(REQUEST, INFLIGHT)   # atomic claim
    except OSError:
        return None
    return payload


def write_response(patient_id, status, outputs=None):
    """S2 -> S2.5 on save ('completed') or cancel ('cancelled')."""
    _ensure_dir()
    payload = {
        "patient_id": patient_id,
        "status": status,
        "outputs": [str(o) for o in (outputs or [])],
        "ts": time.time(),
    }
    tmp = RESPONSE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, RESPONSE)
    # clear the in-flight marker now that we've answered
    try:
        INFLIGHT.unlink()
    except OSError:
        pass
    return payload


def read_response():
    if not RESPONSE.exists():
        return None
    try:
        return json.loads(RESPONSE.read_text())
    except Exception:
        return None


def clear_response():
    for p in (RESPONSE, INFLIGHT):
        try:
            p.unlink()
        except OSError:
            pass
