"""
app.py — EE200 Course Project, Q3B: 'Signals to Softwares'

A small Shazam-style audio identifier: indexes a song library as
spectrogram-fingerprint hashes, then identifies a query clip by an
offset-histogram vote. Wraps the Q3A pipeline (fingerprint.py) in an
interactive Streamlit app with three modes:

  LIBRARY  — browse the indexed song database (constellation thumbnails)
  IDENTIFY — upload / pick a clip, see every intermediate step, get a match
  BATCH    — identify many clips at once -> results.csv (filename,prediction)
"""

import io
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import fingerprint as fp

# ───────────────────────────────────── Config ──────────────────────────────────
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
THUMB_DIR = os.path.join(DATA_DIR, "thumbnails")
SAMPLE_DIR = os.path.join(APP_DIR, "samples")
DB_PARTS = [os.path.join(DATA_DIR, "fingerprint_db_0.pkl"),
            os.path.join(DATA_DIR, "fingerprint_db_1.pkl")]

st.set_page_config(page_title="EE200: Audio Fingerprinting",
                    page_icon="🎵", layout="wide")

# ───────────────────────────────────── Styling ─────────────────────────────────
st.markdown("""
<style>
:root {
  --bg: #0a0f0d; --panel: #101814; --accent: #2ee6b8; --accent-dim: #1c8f72;
  --text: #d7e6e0; --muted: #7d9189;
}
.stApp { background-color: var(--bg); }
h1, h2, h3, h4 { color: var(--text) !important; }
.ee-title { color: var(--text); font-weight: 800; }
.ee-title span { color: var(--accent); }
.ee-sub { color: var(--muted); letter-spacing: .12em; font-size: 0.78rem;
          text-transform: uppercase; }
.ee-card {
  background: var(--panel); border: 1px solid #1c2620; border-radius: 10px;
  padding: 10px 12px; margin-bottom: 8px;
}
.ee-pill {
  display:inline-block; background:#0d1a16; border:1px solid #1c2620;
  border-radius:8px; padding:6px 10px; margin-right:6px; color:var(--muted);
  font-size:0.75rem;
}
.ee-pill b { color: var(--accent); display:block; font-size:1rem; }
.match-found {
  background: linear-gradient(180deg, #0e1f18, #0a0f0d); border:1px solid var(--accent-dim);
  border-radius: 12px; padding: 18px 20px; margin-top: 10px;
}
.match-title { color: var(--accent); font-size: 1.8rem; font-weight: 800; margin:0; }
.no-match { background:#220f0f; border:1px solid #6b2a2a; border-radius:12px;
            padding:18px 20px; margin-top:10px; color:#e6a3a3; }
.bar-row { display:flex; align-items:center; margin-bottom:6px; font-size:0.85rem; color:var(--text);}
.bar-name { width: 220px; flex-shrink:0; color: var(--muted);}
.bar-track { flex:1; background:#101814; border-radius:6px; height:14px; overflow:hidden; margin:0 8px;}
.bar-fill { height:100%; background: linear-gradient(90deg, var(--accent-dim), var(--accent)); }
.bar-score { width: 70px; text-align:right; color: var(--text); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────── Data loading ──────────────────────────────
@st.cache_resource(show_spinner="Loading fingerprint database (one-time)...")
def load_database():
    import pickle
    db_pair, db_single, stats = {}, {}, {}
    for path in DB_PARTS:
        with open(path, "rb") as fh:
            part = pickle.load(fh)
        # the two halves were split by hash(key)%2, so they're disjoint —
        # a plain dict union is correct (no value-level merging needed) & fast.
        db_pair = {**db_pair, **part["db_pair"]}
        if part.get("db_single"):
            db_single = part["db_single"]
        if part.get("stats"):
            stats = part["stats"]
    return db_pair, db_single, stats


@st.cache_data(show_spinner=False)
def list_sample_files():
    if not os.path.isdir(SAMPLE_DIR):
        return []
    return sorted(f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith((".mp3", ".wav")))


def thumb_path(song_name):
    p = os.path.join(THUMB_DIR, f"{song_name}.png")
    return p if os.path.exists(p) else None


# ─────────────────────────────────── Header ────────────────────────────────────
st.markdown(
    '<div class="ee-title" style="font-size:2.2rem;">EE200<span>:</span> Audio Fingerprinting</div>'
    '<div class="ee-sub">Signals, Systems &amp; Networks &middot; Project Demo (Q3B)</div>'
    '<p style="color:#9fb3ab; margin-top:6px;">Index a library of songs as spectrogram '
    'fingerprints, then identify any short clip against it.</p>',
    unsafe_allow_html=True)

db_pair, db_single, stats = load_database()

tab_library, tab_identify, tab_batch = st.tabs(["◆ LIBRARY", "◎ IDENTIFY", "▦ BATCH"])

# ════════════════════════════════════ LIBRARY ═══════════════════════════════════
with tab_library:
    st.markdown("#### Indexed song library")
    st.caption(f"{len(stats)} songs &middot; {len(db_pair):,} unique pair-hash keys &middot; "
               f"{len(db_single):,} unique single-peak keys", unsafe_allow_html=True)

    names = sorted(stats.keys())
    cols_per_row = 4
    for row_start in range(0, len(names), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, name in zip(cols, names[row_start:row_start + cols_per_row]):
            with col:
                tp = thumb_path(name)
                if tp:
                    st.image(tp, use_container_width=True)
                s = stats[name]
                st.markdown(
                    f"<div class='ee-card'><b>{name}</b><br>"
                    f"<span class='ee-sub'>{s['peaks']:,} peaks &middot; "
                    f"{s['pair_hashes']:,} hashes &middot; {s['duration_s']:.0f}s</span></div>",
                    unsafe_allow_html=True)


# ═══════════════════════════════════ helpers ════════════════════════════════════
def run_pipeline(y, sr):
    return fp.match_clip_verbose(y, sr, db_pair, db_single)


def render_timing_strip(timings):
    stages = [("SPECTROGRAM", "spectrogram"), ("CONSTELLATION", "constellation"),
              ("HASHING", "hashing"), ("DB LOOKUP", "db_lookup"), ("SCORING", "scoring")]
    cols = st.columns(len(stages) + 1)
    for col, (label, key) in zip(cols, stages):
        col.markdown(f"<div class='ee-pill'>{label}<b>{timings[key]*1000:.0f} ms</b></div>",
                     unsafe_allow_html=True)
    cols[-1].markdown(
        f"<div class='ee-pill'>TOTAL<b>{timings['total']*1000:.0f} ms</b></div>",
        unsafe_allow_html=True)


def render_candidate_bars(score_dict, top=5):
    ranked = fp.candidate_ranking(score_dict, top=top)
    if not ranked:
        st.caption("No hash collisions found against any indexed song.")
        return
    max_score = max(s for _, s in ranked) or 1
    html = ""
    for name, score in ranked:
        pct = max(2, int(100 * score / max_score))
        html += (f"<div class='bar-row'><div class='bar-name'>{name}</div>"
                 f"<div class='bar-track'><div class='bar-fill' style='width:{pct}%;'></div></div>"
                 f"<div class='bar-score'>{score}</div></div>")
    st.markdown(html, unsafe_allow_html=True)


def render_visuals(S_q, peaks_q, off_p, sp, matched_song):
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.imshow(S_q, origin="lower", aspect="auto", cmap="magma")
        ax.set_title("Query spectrogram", fontsize=10)
        ax.set_xlabel("time (frames)"); ax.set_ylabel("freq bin")
        st.pyplot(fig, clear_figure=True)
    with c2:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        if peaks_q:
            t = [p[0] for p in peaks_q]; f = [p[1] for p in peaks_q]
            ax.scatter(t, f, s=4, c="#2ee6b8")
        ax.set_facecolor("#0b1411")
        ax.set_title(f"Constellation ({len(peaks_q)} peaks)", fontsize=10)
        ax.set_xlabel("time (frames)"); ax.set_ylabel("freq bin")
        st.pyplot(fig, clear_figure=True)

    fig, ax = plt.subplots(figsize=(10, 3))
    ranked = fp.candidate_ranking(sp, top=4)
    palette = ["#2ee6b8", "#6699ff", "#ff9966", "#cc88ff"]
    for color, (name, score) in zip(palette, ranked):
        d = off_p.get(name, {})
        if d:
            ax.vlines(list(d.keys()), 0, list(d.values()), color=color, alpha=0.8, lw=1,
                      label=f"{name} (peak={score})")
    ax.set_title("Offset histogram — true match shows one tall spike", fontsize=10)
    ax.set_xlabel("time offset (frames)"); ax.set_ylabel("hash matches")
    ax.legend(fontsize=8)
    st.pyplot(fig, clear_figure=True)


# ════════════════════════════════════ IDENTIFY ══════════════════════════════════
with tab_identify:
    st.markdown("#### Identify a clip")
    uploaded = st.file_uploader("Upload a query clip", type=["wav", "mp3", "flac", "ogg", "m4a"],
                                label_visibility="collapsed")
    st.caption("200MB per file • WAV, MP3, FLAC, OGG, M4A")

    samples = list_sample_files()
    st.markdown("###### or try a sample")
    chosen_sample = None
    for sname in samples:
        c1, c2 = st.columns([5, 1])
        c1.audio(os.path.join(SAMPLE_DIR, sname))
        if c2.button("Try", key=f"try_{sname}"):
            chosen_sample = sname
            st.session_state["selected_sample"] = sname

    active_sample = chosen_sample or st.session_state.get("selected_sample")

    query_bytes, query_label = None, None
    if uploaded is not None:
        query_bytes, query_label = uploaded.read(), uploaded.name
    elif active_sample:
        with open(os.path.join(SAMPLE_DIR, active_sample), "rb") as fh:
            query_bytes, query_label = fh.read(), active_sample

    if st.button("Identify", type="primary", disabled=query_bytes is None):
        y = fp.decode_audio(query_bytes)
        mp, ms, sp, ss, off_p, off_s, peaks_q, S_q, timings = run_pipeline(y, fp.SR)
        matched, top_song, top_score, runner, runner_score = fp.decide_match(sp)

        render_timing_strip(timings)

        if matched:
            margin = (top_score / runner_score) if runner_score else float("inf")
            margin_txt = f"{margin:.0f}&times;" if margin != float("inf") else "&infin;&times;"
            st.markdown(
                f"<div class='match-found'><div class='ee-sub'>MATCH FOUND</div>"
                f"<p class='match-title'>{matched}</p>"
                f"<div class='ee-sub'>cluster score {top_score} &middot; "
                f"{margin_txt} the runner-up ({runner or '—'})</div></div>",
                unsafe_allow_html=True)
        else:
            st.markdown(
                "<div class='no-match'><div class='ee-sub'>NO CONFIDENT MATCH</div>"
                f"<p style='font-size:1.3rem;margin:6px 0;'>Best guess: {top_song or '—'} "
                f"(score {top_score})</p>"
                "<div class='ee-sub'>No candidate cleared the confidence threshold.</div></div>",
                unsafe_allow_html=True)

        st.markdown("###### candidate scores")
        render_candidate_bars(sp)

        with st.expander("Step 1-3 · Feature extraction & matching visuals", expanded=True):
            render_visuals(S_q, peaks_q, off_p, sp, matched)


# ═══════════════════════════════════════ BATCH ══════════════════════════════════
with tab_batch:
    st.markdown("#### Identify many clips at once")
    st.caption("Upload a set of query clips. Each is identified against the currently "
               "indexed library, and the results are written to a standardised "
               "`results.csv` with columns `filename, prediction`. `prediction` is the "
               "matched track's filename without extension, or `none` when no candidate "
               "clears the confidence threshold.")

    batch_files = st.file_uploader("Upload clips", type=["wav", "mp3", "flac", "ogg", "m4a"],
                                   accept_multiple_files=True, key="batch_uploader")

    use_samples = st.checkbox("Also include the 5 sample clips", value=False)

    if st.button("Run batch", type="primary"):
        jobs = []
        for f in (batch_files or []):
            jobs.append((f.name, f.read()))
        if use_samples:
            for sname in list_sample_files():
                with open(os.path.join(SAMPLE_DIR, sname), "rb") as fh:
                    jobs.append((sname, fh.read()))

        if not jobs:
            st.warning("Upload at least one clip (or check the sample box) first.")
        else:
            progress = st.progress(0.0, text="Identifying...")
            rows = []
            for i, (fname, fbytes) in enumerate(jobs, 1):
                y = fp.decode_audio(fbytes)
                _, _, sp, _, _, _, _, _, _ = run_pipeline(y, fp.SR)
                matched, top_song, top_score, runner, runner_score = fp.decide_match(sp)
                rows.append({"filename": fname, "prediction": matched if matched else "none"})
                progress.progress(i / len(jobs), text=f"Identifying... {i}/{len(jobs)}")
            progress.empty()

            df = pd.DataFrame(rows, columns=["filename", "prediction"])
            st.dataframe(df, use_container_width=True, hide_index=True)

            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download results.csv", data=csv_bytes,
                               file_name="results.csv", mime="text/csv")
