"""
fingerprint.py
EE200 Course Project — Q3B: Signals to Softwares
Core audio-fingerprinting engine shared by the Streamlit app (app.py) and the
database-building script (build_db.py).

This reimplements, in plain NumPy, exactly the same pipeline developed in
Q3A ('Sonic Signatures'):
    decode -> STFT spectrogram -> constellation peaks -> pair/single hashes
    -> hash database -> offset-histogram matching

Design note
------------
Q3A's notebook used librosa + scipy for convenience inside a Jupyter
environment. For the *deployed* app we re-derive the identical math
(short-time Fourier transform, 2-D local-maximum peak picking, pair-hash
generation, offset-histogram voting) directly in NumPy, and decode audio with
ffmpeg (a single subprocess call) instead of librosa/audioread/soundfile.
This keeps the deployed app's dependency footprint tiny (numpy + streamlit +
matplotlib + ffmpeg binary), avoids librosa's slow cold-start / native-build
issues on free hosting tiers, and is mathematically equivalent to the Q3A
notebook's approach.
"""

import io
import os
import subprocess
from collections import defaultdict

import numpy as np

# ─────────────────────────── Default configuration ───────────────────────────
SR = 22050          # uniform resampling rate (Hz)         — same as Q3A
N_FFT = 2048        # STFT window length                   — same as Q3A
HOP_LEN = 512        # STFT hop length (samples)            — same as Q3A
FAN_VALUE = 15          # max forward-neighbours per anchor peak
MIN_DT = 1           # minimum Dt between anchor & target (frames)
MAX_DT = 200         # maximum Dt
NBHD = (20, 20)     # (freq_bins, time_frames) neighbourhood for peak-picking
THRESH_DB = -40         # dB threshold relative to spectrogram maximum


# ─────────────────────────────── Audio decoding ───────────────────────────────
def decode_audio(path_or_bytes, sr=SR):
    """
    Decode any audio file (mp3/wav/flac/ogg/m4a) to a mono float32 NumPy array
    in [-1, 1], resampled to `sr`, using ffmpeg as the only external dependency.

    Parameters
    ----------
    path_or_bytes : str | bytes
        A filesystem path, OR raw file bytes (e.g. from a Streamlit uploader).
    sr : int
        Target sample rate.

    Returns
    -------
    y : np.ndarray (float32, mono, normalised to [-1, 1])
    """
    cmd = ["ffmpeg", "-v", "error", "-i",
           "-" if isinstance(path_or_bytes, bytes) else str(path_or_bytes),
           "-ar", str(sr), "-ac", "1", "-f", "s16le", "-"]

    input_bytes = path_or_bytes if isinstance(path_or_bytes, bytes) else None
    proc = subprocess.run(cmd, input=input_bytes, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to decode audio: {proc.stderr.decode(errors='ignore')}")

    pcm = np.frombuffer(proc.stdout, dtype=np.int16)
    y = pcm.astype(np.float32) / 32768.0
    return y


# ───────────────────────────────── Spectrogram ────────────────────────────────
def _sliding_windows(a, window, hop):
    """Return shape (n_frames, window) view of overlapping frames of 1-D array a."""
    n_frames = 1 + (len(a) - window) // hop
    if n_frames <= 0:
        raise ValueError("Signal shorter than one STFT window.")
    stride = a.strides[0]
    shape = (n_frames, window)
    strides = (stride * hop, stride)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def compute_spectrogram(y, sr=SR, n_fft=N_FFT, hop_length=HOP_LEN):
    """
    Short-Time Fourier Transform -> log-magnitude (dB) spectrogram.

    Equivalent to:  librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft, hop)), ref=np.max)

    Returns
    -------
    S_db : np.ndarray, shape (n_fft//2 + 1, n_frames)
    """
    y = np.asarray(y, dtype=np.float64)
    pad = n_fft // 2
    y_padded = np.pad(y, pad, mode="reflect")

    window = np.hanning(n_fft)
    frames = _sliding_windows(y_padded, n_fft, hop_length).copy()
    frames *= window

    spec = np.fft.rfft(frames, axis=1).T          # (n_fft//2+1, n_frames)
    mag = np.abs(spec)

    ref = mag.max() if mag.max() > 0 else 1.0
    amin = 1e-10
    S_db = 20.0 * np.log10(np.maximum(mag, amin) / ref)
    S_db = np.maximum(S_db, -80.0)                  # top_db=80, matches librosa default
    return S_db.astype(np.float32)


# ─────────────────────────────── Peak extraction ──────────────────────────────
def _max_filter_1d(a, size, axis):
    """Sliding-window maximum along one axis (replaces scipy.ndimage.maximum_filter)."""
    pad_before = (size - 1) // 2
    pad_after = size // 2
    pad_width = [(0, 0)] * a.ndim
    pad_width[axis] = (pad_before, pad_after)
    padded = np.pad(a, pad_width, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, size, axis=axis)
    return windows.max(axis=-1)


def _max_filter_2d(a, size):
    """Separable rectangular max filter: equivalent to scipy.ndimage.maximum_filter(a, size)."""
    return _max_filter_1d(_max_filter_1d(a, size[0], axis=0), size[1], axis=1)


def extract_peaks(S_db, neighborhood=NBHD, threshold_db=THRESH_DB):
    """
    Extract spectral peaks (local maxima) from a log-magnitude spectrogram.

    Parameters
    ----------
    S_db         : 2-D ndarray  (freq_bins x time_frames)
    neighborhood : (freq_bins, time_frames) neighbourhood size for 2-D max-filter
    threshold_db : minimum value relative to spectrogram maximum (dB)

    Returns
    -------
    peaks : list of (time_frame_idx, freq_bin_idx) tuples, sorted by time
    """
    loc_max = _max_filter_2d(S_db, neighborhood)
    is_peak = (S_db >= loc_max) & (S_db > S_db.max() + threshold_db)
    f_idx, t_idx = np.where(is_peak)
    return sorted(zip(t_idx.tolist(), f_idx.tolist()), key=lambda p: p[0])


# ──────────────────────────────── Hash generation ─────────────────────────────
def generate_pair_hashes(peaks, fan_value=FAN_VALUE, min_dt=MIN_DT, max_dt=MAX_DT):
    """
    For each anchor peak (t1,f1), pair it with up to fan_value forward peaks
    (t2,f2) where t2-t1 is in [min_dt, max_dt] frames.

    Returns list of (hash_key_tuple, anchor_time_frame).
    hash_key = (int(f1), int(f2), int(dt))
    """
    hashes, n = [], len(peaks)
    for i in range(n):
        t1, f1 = peaks[i]
        cnt = 0
        for j in range(i + 1, n):
            if cnt >= fan_value:
                break
            t2, f2 = peaks[j]
            dt = t2 - t1
            if dt < min_dt:
                continue
            if dt > max_dt:
                break
            hashes.append(((int(f1), int(f2), int(dt)), t1))
            cnt += 1
    return hashes


def generate_single_hashes(peaks):
    """Single-peak fingerprints h=(f_bin,) — used for comparison."""
    return [((int(f),), t) for t, f in peaks]


# ──────────────────────────────────── Database ────────────────────────────────
def fingerprint_song(path, n_fft=N_FFT, hop_length=HOP_LEN, sr=SR,
                      nbhd=NBHD, thresh=THRESH_DB, fan=FAN_VALUE,
                      min_dt=MIN_DT, max_dt=MAX_DT):
    """Decode + fingerprint a single song. Returns (peaks, pair_hashes, single_hashes)."""
    y = decode_audio(path, sr)
    S_db = compute_spectrogram(y, sr, n_fft, hop_length)
    peaks = extract_peaks(S_db, nbhd, thresh)
    ph = generate_pair_hashes(peaks, fan, min_dt, max_dt)
    sh = generate_single_hashes(peaks)
    return peaks, ph, sh, S_db


def build_database(song_files, n_fft=N_FFT, hop_length=HOP_LEN, sr=SR,
                    nbhd=NBHD, thresh=THRESH_DB, fan=FAN_VALUE,
                    min_dt=MIN_DT, max_dt=MAX_DT, verbose=True):
    """Fingerprint all songs and populate two lookup tables (pair-hash & single-hash)."""
    db_p, db_s = defaultdict(list), defaultdict(list)
    stats = {}
    for path in song_files:
        name = os.path.splitext(os.path.basename(path))[0]
        peaks, ph, sh, _ = fingerprint_song(path, n_fft, hop_length, sr,
                                             nbhd, thresh, fan, min_dt, max_dt)
        for h, t in ph:
            db_p[h].append((name, t))
        for h, t in sh:
            db_s[h].append((name, t))
        stats[name] = {"peaks": len(peaks), "pair_hashes": len(ph), "path": path}
        if verbose:
            print(f"  {name!r:45s} {len(peaks):>6,} peaks  |  {len(ph):>7,} pair-hashes")
    return dict(db_p), dict(db_s), stats


# ──────────────────────────────── Query matching ──────────────────────────────
def match_clip(y_clip, sr, db_p, db_s,
               n_fft=N_FFT, hop_length=HOP_LEN,
               nbhd=NBHD, thresh=THRESH_DB,
               fan=FAN_VALUE, min_dt=MIN_DT, max_dt=MAX_DT):
    """
    Fingerprint y_clip and query the pair-hash and single-hash databases.

    Returns
    -------
    mp, ms       : matched song name (pair hashes / single hashes), or None
    sp, ss       : {song: peak_score} dicts
    off_p, off_s : {song: {offset: count}} offset histograms
    peaks_q      : peaks extracted from the query
    S_q          : query spectrogram
    """
    S_q = compute_spectrogram(y_clip, sr, n_fft, hop_length)
    peaks_q = extract_peaks(S_q, nbhd, thresh)
    ph_q = generate_pair_hashes(peaks_q, fan, min_dt, max_dt)
    sh_q = generate_single_hashes(peaks_q)

    off_p = defaultdict(lambda: defaultdict(int))
    off_s = defaultdict(lambda: defaultdict(int))

    for h, tq in ph_q:
        for sname, tdb in db_p.get(h, []):
            off_p[sname][tdb - tq] += 1
    for h, tq in sh_q:
        for sname, tdb in db_s.get(h, []):
            off_s[sname][tdb - tq] += 1

    def best(od):
        sc = {s: max(d.values()) for s, d in od.items() if d}
        return (max(sc, key=sc.get) if sc else None), sc

    mp, sp = best(off_p)
    ms, ss = best(off_s)
    return mp, ms, sp, ss, off_p, off_s, peaks_q, S_q


def match_clip_verbose(y_clip, sr, db_p, db_s,
                        n_fft=N_FFT, hop_length=HOP_LEN,
                        nbhd=NBHD, thresh=THRESH_DB,
                        fan=FAN_VALUE, min_dt=MIN_DT, max_dt=MAX_DT):
    """
    Same as match_clip, but also returns a stage-by-stage timing breakdown
    (used by the app's 'pipeline timing' display): spectrogram, constellation,
    hashing, db lookup, scoring — mirroring the 5 stages shown in the demo.
    """
    import time as _time
    timings = {}

    t0 = _time.perf_counter()
    S_q = compute_spectrogram(y_clip, sr, n_fft, hop_length)
    timings["spectrogram"] = _time.perf_counter() - t0

    t0 = _time.perf_counter()
    peaks_q = extract_peaks(S_q, nbhd, thresh)
    timings["constellation"] = _time.perf_counter() - t0

    t0 = _time.perf_counter()
    ph_q = generate_pair_hashes(peaks_q, fan, min_dt, max_dt)
    sh_q = generate_single_hashes(peaks_q)
    timings["hashing"] = _time.perf_counter() - t0

    t0 = _time.perf_counter()
    off_p = defaultdict(lambda: defaultdict(int))
    off_s = defaultdict(lambda: defaultdict(int))
    for h, tq in ph_q:
        for sname, tdb in db_p.get(h, []):
            off_p[sname][tdb - tq] += 1
    for h, tq in sh_q:
        for sname, tdb in db_s.get(h, []):
            off_s[sname][tdb - tq] += 1
    timings["db_lookup"] = _time.perf_counter() - t0

    t0 = _time.perf_counter()

    def best(od):
        sc = {s: max(d.values()) for s, d in od.items() if d}
        return (max(sc, key=sc.get) if sc else None), sc

    mp, sp = best(off_p)
    ms, ss = best(off_s)
    timings["scoring"] = _time.perf_counter() - t0
    timings["total"] = sum(timings.values())

    return mp, ms, sp, ss, off_p, off_s, peaks_q, S_q, timings


def candidate_ranking(score_dict, top=5):
    """Return [(song, score), ...] sorted descending, top-N."""
    return sorted(score_dict.items(), key=lambda kv: -kv[1])[:top]


# ─────────────────────────────── Confidence gate ──────────────────────────────
# A true match produces one tall, decisive offset-histogram bin; an unrelated
# clip produces only small, scattered hash collisions (we measured genuine
# matches scoring in the hundreds-to-thousands vs. ~3-10 for false candidates,
# even down to 0 dB SNR — see Q3A's robustness section). We therefore require
# both an absolute minimum score AND a clear margin over the runner-up before
# declaring a confident match; otherwise we report "no match".
MIN_SCORE = 15
MARGIN_RATIO = 4.0


def decide_match(score_dict, min_score=MIN_SCORE, margin_ratio=MARGIN_RATIO):
    """
    Apply the confidence gate to a {song: score} dict.

    Returns (matched_song_or_None, top_song_or_None, top_score,
              runner_up_song_or_None, runner_up_score)
    `matched_song` is `top_song` only if the confidence gate is cleared,
    otherwise None — but `top_song` (the best raw candidate) is always returned
    so the UI can still show "best guess" when unconfident.
    """
    ranked = candidate_ranking(score_dict, top=2)
    if not ranked:
        return None, None, 0, None, 0
    top_song, top_score = ranked[0]
    runner_song, runner_score = ranked[1] if len(ranked) > 1 else (None, 0)
    confident = (top_score >= min_score) and (top_score >= margin_ratio * max(runner_score, 1))
    return (top_song if confident else None), top_song, top_score, runner_song, runner_score
