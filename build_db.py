"""
build_db.py
EE200 Course Project — Q3B

One-time indexing script: fingerprints every song in `songs/` and writes
the resulting database + per-song constellation thumbnails to `data/`,
so the Streamlit app (app.py) can load instantly instead of re-indexing
on every run / every redeploy.

Run once:
    python build_db.py

Outputs
-------
data/fingerprint_db.pkl   pickled dict: {db_pair, db_single, stats, config}
data/thumbnails/<song>.png   small constellation-map thumbnail per song
"""

import glob
import os
import pickle
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import fingerprint as fp

SONG_DIR = "songs"
DATA_DIR = "data"
THUMB_DIR = os.path.join(DATA_DIR, "thumbnails")


def make_thumbnail(peaks, S_db, song_name, out_path):
    """Small dark constellation-map thumbnail (no axes) for the Library grid."""
    t = np.array([p[0] for p in peaks])
    f = np.array([p[1] for p in peaks])
    fig, ax = plt.subplots(figsize=(2.4, 1.6), dpi=100)
    fig.patch.set_facecolor("#0b1411")
    ax.set_facecolor("#0b1411")
    ax.scatter(t, f, s=1.2, c="#2ee6b8", alpha=0.75, linewidths=0)
    ax.set_xlim(0, S_db.shape[1])
    ax.set_ylim(0, S_db.shape[0])
    ax.axis("off")
    plt.tight_layout(pad=0)
    fig.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main():
    os.makedirs(THUMB_DIR, exist_ok=True)
    song_files = sorted(glob.glob(os.path.join(SONG_DIR, "*.mp3")))
    if not song_files:
        raise FileNotFoundError(f"No .mp3 files found in '{SONG_DIR}/'.")
    print(f"Found {len(song_files)} songs. Indexing...\n")

    db_p, db_s, stats = {}, {}, {}
    from collections import defaultdict
    db_p_acc, db_s_acc = defaultdict(list), defaultdict(list)

    t_start = time.time()
    for i, path in enumerate(song_files, 1):
        name = os.path.splitext(os.path.basename(path))[0]
        t0 = time.time()
        y = fp.decode_audio(path)
        duration = len(y) / fp.SR
        S_db = fp.compute_spectrogram(y)
        peaks = fp.extract_peaks(S_db)
        ph = fp.generate_pair_hashes(peaks)
        sh = fp.generate_single_hashes(peaks)

        for h, t in ph:
            db_p_acc[h].append((name, t))
        for h, t in sh:
            db_s_acc[h].append((name, t))

        stats[name] = {
            "peaks": len(peaks),
            "pair_hashes": len(ph),
            "single_hashes": len(sh),
            "duration_s": duration,
            "filename": os.path.basename(path),
        }

        thumb_path = os.path.join(THUMB_DIR, f"{name}.png")
        make_thumbnail(peaks, S_db, name, thumb_path)

        dt = time.time() - t0
        print(f"[{i:>2}/{len(song_files)}] {name:45s} "
              f"{duration:6.1f}s  {len(peaks):>6,} peaks  {len(ph):>7,} hashes  ({dt:.1f}s)")

    db_pair = dict(db_p_acc)
    db_single = dict(db_s_acc)

    payload = {
        "db_pair": db_pair,
        "db_single": db_single,
        "stats": stats,
        "config": {
            "SR": fp.SR, "N_FFT": fp.N_FFT, "HOP_LEN": fp.HOP_LEN,
            "FAN_VALUE": fp.FAN_VALUE, "MIN_DT": fp.MIN_DT, "MAX_DT": fp.MAX_DT,
            "NBHD": fp.NBHD, "THRESH_DB": fp.THRESH_DB,
        },
    }
    out_path = os.path.join(DATA_DIR, "fingerprint_db.pkl")
    with open(out_path, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    total_t = time.time() - t_start
    total_peaks = sum(s["peaks"] for s in stats.values())
    total_hashes = sum(s["pair_hashes"] for s in stats.values())
    print(f"\nDone in {total_t:.1f}s.")
    print(f"Songs indexed     : {len(stats)}")
    print(f"Total peaks       : {total_peaks:,}")
    print(f"Total pair-hashes : {total_hashes:,}")
    print(f"Unique pair keys  : {len(db_pair):,}")
    print(f"Unique single keys: {len(db_single):,}")
    print(f"Saved database -> {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)")
    print(f"Saved thumbnails -> {THUMB_DIR}/")


if __name__ == "__main__":
    main()
