"""
build_db_chunk.py — index a slice of songs/*.mp3 and save a partial pickle.
Used because each indexing call runs in a time-limited sandbox session;
chunking lets the full 50-song library be indexed across several calls.

Usage: python3 build_db_chunk.py <start> <end> <chunk_id>
"""
import glob
import os
import pickle
import sys
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import fingerprint as fp

SONG_DIR = "songs"
DATA_DIR = "data"
CHUNK_DIR = os.path.join(DATA_DIR, "chunks")
THUMB_DIR = os.path.join(DATA_DIR, "thumbnails")


def make_thumbnail(peaks, S_db, out_path):
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
    start, end, chunk_id = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    os.makedirs(THUMB_DIR, exist_ok=True)
    os.makedirs(CHUNK_DIR, exist_ok=True)

    song_files = sorted(glob.glob(os.path.join(SONG_DIR, "*.mp3")))[start:end]
    print(f"Chunk {chunk_id}: indexing songs[{start}:{end}] -> {len(song_files)} files\n")

    db_p_acc, db_s_acc = defaultdict(list), defaultdict(list)
    stats = {}
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
            "peaks": len(peaks), "pair_hashes": len(ph), "single_hashes": len(sh),
            "duration_s": duration, "filename": os.path.basename(path),
        }
        make_thumbnail(peaks, S_db, os.path.join(THUMB_DIR, f"{name}.png"))
        print(f"  [{i}/{len(song_files)}] {name:45s} {duration:6.1f}s "
              f"{len(peaks):>6,} peaks  {len(ph):>7,} hashes  ({time.time()-t0:.1f}s)")

    out = {"db_pair": dict(db_p_acc), "db_single": dict(db_s_acc), "stats": stats}
    out_path = os.path.join(CHUNK_DIR, f"chunk_{chunk_id}.pkl")
    with open(out_path, "wb") as fh:
        pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nChunk {chunk_id} done in {time.time()-t_start:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()
