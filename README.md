# EE200 Q3B — Audio Fingerprinting App

A small Shazam-style identifier wrapping the Q3A fingerprinting pipeline
(spectrogram → constellation peaks → pair-hashes → offset-histogram matching)
in an interactive Streamlit app with three modes: **Library**, **Identify**,
and **Batch**.

## How it's built

- `fingerprint.py` — the matching engine. Re-derives the same math as the
  Q3A notebook (STFT spectrogram, 2-D local-maximum peak picking, pair/single
  hash generation, offset-histogram voting) directly in NumPy, decoding audio
  with a single `ffmpeg` subprocess call instead of librosa/soundfile. This
  keeps the deployed app's dependencies tiny (`numpy`, `streamlit`,
  `matplotlib`, `pandas` + the `ffmpeg` binary) and avoids librosa's slower
  cold-start / native-build issues on free hosting tiers. It is mathematically
  equivalent to the notebook's librosa+scipy version.
- `app.py` — the Streamlit app (Library / Identify / Batch tabs).
- `build_db.py` / `build_db_chunk.py` / `merge_two.py` / `split_chunk.py` —
  the one-time indexing scripts used to fingerprint the 50-song library and
  produce `data/fingerprint_db_0.pkl` + `data/fingerprint_db_1.pkl` (the
  database is split into two halves so neither file exceeds GitHub's 100MB
  per-file limit; `app.py` loads and merges both at startup).
- `data/thumbnails/*.png` — small constellation-map thumbnails per song,
  shown in the Library tab.
- `samples/*.mp3` — five ~25s demo clips (cut from songs already in the
  library) for the "try a sample" buttons.

The confidence gate (`fingerprint.decide_match`) requires both a minimum
absolute offset-histogram score **and** a 4× margin over the runner-up
before declaring a match — calibrated against real measurements: true
matches scored in the hundreds-to-thousands even at 0 dB SNR, while
unrelated clips never exceeded ~10.

## Running locally

```bash
pip install -r requirements.txt
# ffmpeg must also be on PATH (e.g. `brew install ffmpeg` / `apt install ffmpeg`)
streamlit run app.py
```

The database is already built (`data/fingerprint_db_0.pkl` +
`data/fingerprint_db_1.pkl` ship in this repo), so the app works immediately
— no indexing step required.

## Rebuilding the database (only needed if you change the song library)

1. Drop the song library's `.mp3` files into `songs/` (filenames become the
   labels your identifier outputs — don't rename them).
2. Run the indexing scripts (they're split into chunks because indexing all
   50 songs at once is CPU/time heavy):
   ```bash
   python build_db_chunk.py 0 7 0      # repeat for whatever ranges you need
   python build_db_chunk.py 7 14 1
   ...
   ```
3. Merge the chunks and split the result back into two <100MB halves:
   ```bash
   python merge_two.py data/chunks/chunk_0.pkl data/chunks/chunk_1.pkl data/chunks/m_01.pkl
   # ...merge pairwise until you have one combined pickle...
   python split_chunk.py data/chunks/<combined>.pkl data/fingerprint_db
   # -> produces data/fingerprint_db_0.pkl and data/fingerprint_db_1.pkl
   ```
   (Or simplify this into a single script if you have more headroom than a
   45-second sandboxed shell — `build_db.py` does it in one pass.)

## Deploying to Streamlit Community Cloud

This needs your own GitHub + Streamlit accounts — it can't be done on your
behalf. Steps:

1. **Create a GitHub repo** and push this folder to it. The `songs/` folder
   is git-ignored (it's 377MB of raw audio that the deployed app doesn't
   need — only the precomputed `data/fingerprint_db_*.pkl` does). Everything
   else (`app.py`, `fingerprint.py`, `data/`, `samples/`, `requirements.txt`,
   `packages.txt`) should be committed.
   ```bash
   git init
   git add .
   git commit -m "EE200 Q3B audio fingerprinting app"
   git branch -M main
   git remote add origin https://github.com/<you>/<repo>.git
   git push -u origin main
   ```
2. Go to **share.streamlit.io** → sign in with GitHub → **New app**.
3. Pick your repo, branch `main`, main file path `app.py`.
4. Click **Deploy**. Streamlit Cloud will read `requirements.txt` (Python
   packages) and `packages.txt` (apt packages — installs `ffmpeg`)
   automatically. First boot takes ~1-2 minutes while it loads the ~150MB
   fingerprint database into memory; it's cached afterward
   (`@st.cache_resource`).
5. Once it's live, copy the app URL (`https://<something>.streamlit.app`)
   and the GitHub repo URL into your Q3 report PDF, alongside your Q3A
   write-up.

### Submission checklist (from the assignment appendix)

- [ ] Live app link in the report, tested for both **single-clip mode**
      (Identify tab) and **batch mode** (Batch tab → `results.csv`)
- [ ] Link to the source code (the GitHub repo)
- [ ] A zip of all the code submitted separately (this folder, zipped)
- [ ] `results.csv` columns are exactly `filename,prediction` and
      `prediction` is the matched filename **without extension**, or
      `none` if no candidate clears the confidence threshold — already
      implemented in the Batch tab.
