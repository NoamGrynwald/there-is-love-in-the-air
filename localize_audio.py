"""
main.py — DOA Estimation applied to a DataFrame of multi-channel audio files
=============================================================================
Each row in the DataFrame has an `audio_path` column pointing to a multi-channel
WAV file. The script trains a DOA model (once) and adds estimated angle columns.

Output columns added to the DataFrame:
    doa_angle_deg   — best model's estimated angle (integer, -90…+90)
    doa_rf_deg      — Random Forest estimate
    doa_mlp_deg     — MLP Neural Network estimate
    doa_model_used  — which model was selected as best
    doa_n_channels  — number of channels found in the file
    doa_error       — error message if estimation failed, else None
"""

import os
import warnings
import pickle
import numpy as np
import pandas as pd
import soundfile as sf
import scipy.signal as signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  — adjust to match your array hardware
# ─────────────────────────────────────────────────────────────────────────────
SPEED_OF_SOUND  = 343.0    # m/s
SAMPLE_RATE     = 16000    # Hz  (files are resampled to this if needed)
N_MICS          = 2        # channels expected in each WAV
MIC_SPACING     = 0.05     # metres between adjacent mics (ULA)
MODEL_PATH      = "doa_models.pkl"   # cached model; retrained if missing
N_TRAIN_PER_ANG = 30       # synthetic training samples per angle (increase for better accuracy)
ANGLES          = np.arange(-90, 91, 1)   # 181 classes


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL SIMULATION  (used only for training)
# ─────────────────────────────────────────────────────────────────────────────

def _simulate_ula(angle_deg, snr_db=20, duration=0.5):
    n = int(duration * SAMPLE_RATE)
    theta = np.deg2rad(angle_deg)
    src = np.random.randn(n)
    b, a = signal.butter(4, [300 / (SAMPLE_RATE / 2), 3000 / (SAMPLE_RATE / 2)], btype="band")
    src = signal.lfilter(b, a, src)
    delays = [m * MIC_SPACING * np.sin(theta) / SPEED_OF_SOUND * SAMPLE_RATE for m in range(N_MICS)]
    multi = np.zeros((N_MICS, n))
    for m, d in enumerate(delays):
        id_, fd = int(np.floor(d)), d - int(np.floor(d))
        s = np.roll(src, id_).astype(float)
        if fd and id_ + 1 < n:
            s = (1 - fd) * s + fd * np.roll(src, id_ + 1).astype(float)
        multi[m] = s
    noise_pow = np.mean(src ** 2) / (10 ** (snr_db / 10))
    multi += np.random.randn(N_MICS, n) * np.sqrt(noise_pow)
    return multi


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _gcc_phat(s1, s2, max_delay):
    nfft = 2 ** int(np.ceil(np.log2(len(s1) + len(s2) - 1)))
    S1, S2 = np.fft.rfft(s1, nfft), np.fft.rfft(s2, nfft)
    R = S1 * np.conj(S2)
    denom = np.abs(R); denom[denom < 1e-10] = 1e-10
    cc = np.fft.irfft(R / denom, nfft)
    cc = np.concatenate([cc[-(nfft // 2):], cc[:nfft // 2]])
    c  = nfft // 2
    return cc[c - max_delay: c + max_delay + 1]


def extract_features(multi):
    """multi: (N_MICS, n_samples) float array → 1-D feature vector"""
    max_d = int(np.ceil((N_MICS - 1) * MIC_SPACING / SPEED_OF_SOUND * SAMPLE_RATE)) + 5
    feats = []

    # GCC-PHAT for every mic pair
    for i in range(N_MICS):
        for j in range(i + 1, N_MICS):
            feats.extend(_gcc_phat(multi[i], multi[j], max_d).tolist())

    # Inter-channel phase & magnitude (IPD / ILD)
    nfft = 512
    fb   = slice(5, nfft // 2)
    specs = [np.fft.rfft(multi[m], nfft) for m in range(N_MICS)]
    for i in range(N_MICS):
        for j in range(i + 1, N_MICS):
            ipd = np.angle(specs[i][fb] * np.conj(specs[j][fb]))
            feats += [np.mean(np.cos(ipd)), np.mean(np.sin(ipd)),
                      np.std(np.cos(ipd)),  np.std(np.sin(ipd))]
            ild = np.log((np.abs(specs[i][fb]) + 1e-10) / (np.abs(specs[j][fb]) + 1e-10))
            feats += [np.mean(ild), np.std(ild)]

    return np.array(feats, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def _train_and_save():
    print(f"Training DOA models on synthetic data ({N_TRAIN_PER_ANG} samples/angle) …")
    rng = np.random.default_rng(42)
    X, y = [], []
    for angle in ANGLES:
        for _ in range(N_TRAIN_PER_ANG):
            snr   = rng.uniform(5, 30)
            multi = _simulate_ula(angle, snr_db=snr)
            X.append(extract_features(multi))
            y.append(angle)
    X, y = np.array(X), np.array(y)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.15,
                                               random_state=42, stratify=y)
    scaler   = StandardScaler()
    X_tr_s   = scaler.fit_transform(X_tr)
    X_te_s   = scaler.transform(X_te)

    print("  Fitting Random Forest …")
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    rf.fit(X_tr, y_tr)
    rf_mae = mean_absolute_error(y_te, rf.predict(X_te))
    print(f"    RF  MAE = {rf_mae:.2f}°")

    print("  Fitting MLP …")
    mlp = MLPClassifier(hidden_layer_sizes=(512, 256, 128), max_iter=300,
                        early_stopping=True, random_state=42)
    mlp.fit(X_tr_s, y_tr)
    mlp_mae = mean_absolute_error(y_te, mlp.predict(X_te_s))
    print(f"    MLP MAE = {mlp_mae:.2f}°")

    best = "random_forest" if rf_mae <= mlp_mae else "mlp"
    print(f"  ✔  Best model: {best}")

    bundle = dict(scaler=scaler, rf=rf, mlp=mlp,
                  best=best, rf_mae=rf_mae, mlp_mae=mlp_mae)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"  Saved → {MODEL_PATH}\n")
    return bundle


def _load_or_train():
    if os.path.exists(MODEL_PATH):
        print(f"Loading cached models from {MODEL_PATH} …")
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return _train_and_save()


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-FILE INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def _resample(sig, from_sr, to_sr):
    from math import gcd
    g    = gcd(int(from_sr), int(to_sr))
    up   = to_sr // g
    down = from_sr // g
    return signal.resample_poly(sig, up, down)


def estimate_doa_file(audio_path, bundle):
    """
    Load a multi-channel WAV and return a dict with angle estimates.
    Returns error key on failure.
    """
    try:
        data, sr = sf.read(audio_path, always_2d=True)   # (n_samples, n_ch)
        n_ch = data.shape[1]

        if n_ch < N_MICS:
            return dict(doa_angle_deg=None, doa_rf_deg=None, doa_mlp_deg=None,
                        doa_model_used=None, doa_n_channels=n_ch,
                        doa_error=f"Only {n_ch} channel(s), need {N_MICS}")

        # Use first N_MICS channels; resample if needed
        channels = data[:, :N_MICS].T   # (N_MICS, n_samples)
        if sr != SAMPLE_RATE:
            channels = np.array([_resample(channels[m], sr, SAMPLE_RATE)
                                  for m in range(N_MICS)])

        feat   = extract_features(channels)
        feat_s = bundle["scaler"].transform(feat.reshape(1, -1))

        rf_angle  = int(bundle["rf"].predict(feat.reshape(1, -1))[0])
        mlp_angle = int(bundle["mlp"].predict(feat_s)[0])
        best_angle = rf_angle if bundle["best"] == "random_forest" else mlp_angle

        return dict(
            doa_angle_deg  = best_angle,
            doa_rf_deg     = rf_angle,
            doa_mlp_deg    = mlp_angle,
            doa_model_used = bundle["best"],
            doa_n_channels = n_ch,
            doa_error      = None,
        )
    except Exception as e:
        return dict(doa_angle_deg=None, doa_rf_deg=None, doa_mlp_deg=None,
                    doa_model_used=None, doa_n_channels=None, doa_error=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# DATAFRAME PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def apply_doa(df: pd.DataFrame,
              audio_col: str = "audio_path",
              base_dir: str  = "") -> pd.DataFrame:
    """
    Add DOA estimate columns to df.

    Parameters
    ----------
    df        : DataFrame with at least an `audio_col` column
    audio_col : column name holding the path to each WAV file
    base_dir  : optional prefix prepended to every path (e.g. "/data/dataset/")

    Returns
    -------
    df with new columns:
        doa_angle_deg, doa_rf_deg, doa_mlp_deg,
        doa_model_used, doa_n_channels, doa_error
    """
    bundle = _load_or_train()

    result_cols = ["doa_angle_deg", "doa_rf_deg", "doa_mlp_deg",
                   "doa_model_used", "doa_n_channels", "doa_error"]
    results = {c: [] for c in result_cols}

    total = len(df)
    for idx, row in df.iterrows():
        path = os.path.join(base_dir, row[audio_col]) if base_dir else row[audio_col]
        res  = estimate_doa_file(path, bundle)
        for c in result_cols:
            results[c].append(res[c])

        # Progress
        done = df.index.get_loc(idx) + 1
        print(f"  [{done:>5}/{total}]  {os.path.basename(path):<45}  "
              f"→  {res['doa_angle_deg']:+4}°" if res["doa_angle_deg"] is not None
              else f"  [{done:>5}/{total}]  {os.path.basename(path):<45}  "
                   f"→  ERROR: {res['doa_error']}", flush=True)

    for c in result_cols:
        df[c] = results[c]

    ok    = df["doa_error"].isna().sum()
    fails = total - ok
    print(f"\nDone. {ok}/{total} succeeded"
          + (f", {fails} failed (see doa_error column)" if fails else "") + ".")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Apply DOA estimation to a CSV/DataFrame")
    parser.add_argument("--csv",       required=True,  help="Input CSV file path")
    parser.add_argument("--out",       default="doa_results.csv", help="Output CSV path")
    parser.add_argument("--audio-col", default="audio_path",      help="Column with WAV paths")
    parser.add_argument("--base-dir",  default="",                help="Base directory for audio paths")
    parser.add_argument("--retrain",   action="store_true",       help="Force model retraining")
    args = parser.parse_args()

    if args.retrain and os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
        print("Removed cached model — will retrain.")

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")

    df = apply_doa(df, audio_col=args.audio_col, base_dir=args.base_dir)

    df.to_csv(args.out, index=False)
    print(f"Results saved → {args.out}")

    # Quick summary
    valid = df["doa_angle_deg"].dropna()
    if len(valid):
        print(f"\nAngle distribution (n={len(valid)}):")
        print(f"  Mean : {valid.mean():.1f}°")
        print(f"  Std  : {valid.std():.1f}°")
        print(f"  Range: {valid.min():.0f}° … {valid.max():.0f}°")