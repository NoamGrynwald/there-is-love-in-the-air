"""
Improved DoA estimation:
  • SRP-PHAT over all mic pairs (66 pairs from 12-mic benchmark2 array)
  • VAD-gated windowing — only speech-active frames
  • Frame-level SRP with embedding-weighted voting
  • Real 3-D mic geometry from position files
  • Circular error metric (handles full 360° range)
  • Rich plotting: spatial power maps, per-frame traces, embedding PCA
"""

import os, math, glob, warnings
import numpy as np
import pandas as pd
import soundfile as sf
import scipy.signal as sig
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import combinations
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor

warnings.filterwarnings("ignore")

# ─── config ──────────────────────────────────────────────────────────────────
SPEED_OF_SOUND = 343.0
TARGET_SR      = 16_000     # model was trained at 16 kHz
MODEL_PATH     = "model/model.safetensors"
DATA_ROOT      = "files/dev/dev"
ARRAY_TYPE     = "benchmark2"
OUTPUT_DIR     = "doa_results_improved"


ANGLE_STEP_DEG = 1          # SRP grid resolution
WIN_SEC        = 1        # SRP window length (seconds at ORIG_SR)
HOP_SEC        = 0.25       # SRP window hop
MAX_WINS       = 200         # cap windows per recording to limit runtime
MIN_ACTIVE_RATIO = 0.30     # skip window if less than this fraction is VAD-active
FREQ_BANDS     = [(300, 1500), (1500, 4000), (4000, 8000)]  # Hz, for sub-band voting

os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ─── model ───────────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.pe = None

    def _build(self, device, T):
        pos      = torch.arange(T, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device)
                             * (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(T, self.d_model, device=device)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        B, T, D = x.shape
        if self.pe is None or self.pe.shape[1] < T:
            self.pe = self._build(x.device, T)
        return x + self.pe[:, :T]


class SpatialTransformerForSSL(nn.Module):
    def __init__(self, projection_dim=64, num_layers=1, num_heads=2, dim_feedforward=128):
        super().__init__()
        self.frontend = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=31, stride=8,  padding=15),
            nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=9,  stride=2, padding=4),
            nn.ReLU(),
        )
        self.spatial_mix   = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=1),
        )
        self.temporal_down = nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1)
        self.pos_encoder   = PositionalEncoding(64)
        self.transformer   = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=num_heads,
                                       dim_feedforward=dim_feedforward,
                                       batch_first=True, bias=False),
            num_layers=num_layers,
        )
        self.norm = nn.LayerNorm(64)
        self.proj = nn.Linear(64, projection_dim)

    def embed(self, input_values):
        """(B, 2, T) → (B, proj_dim) mean-pooled normalised embedding."""
        x = self.frontend(input_values)
        x = self.spatial_mix(x)
        x = self.temporal_down(x)
        x = x.permute(0, 2, 1)
        x = self.pos_encoder(x)
        h = self.transformer(x)
        h = self.norm(h)
        z = F.normalize(self.proj(h), dim=-1)
        return z.mean(dim=1)


def load_model(path):
    model = SpatialTransformerForSSL().to(DEVICE)
    missing, unexpected = model.load_state_dict(load_file(path), strict=False)
    if missing:
        print(f"[WARN] Model keys missing from checkpoint: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys in checkpoint (ignored): {unexpected}")
    model.eval()
    return model


# ─── geometry helpers ────────────────────────────────────────────────────────
def read_mic_positions(arr_pos_file):
    """Return (N_mics, 3) float array of mic XYZ positions."""
    df = pd.read_csv(arr_pos_file, sep="\t")
    row = df.iloc[0]
    mics = []
    for i in range(1, 13):
        mics.append([row[f"mic{i}_x"], row[f"mic{i}_y"], row[f"mic{i}_z"]])
    return np.array(mics, dtype=np.float64)  # (12, 3)


def compute_gt_azimuth_from_files(arr_pos_file, src_pos_file,
                                   vad_mask=None, n_samples=None):
    arr = pd.read_csv(arr_pos_file, sep="\t").iloc[0]
    src = pd.read_csv(src_pos_file, sep="\t")
    dx = src["x"].values - arr["x"]
    dy = src["y"].values - arr["y"]
    azimuths = np.arctan2(dy, dx)           # radians, one per position frame

    if vad_mask is not None and n_samples is not None:
        # Map each position frame to its corresponding VAD window and weight by
        # the fraction of that window that is speech-active.
        N_pos = len(azimuths)
        weights = np.array([
            vad_mask[int(i * n_samples / N_pos): int((i + 1) * n_samples / N_pos)].mean()
            for i in range(N_pos)
        ])
        if weights.sum() < 1e-8:
            weights = np.ones(N_pos)        # fallback: no VAD info
    else:
        weights = np.ones(len(azimuths))

    sin_mean = np.average(np.sin(azimuths), weights=weights)
    cos_mean = np.average(np.cos(azimuths), weights=weights)
    mean_az  = float(np.degrees(np.arctan2(sin_mean, cos_mean)))

    # Circular std of the trajectory over speech-active position frames
    active_az = np.degrees(azimuths[weights > 0]) if (weights > 0).any() else np.degrees(azimuths)
    r = float(np.abs(np.mean(np.exp(1j * np.deg2rad(active_az)))))
    r = min(r, 1.0 - 1e-9)                          # prevent sqrt(negative) when r≈1
    gt_circ_std = float(np.degrees(np.sqrt(-2.0 * np.log(max(r, 1e-10)))))

    # Also return raw trajectory (radians) so callers can compute per-frame GT
    return mean_az, gt_circ_std, azimuths


def circular_error(pred, gt):
    """Absolute angular error in degrees, clamped to [0, 180]."""
    diff = abs(pred - gt) % 360
    return min(diff, 360 - diff)


# ─── resampling ──────────────────────────────────────────────────────────────
def resample(audio, from_sr, to_sr):
    if from_sr == to_sr:
        return audio
    g = math.gcd(int(from_sr), int(to_sr))
    return sig.resample_poly(audio, to_sr // g, from_sr // g)


# ─── SRP-PHAT (vectorised, full 360°) ────────────────────────────────────────
def srp_phat(channels, mic_positions, fs,
             angle_grid_rad, speed_of_sound=343.0, nfft=4096, band_mask=None):
    """
    Steered Response Power with PHAT weighting.
    Aggregates GCC-PHAT across ALL mic pairs (two channels at a time).

    Parameters
    ----------
    channels     : (N_mics, T) float — time-domain multichannel audio
    mic_positions: (N_mics, 3) float — XYZ mic positions in metres
    fs           : sample rate (Hz)
    angle_grid_rad: (A,) azimuth candidates in radians
    speed_of_sound: m/s
    band_mask    : optional (F,) bool — restrict to these frequency bins

    Returns
    -------
    power : (A,) spatial power spectrum (higher = more likely DoA)
    """
    N_mics = channels.shape[0]
    freqs  = np.fft.rfftfreq(nfft, d=1.0 / fs)          # (F,)
    Specs  = np.fft.rfft(channels, n=nfft, axis=1)       # (N_mics, F)

    # Precompute all cross-spectra with PHAT weighting
    pairs       = list(combinations(range(N_mics), 2))
    cross_specs = np.zeros((len(pairs), len(freqs)), dtype=np.complex128)
    for k, (i, j) in enumerate(pairs):
        R = Specs[i] * np.conj(Specs[j])
        denom = np.abs(R); denom[denom < 1e-10] = 1e-10
        cs = R / denom                                    # (F,)
        if band_mask is not None:
            cs = cs * band_mask
        cross_specs[k] = cs

    # Direction unit vectors for every candidate angle (XY plane only)
    d_vecs = np.stack([np.cos(angle_grid_rad),
                       np.sin(angle_grid_rad),
                       np.zeros_like(angle_grid_rad)], axis=1)  # (A, 3)

    # Expected TDOA for each pair × angle: (n_pairs, A)
    delta_pos = mic_positions[np.array([p[0] for p in pairs])] \
              - mic_positions[np.array([p[1] for p in pairs])]  # (n_pairs, 3)
    expected_tau = (delta_pos @ d_vecs.T) / speed_of_sound      # (n_pairs, A)

    # Steering phase: (n_pairs, A, F) = exp(-j 2π f τ)
    # Use einsum to avoid materialising the full (n_pairs, A, F) array at once
    power = np.zeros(len(angle_grid_rad))
    for k in range(len(pairs)):
        # (A, F)
        phase = np.exp(-1j * 2 * np.pi
                       * freqs[None, :] * expected_tau[k, :, None])
        power += np.real(phase @ cross_specs[k])          # (A,)

    return power / len(pairs)


def smooth_power(power, sigma_deg=3, angle_step_deg=1):
    """Apply Gaussian smoothing to circular SRP power map."""
    sigma_bins = sigma_deg / angle_step_deg
    # Circular smoothing by padding
    pad = int(4 * sigma_bins)
    padded = np.concatenate([power[-pad:], power, power[:pad]])
    smoothed = gaussian_filter1d(padded, sigma=sigma_bins)
    return smoothed[pad: pad + len(power)]


def circular_median_deg(angles_deg):
    """Angle (degrees) that minimises the sum of circular distances to all inputs."""
    a = np.deg2rad(np.asarray(angles_deg, dtype=float))
    costs = [np.abs(np.angle(np.exp(1j * (a - c)))).sum() for c in a]
    return float(np.degrees(a[int(np.argmin(costs))]))


def circular_std_deg(angles_deg):
    """Mardia's circular standard deviation in degrees."""
    r = np.abs(np.mean(np.exp(1j * np.deg2rad(np.asarray(angles_deg, dtype=float)))))
    r = min(float(r), 1.0 - 1e-9)   # prevent sqrt(negative) when all angles agree exactly
    return float(np.degrees(np.sqrt(-2.0 * np.log(max(r, 1e-10)))))


# ─── VAD helpers ─────────────────────────────────────────────────────────────
def load_vad(vad_path, n_samples):
    """Return (mask, n_covered): boolean mask at original SR and how many samples it covers."""
    vad = pd.read_csv(vad_path)["VAD"].values.astype(bool)
    n = min(len(vad), n_samples)
    mask = np.zeros(n_samples, dtype=bool)
    mask[:n] = vad[:n]
    return mask, n


# ─── model embedding for all mic pairs in one batched forward pass ───────────
@torch.no_grad()
def embed_all_pairs(model, win_channels, from_sr, to_sr):
    """Resample all channels once, embed every mic pair as a single batch,
    return the mean 64-d embedding vector across all pairs."""
    resampled = [resample(win_channels[i], from_sr, to_sr).astype(np.float32)
                 for i in range(win_channels.shape[0])]
    n = min(len(r) for r in resampled)
    pairs = list(combinations(range(len(resampled)), 2))
    batch = np.stack([np.stack([resampled[i][:n], resampled[j][:n]])
                      for i, j in pairs])                      # (n_pairs, 2, T)
    tensor = torch.tensor(batch, dtype=torch.float32).to(DEVICE)
    embs = model.embed(tensor)                                 # (n_pairs, 64)
    return embs.mean(dim=0).cpu().numpy()


# ─── per-recording pipeline ───────────────────────────────────────────────────
def process_recording(wav_path, arr_pos_file, src_pos_file, model):
    """
    Returns a dict with GCC-PHAT / SRP-PHAT estimates and diagnostics,
    or raises an exception on error.
    """
    # ── load audio ──────────────────────────────────────────────────────────
    audio, sr = sf.read(wav_path, always_2d=True)   # (T, 12)
    audio = audio.T.astype(np.float32)              # (12, T)
    n_samples = audio.shape[1]

    # ── load VAD ────────────────────────────────────────────────────────────
    vad_name = os.path.basename(src_pos_file).replace("position_source_", "VAD_" + ARRAY_TYPE + "_")
    vad_path = os.path.join(os.path.dirname(wav_path), vad_name)
    if not os.path.exists(vad_path):
        vad_mask = np.ones(n_samples, dtype=bool)
        vad_n    = n_samples
    else:
        vad_mask, vad_n = load_vad(vad_path, n_samples)

    # ── mic geometry ────────────────────────────────────────────────────────
    mic_pos = read_mic_positions(arr_pos_file)       # (12, 3)

    # ── ground truth ────────────────────────────────────────────────────────
    gt_az, gt_circ_std, gt_trajectory = compute_gt_azimuth_from_files(
        arr_pos_file, src_pos_file, vad_mask, n_samples
    )
    # gt_trajectory is (N_pos,) radians; used to assign per-window GT below.
    # Steering assumes sources are in the horizontal plane (Z component of the
    # steering vector is zero), which is a standard approximation for azimuth-only DoA.

    # ── angle grid ──────────────────────────────────────────────────────────
    angle_grid_deg = np.arange(-180, 180, ANGLE_STEP_DEG)
    angle_grid_rad = np.deg2rad(angle_grid_deg)

    # ── build VAD-aligned windows ────────────────────────────────────────────
    win_n  = int(WIN_SEC * sr)
    hop_n  = int(HOP_SEC * sr)
    starts = list(range(0, n_samples - win_n + 1, hop_n))

    active_starts = [s for s in starts
                     if vad_mask[s: s + win_n].mean() >= MIN_ACTIVE_RATIO]
    if len(active_starts) > MAX_WINS:
        idx = np.linspace(0, len(active_starts) - 1, MAX_WINS, dtype=int)
        active_starts = [active_starts[i] for i in idx]

    if len(active_starts) == 0:
        raise ValueError("No VAD-active windows found")

    # Per-window GT: interpolate position trajectory to each window's midpoint.
    N_pos = len(gt_trajectory)
    frame_gts = [
        float(np.degrees(gt_trajectory[min(int((s + win_n // 2) / n_samples * N_pos), N_pos - 1)]))
        for s in active_starts
    ]

    # ── frame-level SRP-PHAT ────────────────────────────────────────────────
    srp_accumulator  = np.zeros(len(angle_grid_deg))
    frame_angles     = []
    frame_embeddings = []
    frame_weights    = []   # sub-band agreement confidence
    srp_acc_weights  = []   # full-band peak prominence (for accumulator only)

    # _freqs is constant across all windows — compute once
    _freqs = np.fft.rfftfreq(4096, d=1.0 / sr)

    for s in active_starts:
        win = audio[:2, s: s + win_n]                # (2, win_n) — 2-mic like a phone

        # Sub-band SRP-PHAT: 3 bands → circular median → frame angle
        _band_peaks = []
        for _f_lo, _f_hi in FREQ_BANDS:
            _band_mask = (_freqs >= _f_lo) & (_freqs <= _f_hi)
            if _band_mask.sum() < 4:
                continue
            _pwr = srp_phat(win, mic_pos[:2], sr, angle_grid_rad,
                            SPEED_OF_SOUND, band_mask=_band_mask)
            _pwr = smooth_power(_pwr, sigma_deg=3, angle_step_deg=ANGLE_STEP_DEG)
            _band_peaks.append(angle_grid_deg[int(np.argmax(_pwr))])

        frame_angle = circular_median_deg(_band_peaks) if _band_peaks else 0.0
        frame_angles.append(frame_angle)

        # Confidence for frame_angle: agreement across bands (low spread = confident)
        if len(_band_peaks) >= 2:
            band_spread = circular_std_deg(_band_peaks)
            sub_weight  = 1.0 / (band_spread + 5.0)  # 5° floor prevents extreme values
        else:
            sub_weight = 1e-6
        frame_weights.append(sub_weight)

        # Full-band SRP: drives accumulator (independent of sub-band angles)
        power    = srp_phat(win, mic_pos[:2], sr, angle_grid_rad, SPEED_OF_SOUND)
        power    = smooth_power(power, sigma_deg=3, angle_step_deg=ANGLE_STEP_DEG)
        peak_idx = int(np.argmax(power))
        full_w   = max(power[peak_idx] - np.median(power), 1e-6)
        srp_acc_weights.append(full_w)
        srp_accumulator += power * full_w

        # Model embedding — 2-mic pair
        emb = embed_all_pairs(model, win, sr, TARGET_SR)
        frame_embeddings.append(emb)

    frame_angles     = np.array(frame_angles)
    frame_weights    = np.array(frame_weights)
    frame_embeddings = np.array(frame_embeddings)   # (n_wins, 64)
    frame_gts        = np.array(frame_gts)

    # ── aggregate angle estimate ─────────────────────────────────────────────
    # Peak of the time-averaged SRP map (full-band, confidence-weighted)
    srp_global_deg = angle_grid_deg[int(np.argmax(srp_accumulator))]

    errors = {"srp_global": circular_error(srp_global_deg, gt_az)}

    # Standard FWHM: width at half the peak value (not half the range).
    _smooth_acc  = smooth_power(srp_accumulator, sigma_deg=3, angle_step_deg=ANGLE_STEP_DEG)
    _half_max    = 0.5 * _smooth_acc.max()
    srp_peak_width = int((_smooth_acc >= _half_max).sum())

    return dict(
        gt_az=gt_az,
        gt_circ_std=gt_circ_std,
        srp_global=srp_global_deg,
        srp_circ_std=circular_std_deg(frame_angles),
        srp_peak_width=srp_peak_width,
        errors=errors,
        n_windows=len(active_starts),
        vad_active_pct=float(vad_mask[:vad_n].mean() * 100),
        srp_accumulator=srp_accumulator,
        angle_grid_deg=angle_grid_deg,
        frame_angles=frame_angles,
        frame_weights=frame_weights,
        frame_gts=frame_gts,
        frame_embeddings=frame_embeddings,
    )


# ─── collect all recordings ───────────────────────────────────────────────────
def collect_recordings(root, array_type):
    records = []
    for task_dir in sorted(glob.glob(os.path.join(root, "task*"))):
        task = os.path.basename(task_dir)
        for rec_dir in sorted(glob.glob(os.path.join(task_dir, "recording*"))):
            rec     = os.path.basename(rec_dir)
            arr_dir = os.path.join(rec_dir, array_type)
            if not os.path.isdir(arr_dir):
                continue
            wav      = os.path.join(arr_dir, f"audio_array_{array_type}.wav")
            arr_pos  = os.path.join(arr_dir, f"position_array_{array_type}.txt")
            if not os.path.exists(wav) or not os.path.exists(arr_pos):
                continue
            for spf in sorted(glob.glob(os.path.join(arr_dir, "position_source_*.txt"))):
                records.append(dict(task=task, rec=rec, wav=wav,
                                    arr_pos=arr_pos, src_pos=spf))
    return records


# ─── main ─────────────────────────────────────────────────────────────────────
def run():
    model   = load_model(MODEL_PATH)
    records = collect_recordings(DATA_ROOT, ARRAY_TYPE)
    print(f"Found {len(records)} recording(s).\n")

    rows             = []
    srp_samples      = []   # one (label, res) per task for spatial power map plots
    srp_tasks_seen   = set()

    for r in records:
        label = f"{r['task']}/{r['rec']}/{os.path.basename(r['src_pos']).replace('position_source_','').replace('.txt','')}"
        try:
            res = process_recording(r["wav"], r["arr_pos"], r["src_pos"], model)
            rows.append(dict(
                task          = r["task"],
                recording     = r["rec"],
                gt_az         = res["gt_az"],
                srp_global    = res["srp_global"],
                srp_circ_std  = res["srp_circ_std"],
                srp_peak_width= res["srp_peak_width"],
                n_windows     = res["n_windows"],
                vad_active    = res["vad_active_pct"],
                err_srp       = res["errors"]["srp_global"],
                embeddings    = res["frame_embeddings"],
                frame_gts     = res["frame_gts"],
            ))
            if r["task"] not in srp_tasks_seen:
                srp_samples.append((label, res))
                srp_tasks_seen.add(r["task"])
            print(f"  {label:<55}  GT={res['gt_az']:+7.1f}° (±{res['gt_circ_std']:.1f}°)  "
                  f"SRP={res['srp_global']:+7.1f}° (err {res['errors']['srp_global']:.1f}°)  "
                  f"pw={res['srp_peak_width']}°  wins={res['n_windows']}")
        except Exception as e:
            print(f"  ERROR  {label}: {e}")

    if not rows:
        print("No results.")
        return

    _skip = {"embeddings", "frame_gts"}
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in _skip} for r in rows])

    # ── LOO KNN EMBEDDING REGRESSION ─────────────────────────────────────────
    # Ridge failed because it fits a global hyperplane across a curved manifold.
    # KNN is local — each test frame finds its nearest training frames and takes
    # their circular mean, naturally following the arc topology.
    # k=7 balances smoothing vs. resolution; distance-weighted to prefer closer
    # neighbours on the arc.
    # LOO KNN: embeddings are already L2-normalised by the model, so cosine
    # distance is used directly without StandardScaler (which would destroy the
    # normalisation). Per-frame GTs are used as targets so moving sources get
    # the correct instantaneous position for each window, not a recording-level mean.
    print("\nFitting LOO KNN embedding regressors …")
    _all_embs      = [r["embeddings"]  for r in rows]
    _all_gts       = [r["gt_az"]       for r in rows]
    _all_frame_gts = [r["frame_gts"]   for r in rows]
    emb_reg_angles = []
    for i in range(len(rows)):
        tr = [j for j in range(len(rows)) if j != i]
        X_tr = np.vstack([_all_embs[j] for j in tr])
        # Per-frame [cos, sin] targets use the instantaneous GT for each window
        y_tr = np.vstack([
            np.column_stack([
                np.cos(np.deg2rad(_all_frame_gts[j])),
                np.sin(np.deg2rad(_all_frame_gts[j])),
            ]) for j in tr
        ])
        knn = KNeighborsRegressor(n_neighbors=7, weights="distance", metric="cosine")
        knn.fit(X_tr, y_tr)
        pred_cs = knn.predict(_all_embs[i]).mean(axis=0)
        emb_reg_angles.append(float(np.degrees(np.arctan2(pred_cs[1], pred_cs[0]))))

    df["emb_reg"]     = emb_reg_angles
    df["err_emb_reg"] = [circular_error(p, g)
                         for p, g in zip(emb_reg_angles, _all_gts)]

    # ── PLOTS ────────────────────────────────────────────────────────────────

    # 1. Spatial power maps
    n = len(srp_samples)
    ncols = min(4, n); nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.5 * nrows),
                             subplot_kw=dict(projection="polar"))
    axes = np.array(axes).flatten()
    for ax, (label, res) in zip(axes, srp_samples):
        th  = np.deg2rad(res["angle_grid_deg"])
        pwr = res["srp_accumulator"]
        pwr_norm = (pwr - pwr.min()) / (pwr.max() - pwr.min() + 1e-8)
        ax.plot(th, pwr_norm, linewidth=0.8, color="steelblue")
        ax.fill(th, pwr_norm, alpha=0.25, color="steelblue")
        ax.axvline(np.deg2rad(res["gt_az"]),  color="green",  linewidth=2, label="GT")
        ax.axvline(np.deg2rad(res["srp_global"]), color="crimson", linewidth=2, linestyle="--", label="SRP")
        ax.set_title(label, fontsize=7, pad=4)
        ax.legend(fontsize=6, loc="upper right")
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("SRP-PHAT Spatial Power Maps (all mic pairs)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "srp_polar_maps.png"), dpi=150)
    plt.close(fig)
    print("\nSaved: srp_polar_maps.png")

    # 2. Per-frame angle traces
    fig, axes = plt.subplots(min(4, n), 1,
                             figsize=(10, 3.5 * min(4, n)), sharex=False)
    if min(4, n) == 1:
        axes = [axes]
    for ax, (label, res) in zip(axes, srp_samples[:4]):
        wins = np.arange(len(res["frame_angles"]))
        ax.scatter(wins, res["frame_angles"], s=30 + 200 * res["frame_weights"] /
                   (res["frame_weights"].max() + 1e-8),
                   c=res["frame_angles"], cmap="hsv",
                   vmin=-180, vmax=180, edgecolors="k", lw=0.3, zorder=3)
        ax.axhline(res["gt_az"],   color="green",  linewidth=1.5, linestyle="--", label=f"GT {res['gt_az']:+.1f}°")
        ax.axhline(res["srp_global"], color="crimson", linewidth=1.5, linestyle=":", label=f"SRP {res['srp_global']:+.1f}°")
        ax.set_ylim(-180, 180)
        ax.set_ylabel("Azimuth (°)")
        ax.set_title(label, fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Window index")
    fig.suptitle("Per-Frame SRP-PHAT Angle Estimates (point size ∝ confidence)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "frame_angle_traces.png"), dpi=150)
    plt.close(fig)
    print("Saved: frame_angle_traces.png")

    # 3. Pred vs GT — two methods
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (col, ecol, label) in zip(axes, [
        ("srp_global", "err_srp",     "SRP-PHAT (no calibration)"),
        ("emb_reg",    "err_emb_reg", "KNN embedding regression (calibrated)"),
    ]):
        sc = ax.scatter(df["gt_az"], df[col], c=df[ecol],
                        cmap="RdYlGn_r", vmin=0, vmax=90,
                        s=60, alpha=0.85, edgecolors="k", lw=0.3)
        lo, hi = -185, 185
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("GT Azimuth (°)"); ax.set_ylabel("Predicted (°)")
        ax.set_title(label)
        plt.colorbar(sc, ax=ax, label="Error (°)")
    fig.suptitle("DoA Predicted vs Ground Truth", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "pred_vs_gt_methods.png"), dpi=150)
    plt.close(fig)
    print("Saved: pred_vs_gt_methods.png")

    # 4. Error histograms
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (ecol, label, color) in zip(axes, [
        ("err_srp",     "SRP-PHAT",             "steelblue"),
        ("err_emb_reg", "KNN embedding (LOO)",  "seagreen"),
    ]):
        ax.hist(df[ecol], bins=18, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(df[ecol].mean(),   color="k",       lw=1.5, ls="--",
                   label=f"Mean {df[ecol].mean():.1f}°")
        ax.axvline(df[ecol].median(), color="dimgray", lw=1.5, ls=":",
                   label=f"Median {df[ecol].median():.1f}°")
        ax.set_xlabel("Abs. Error (°)"); ax.set_ylabel("Count")
        ax.set_title(label); ax.legend(fontsize=8)
    fig.suptitle("Circular Error Distribution by Method", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "error_histograms.png"), dpi=150)
    plt.close(fig)
    print("Saved: error_histograms.png")

    # 5. Per-task error bar
    method_cols = [("err_srp",     "SRP",    "steelblue"),
                   ("err_emb_reg", "KNN",    "seagreen")]
    tasks = sorted(df["task"].unique())
    x = np.arange(len(tasks))
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.35
    for k, (col, mlabel, color) in enumerate(method_cols):
        means = [df[df.task == t][col].mean() for t in tasks]
        stds  = [df[df.task == t][col].std()  for t in tasks]
        ax.bar(x + k * width, means, width, yerr=stds, capsize=4,
               label=mlabel, color=color, alpha=0.85, edgecolor="k")
    ax.set_xticks(x + width / 2); ax.set_xticklabels(tasks)
    ax.set_ylabel("Mean Abs. Error (°)"); ax.set_xlabel("Task")
    ax.set_title("DoA Error by Task & Method", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "error_by_task.png"), dpi=150)
    plt.close(fig)
    print("Saved: error_by_task.png")

    # 6. Embedding PCA coloured by GT azimuth
    all_embs = np.vstack([r["embeddings"] for r in rows])
    all_az   = np.concatenate([np.full(len(r["embeddings"]), r["gt_az"]) for r in rows])
    pca = PCA(n_components=2)
    e2  = pca.fit_transform(all_embs)
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(e2[:, 0], e2[:, 1], c=all_az, cmap="hsv",
                    vmin=-180, vmax=180, s=20, alpha=0.7, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="GT Azimuth (°)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("Frame-level Model Embeddings — PCA coloured by GT Azimuth",
                 fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "embedding_pca_frames.png"), dpi=150)
    plt.close(fig)
    print("Saved: embedding_pca_frames.png")

    # 7. Big summary (2×2 + PCA)
    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    ax00 = fig.add_subplot(gs[0, 0])
    sc00 = ax00.scatter(df["gt_az"], df["srp_global"], c=df["err_srp"],
                        cmap="RdYlGn_r", vmin=0, vmax=90, s=45, alpha=0.85,
                        edgecolors="k", lw=0.3)
    ax00.plot([-185, 185], [-185, 185], "k--", lw=1)
    ax00.set_xlabel("GT Azimuth (°)"); ax00.set_ylabel("Predicted (°)")
    ax00.set_title("SRP-PHAT: Pred vs GT")
    plt.colorbar(sc00, ax=ax00, label="Error (°)")

    ax01 = fig.add_subplot(gs[0, 1])
    sc01 = ax01.scatter(df["gt_az"], df["emb_reg"], c=df["err_emb_reg"],
                        cmap="RdYlGn_r", vmin=0, vmax=90, s=45, alpha=0.85,
                        edgecolors="k", lw=0.3)
    ax01.plot([-185, 185], [-185, 185], "k--", lw=1)
    ax01.set_xlabel("GT Azimuth (°)"); ax01.set_ylabel("Predicted (°)")
    ax01.set_title("KNN Embedding (LOO): Pred vs GT")
    plt.colorbar(sc01, ax=ax01, label="Error (°)")

    ax02 = fig.add_subplot(gs[0, 2])
    ax02.scatter(df["vad_active"], df["err_srp"], color="steelblue",
                 alpha=0.8, s=50, edgecolors="k", lw=0.3, label="SRP")
    ax02.scatter(df["vad_active"], df["err_emb_reg"], color="seagreen",
                 alpha=0.8, s=50, marker="^", edgecolors="k", lw=0.3, label="KNN")
    ax02.set_xlabel("VAD Active (%)"); ax02.set_ylabel("Abs. Error (°)")
    ax02.set_title("Error vs Speech Activity")
    ax02.legend(fontsize=8)

    for k, (ecol, label, color) in enumerate(
        [("err_srp",     "SRP-PHAT",        "steelblue"),
         ("err_emb_reg", "KNN Emb. (LOO)",  "seagreen")]):
        ax = fig.add_subplot(gs[1, k])
        ax.hist(df[ecol], bins=14, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(df[ecol].mean(), color="k", lw=1.5, ls="--",
                   label=f"μ={df[ecol].mean():.1f}°")
        ax.set_xlabel("Abs. Error (°)"); ax.set_title(label)
        ax.legend(fontsize=8)

    ax12 = fig.add_subplot(gs[1, 2])
    sc12 = ax12.scatter(e2[:, 0], e2[:, 1], c=all_az, cmap="hsv",
                        vmin=-180, vmax=180, s=10, alpha=0.6, edgecolors="none")
    plt.colorbar(sc12, ax=ax12, label="GT Az (°)")
    ax12.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax12.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax12.set_title("Embedding PCA")

    fig.suptitle("DoA  ·  2-mic  ·  SRP-PHAT (zero-data)  vs  KNN Embedding (calibrated)",
                 fontsize=13, fontweight="bold")
    fig.savefig(os.path.join(OUTPUT_DIR, "summary.png"), dpi=150)
    plt.close(fig)
    print("Saved: summary.png")

    # ── PRINT SUMMARY ────────────────────────────────────────────────────────
    all_err_cols = ["err_srp", "err_emb_reg"]
    best_col = min(all_err_cols, key=lambda c: df[c].mean())
    print("\n" + "═" * 68)
    print("  DoA RESULTS SUMMARY")
    print("═" * 68)
    print(f"  Recordings:   {len(df)}")
    print(f"  Array:        {ARRAY_TYPE}  (2 mics)")
    print(f"  SRP grid:     {ANGLE_STEP_DEG}° steps, ±180°  (full 360°)")
    print(f"  Window:       {WIN_SEC}s / {HOP_SEC}s hop, ≤{MAX_WINS} wins/rec")
    print()
    print(f"  {'Method':<35} {'Mean err':>9} {'Median':>9} {'Std':>7} {'≤10°':>7} {'≤20°':>7}")
    print(f"  {'-'*35} {'-'*9} {'-'*9} {'-'*7} {'-'*7} {'-'*7}")
    for col, name in [
        ("err_srp",     "SRP-PHAT (zero-data)"),
        ("err_emb_reg", "KNN embedding (calibrated, LOO)"),
    ]:
        e = df[col]
        mark = "◀ best" if col == best_col else ""
        print(f"  {name:<35} {e.mean():>8.1f}° {e.median():>8.1f}° "
              f"{e.std():>6.1f}° {(e<=10).mean()*100:>6.1f}% {(e<=20).mean()*100:>6.1f}%  {mark}")
    print()
    print("  Per-task (best method):")
    for t in tasks:
        sub = df[df.task == t][best_col]
        print(f"    {t}:  mean={sub.mean():.1f}°  std={sub.std():.1f}°  n={len(sub)}")
    print("═" * 68)
    print(f"\nAll outputs → {os.path.abspath(OUTPUT_DIR)}/")

    df.to_csv(os.path.join(OUTPUT_DIR, "doa_improved_results.csv"), index=False)
    print(f"CSV → {OUTPUT_DIR}/doa_improved_results.csv")


if __name__ == "__main__":
    run()
