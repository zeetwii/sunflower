#!/usr/bin/env python3
"""
fly_navigator.py - feed video into a connectome-based fruit fly brain and
show which way the fly "wants" to go.

Pipeline
    video frame
      -> FlyEye       : downsample to fly-like resolution, compute optic flow,
                        extract features the fly's visual projection neurons
                        are known to care about (wide-field motion, looming,
                        small moving objects), separately for each eye
      -> Brain        : whole-brain leaky integrate-and-fire model
                        (Shiu et al. 2024, FlyWire v783, ~139k neurons, ~15M
                        connections). Features become Poisson drive onto
                        identified visual neuron types.
      -> Readout      : spike rates of identified descending neurons
                        (steering, escape, forward, backward)
      -> Display      : arrow + bars drawn over the video, optional
                        comparison against an autopilot log


IMPORTANT: the encoder (video -> neurons) and the decoder (neurons -> motion)
are hand-designed choices, not validated fly biology. The connectome in the
middle is real wiring; everything at the edges is yours to tune and question.

Quick start
    pip install numpy scipy pandas pyarrow opencv-python
    python fly_navigator.py --source drum          # optomotor sanity test
    python fly_navigator.py --source loom-left     # escape sanity test
    python fly_navigator.py --source 0             # webcam 0
    python fly_navigator.py --source flight.mp4 --compare autopilot.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Configuration: THE PART YOU WILL EDIT MOST
# ---------------------------------------------------------------------------

# Visual feature -> FlyWire cell types that receive Poisson drive for it.
# Each feature is computed per eye ("left"/"right"), and drives the cells of
# that type whose soma is on the same side.
FEATURE_TO_CELL_TYPES = {
    # Lobula plate horizontal-system cells: excited by front-to-back
    # ("progressive") wide-field motion on their own eye. Basis of the
    # optomotor turning response.
    "progressive_motion": ["HSN", "HSE", "HSS"],
    # Looming-sensitive visual projection neurons that feed the escape
    # (giant fiber) pathway.
    "looming": ["LC4", "LPLC2"],
    # Small-object / target motion detectors.
    "small_object": ["LC10a", "LPLC1", "LC11"],
}

# Max Poisson rate (Hz) per input neuron when a feature is at 1.0.
# 150 Hz is the default stimulation rate in the Shiu et al. model.
MAX_INPUT_RATE_HZ = 150.0

# Descending neuron readouts. Each entry is a list of FlyWire cell types; rates
# are averaged per side. These are literature-inspired guesses, not a
# validated motor decoder. Add/remove freely.
READOUT_CELL_TYPES = {
    # DNa01/DNa02 activity precedes turns toward the same side in walking flies
    "steer": ["DNa01", "DNa02"],
    # Giant fiber: escape takeoff (jump + wings)
    "escape": ["DNp01"],
    # P9 / DNp09: forward walking and object-directed turning
    "forward": ["DNp09"],
    # Moonwalker descending neurons: backward walking
    "backward": ["MDN"],
    # Flight steering is poorly mapped; add candidate DN types here to
    # experiment, e.g. "flight_steer": ["DNg02_a", ...]
}

READOUT_WINDOW_MS = 200.0   # rolling window for spike-rate estimates
STEER_DEADBAND_HZ = 2.0     # L-R difference below this counts as "straight"
STEER_FULL_SCALE_HZ = 15.0  # L-R difference drawn as a hard turn on screen
MOVE_FULL_SCALE_HZ = 15.0   # forward-minus-backward rate drawn as a full-length arrow
ESCAPE_THRESHOLD_HZ = 20.0  # giant fiber rate that counts as "escape"

# Degrees of visual angle per "ommatidium" after downsampling. Real flies are
# ~5 degrees. The frame is resized so that its width = camera_fov / this.
DEG_PER_PIXEL = 5.0

# Feature gains: raw feature values are multiplied by these, then the floor
# is subtracted (so weak, noisy signals don't leak into the brain) and the
# result rescaled/clipped to [0, 1]. Tune by watching the input bars.
FEATURE_FLOOR = 0.15
FEATURE_GAIN = {
    "progressive_motion": 1 / 40.0,   # raw units: deg/s of front-to-back motion
    "looming": 1 / 3.0,               # raw units: 1/s of flow expansion
    "small_object": 1 / 80.0,         # raw units: deg/s of non-background motion
}

# ---------------------------------------------------------------------------
# Data download + caching
# ---------------------------------------------------------------------------

DATA_URLS = {
    "Completeness_783.csv":
        "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/Completeness_783.csv",
    "Connectivity_783.parquet":
        "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/Connectivity_783.parquet",
    "neuron_annotations.tsv":
        "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/"
        "supplemental_files/Supplemental_file1_neuron_annotations.tsv",
}


def ensure_data(data_dir: Path) -> None:
    """Download the ~130 MB of connectome + annotation files once."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, url in DATA_URLS.items():
        path = data_dir / name
        if path.exists():
            continue
        print(f"Downloading {name} ...")
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)


def load_connectome(data_dir: Path, w_syn_mV: float):
    """Return (W, root_ids, annotations).

    W is a CSR matrix indexed [presynaptic, postsynaptic] holding the jump in
    the postsynaptic conductance variable g (mV) caused by one spike.
    """
    cache = data_dir / "connectome_cache.npz"
    root_ids = pd.read_csv(data_dir / "Completeness_783.csv", index_col=0).index.values
    n = len(root_ids)

    if cache.exists():
        z = np.load(cache)
        counts = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=(n, n))
    else:
        print("Building sparse connectivity matrix (first run only) ...")
        df = pd.read_parquet(
            data_dir / "Connectivity_783.parquet",
            columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"],
        )
        counts = sp.csr_matrix(
            (df["Excitatory x Connectivity"].values.astype(np.float32),
             (df["Presynaptic_Index"].values, df["Postsynaptic_Index"].values)),
            shape=(n, n),
        )
        counts.sum_duplicates()
        np.savez(cache, data=counts.data, indices=counts.indices, indptr=counts.indptr)

    W = (counts * np.float32(w_syn_mV)).tocsr()

    ann = pd.read_csv(
        data_dir / "neuron_annotations.tsv", sep="\t", low_memory=False,
        usecols=["root_id", "cell_type", "hemibrain_type", "side", "super_class"],
    )
    idx_of = pd.Series(np.arange(n), index=root_ids)
    ann = ann[ann["root_id"].isin(idx_of.index)].copy()
    ann["idx"] = idx_of.loc[ann["root_id"]].values
    return W, root_ids, ann


def indices_for(ann: pd.DataFrame, cell_types: list[str], side: str) -> np.ndarray:
    m = (ann["cell_type"].isin(cell_types) | ann["hemibrain_type"].isin(cell_types))
    m &= ann["side"] == side
    return ann.loc[m, "idx"].to_numpy()


# ---------------------------------------------------------------------------
# Brain: event-driven LIF, same equations/parameters as Shiu et al. 2024
# ---------------------------------------------------------------------------

@dataclass
class LIFParams:
    dt_ms: float = 0.1
    v_0: float = -52.0      # mV resting
    v_rst: float = -52.0    # mV reset
    v_th: float = -45.0     # mV threshold
    t_mbr: float = 20.0     # ms membrane time constant
    tau: float = 5.0        # ms synaptic time constant
    t_rfc: float = 2.2      # ms refractory
    t_dly: float = 1.8      # ms synaptic delay
    w_syn: float = 0.275    # mV per synapse
    f_poi: float = 250.0    # Poisson input strength multiplier


class Brain:
    """Whole-brain LIF network.

    dv/dt = (v_0 - v + g) / t_mbr
    dg/dt = -g / tau
    Integrated exactly each step. Spikes add W[pre, :] to g after t_dly.
    Delivery is event-driven (only rows of spiking neurons are touched), which
    keeps it usable on CPU because only a few hundred neurons spike at once.
    """

    def __init__(self, W: sp.csr_matrix, p: LIFParams, seed: int = 0):
        self.W, self.p = W, p
        self.n = W.shape[0]
        self.rng = np.random.default_rng(seed)
        dt = p.dt_ms

        # exact propagators for the linear (v, g) system
        self.a_v = np.exp(-dt / p.t_mbr)
        self.a_g = np.exp(-dt / p.tau)
        self.b_vg = p.tau / (p.tau - p.t_mbr) * (self.a_g - self.a_v)

        self.ref_steps = int(round(p.t_rfc / dt))
        self.delay_steps = max(1, int(round(p.t_dly / dt)))
        self.poisson_jump = p.f_poi * p.w_syn
        self.reset()

    def reset(self):
        p = self.p
        self.v = np.full(self.n, p.v_0, np.float32)
        self.g = np.zeros(self.n, np.float32)
        self.ref = np.zeros(self.n, np.int16)
        self.ring = np.zeros((self.delay_steps, self.n), np.float32)
        self.ring_pos = 0
        self.t_ms = 0.0
        self.drive_idx = np.zeros(0, np.int64)
        self.drive_p = np.zeros(0, np.float32)

    def set_drive(self, idx: np.ndarray, rates_hz: np.ndarray):
        """Poisson input: neuron idx[i] receives events at rates_hz[i]."""
        keep = rates_hz > 0
        self.drive_idx = idx[keep]
        self.drive_p = (rates_hz[keep] * self.p.dt_ms * 1e-3).astype(np.float32)

    def step(self) -> np.ndarray:
        p = self.p
        # delayed synaptic input arriving now
        self.g += self.ring[self.ring_pos]
        self.ring[self.ring_pos] = 0.0

        # Poisson drive
        if self.drive_idx.size:
            hit = self.rng.random(self.drive_idx.size) < self.drive_p
            np.add.at(self.g, self.drive_idx[hit], self.poisson_jump)

        # integrate non-refractory neurons
        active = self.ref <= 0
        u = self.v - p.v_0
        v_new = p.v_0 + u * self.a_v + self.g * self.b_vg
        self.v = np.where(active, v_new, self.v).astype(np.float32)
        self.g = np.where(active, self.g * self.a_g, self.g).astype(np.float32)
        self.ref[~active] -= 1

        # spikes
        spk = np.flatnonzero(self.v > p.v_th)
        if spk.size:
            self.v[spk] = p.v_rst
            self.g[spk] = 0.0
            self.ref[spk] = self.ref_steps
            rows = self.W[spk]
            out = np.bincount(rows.indices, weights=rows.data, minlength=self.n)
            # this slot was just consumed; it is read again delay_steps from now
            self.ring[self.ring_pos] += out.astype(np.float32)

        self.ring_pos = (self.ring_pos + 1) % self.delay_steps
        self.t_ms += p.dt_ms
        return spk

    def run(self, ms: float, watch: np.ndarray) -> np.ndarray:
        """Advance `ms` of brain time; return spike counts for `watch` idx."""
        counts = np.zeros(self.n, np.int32) if watch.size else None
        for _ in range(int(round(ms / self.p.dt_ms))):
            spk = self.step()
            if counts is not None and spk.size:
                counts[spk] += 1
        return counts[watch] if counts is not None else np.zeros(0, np.int32)


# ---------------------------------------------------------------------------
# Eye: frames -> per-eye visual features
# ---------------------------------------------------------------------------

class FlyEye:
    """Turns frames into per-eye feature strengths in [0, 1].

    Optic flow is computed at a modest working resolution (optical flow is
    unreliable on tiny images), converted to degrees of visual angle per
    second, and pooled per eye. `low_res` is the fly-resolution view (~5 deg
    per pixel) shown on screen as the "fly's eye".
    """
    FEATURES = list(FEATURE_TO_CELL_TYPES)
    WORK_WIDTH = 160

    def __init__(self, camera_fov_deg: float, fps: float):
        self.fov = camera_fov_deg
        self.fps = fps
        self.eye_w = max(16, int(round(camera_fov_deg / DEG_PER_PIXEL)))
        self.prev = None
        self.low_res = None

    def __call__(self, frame_bgr: np.ndarray) -> dict[str, dict[str, float]]:
        h0, w0 = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # fly-resolution view for display: blur ~ ommatidial acceptance angle
        eye_h = max(8, int(round(self.eye_w * h0 / w0)))
        sigma = 0.5 * w0 / self.eye_w
        self.low_res = cv2.resize(cv2.GaussianBlur(gray, (0, 0), sigma),
                                  (self.eye_w, eye_h), interpolation=cv2.INTER_AREA)

        ww = self.WORK_WIDTH
        wh = max(16, int(round(ww * h0 / w0)))
        work = cv2.resize(cv2.GaussianBlur(gray, (0, 0), 0.5 * w0 / ww),
                          (ww, wh), interpolation=cv2.INTER_AREA)

        feats = {f: {"left": 0.0, "right": 0.0} for f in self.FEATURES}
        if self.prev is None or self.prev.shape != work.shape:
            self.prev = work
            return feats

        flow = cv2.calcOpticalFlowFarneback(
            self.prev, work, None, pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        self.prev = work
        deg_per_px = self.fov / ww
        fx = flow[..., 0] * deg_per_px * self.fps   # deg/s
        fy = flow[..., 1] * deg_per_px * self.fps

        # divergence (1/s): expansion = something approaching
        div = (np.gradient(fx, axis=1) + np.gradient(fy, axis=0)) / deg_per_px
        # residual motion: local motion that differs from the wide-field
        resid = np.hypot(fx - np.median(fx), fy - np.median(fy))

        mid = ww // 2
        halves = {"left": slice(0, mid), "right": slice(mid, None)}
        for side, sl in halves.items():
            # front-to-back is leftward in the left half, rightward in right
            sign = -1.0 if side == "left" else 1.0
            raw = {
                "progressive_motion": max(0.0, float(np.mean(sign * fx[:, sl]))),
                "looming": max(0.0, float(np.percentile(div[:, sl], 95))),
                "small_object": float(np.percentile(resid[:, sl], 95)),
            }
            for f in self.FEATURES:
                x = raw[f] * FEATURE_GAIN[f]
                feats[f][side] = float(np.clip((x - FEATURE_FLOOR) / (1 - FEATURE_FLOOR), 0, 1))
        return feats


# ---------------------------------------------------------------------------
# Glue: features -> drive, spikes -> rates
# ---------------------------------------------------------------------------

@dataclass
class Readout:
    rates: dict = field(default_factory=dict)   # name -> {"left": Hz, "right": Hz}
    turn_hz: float = 0.0                        # right minus left steering rate
    command: str = "idle"


class FlyController:
    def __init__(self, data_dir: Path, dt_ms: float, seed: int):
        p = LIFParams(dt_ms=dt_ms)
        W, _, ann = load_connectome(data_dir, p.w_syn)
        self.brain = Brain(W, p, seed=seed)

        self.input_idx = {}
        for feat, types in FEATURE_TO_CELL_TYPES.items():
            for side in ("left", "right"):
                idx = indices_for(ann, types, side)
                if idx.size == 0:
                    print(f"warning: no neurons for {feat}/{side} ({types})")
                self.input_idx[(feat, side)] = idx

        self.readout_idx = {}
        watch = []
        for name, types in READOUT_CELL_TYPES.items():
            for side in ("left", "right"):
                idx = indices_for(ann, types, side)
                if idx.size == 0:
                    print(f"warning: no neurons for readout {name}/{side} ({types})")
                self.readout_idx[(name, side)] = (len(watch), len(watch) + idx.size)
                watch.extend(idx.tolist())
        self.watch = np.array(watch, np.int64)

        n_bins = max(1, int(READOUT_WINDOW_MS))
        self.history = collections.deque(maxlen=n_bins)  # (ms, counts)

        n_in = sum(v.size for v in self.input_idx.values())
        print(f"Brain ready: {self.brain.n} neurons, {self.brain.W.nnz} connections, "
              f"{n_in} input neurons, {self.watch.size} readout neurons")

    def reset(self):
        self.brain.reset()
        self.history.clear()

    def step(self, feats: dict, brain_ms: float) -> Readout:
        idx, rates = [], []
        for (feat, side), nidx in self.input_idx.items():
            idx.append(nidx)
            rates.append(np.full(nidx.size, feats[feat][side] * MAX_INPUT_RATE_HZ))
        self.brain.set_drive(np.concatenate(idx), np.concatenate(rates))

        counts = self.brain.run(brain_ms, self.watch)
        self.history.append((brain_ms, counts))

        # keep ~READOUT_WINDOW_MS of history
        total_ms, total = 0.0, np.zeros(self.watch.size, np.int64)
        for ms, c in reversed(self.history):
            if total_ms >= READOUT_WINDOW_MS:
                break
            total_ms += ms
            total += c

        r = Readout()
        for (name, side), (a, b) in self.readout_idx.items():
            hz = float(total[a:b].mean() / (total_ms * 1e-3)) if b > a else 0.0
            r.rates.setdefault(name, {})[side] = hz

        st = r.rates.get("steer", {"left": 0, "right": 0})
        r.turn_hz = st["right"] - st["left"]
        esc = r.rates.get("escape", {"left": 0, "right": 0})
        back = r.rates.get("backward", {"left": 0, "right": 0})
        if max(esc.values()) >= ESCAPE_THRESHOLD_HZ:
            r.command = "ESCAPE / TAKEOFF"
        elif sum(back.values()) > 2 * STEER_DEADBAND_HZ:
            r.command = "back up"
        elif r.turn_hz > STEER_DEADBAND_HZ:
            r.command = "turn right"
        elif r.turn_hz < -STEER_DEADBAND_HZ:
            r.command = "turn left"
        elif any(sum(v.values()) > 0 for v in r.rates.values()):
            r.command = "straight"
        return r


# ---------------------------------------------------------------------------
# Video sources, including synthetic sanity-check stimuli
# ---------------------------------------------------------------------------

class SyntheticSource:
    """drum-left / drum-right: rotating stripes (optomotor test).
    loom-left / loom-right: expanding dark disc (escape test).
    blank: uniform grey (checks for spontaneous bias)."""

    def __init__(self, kind: str, fps: float = 30.0, size=(480, 640)):
        self.kind, self.fps, self.size, self.t = kind, fps, size, 0
        self.n_frames = int(fps * 8)

    def read(self):
        if self.t >= self.n_frames:
            return False, None
        h, w = self.size
        img = np.full((h, w), 128, np.uint8)
        t = self.t / self.fps
        if self.kind.startswith("drum"):
            # textured panorama drifting sideways, like the fly (or camera)
            # rotating inside a patterned drum. (Pure vertical stripes defeat
            # OpenCV's optical flow, so the pattern has some vertical texture.)
            if not hasattr(self, "pano"):
                rng = np.random.default_rng(1)
                pano = rng.random((h // 16, 2 * w // 16))
                pano = cv2.resize(pano, (2 * w, h), interpolation=cv2.INTER_CUBIC)
                self.pano = (np.clip(pano, 0, 1) * 220 + 20).astype(np.uint8)
            direction = 1 if self.kind != "drum-left" else -1
            shift = int(direction * t * 180.0) % (2 * w)   # 180 px/s drift
            img[:] = np.roll(self.pano, shift, axis=1)[:, :w]
        elif self.kind.startswith("loom"):
            cx = w // 4 if self.kind == "loom-left" else 3 * w // 4
            period = 2.0
            tt = t % period
            r = int(10 / max(0.05, period - tt) * 3)  # 1/t expansion like approach
            cv2.circle(img, (cx, h // 2), min(r, w), 10, -1)
        self.t += 1
        return True, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    def get(self, prop):
        return self.fps if prop == cv2.CAP_PROP_FPS else 0

    def release(self):
        pass


def open_source(src: str):
    if src in ("drum", "drum-right", "drum-left", "loom-left", "loom-right", "blank"):
        return SyntheticSource("drum-right" if src == "drum" else src), False
    if src.isdigit():
        return cv2.VideoCapture(int(src)), True
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source {src!r}")
    return cap, False


def load_autopilot(path: Path):
    """CSV with columns time_s,yaw_rate (deg/s, positive = turning right)."""
    df = pd.read_csv(path)
    return df["time_s"].to_numpy(float), df["yaw_rate"].to_numpy(float)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_overlay(frame, eye: FlyEye, feats, r: Readout, info: str,
                 trace_fly, trace_ap):
    h, w = frame.shape[:2]
    out = frame.copy()

    # movement arrow from frame centre: x = steering (R-L), y = forward (up)
    # minus backward (down); length = magnitude of the desired movement
    max_hz = STEER_FULL_SCALE_HZ
    def mean_rate(name):
        sides = r.rates.get(name, {})
        return float(np.mean(list(sides.values()))) if sides else 0.0
    x = np.clip(r.turn_hz / max_hz, -1, 1)
    y = np.clip((mean_rate("forward") - mean_rate("backward")) / MOVE_FULL_SCALE_HZ, -1, 1)
    mag = np.hypot(x, y)
    if mag > 1:
        x, y = x / mag, y / mag
    base = (w // 2, h // 2)
    reach = 0.45 * min(w, h)
    tip = (int(base[0] + reach * x), int(base[1] - reach * y))
    color = (0, 0, 255) if r.command.startswith("ESCAPE") else (0, 255, 255)
    cv2.circle(out, base, 5, color, -1)
    if np.hypot(tip[0] - base[0], tip[1] - base[1]) >= 3:
        cv2.arrowedLine(out, base, tip, color, 6, tipLength=0.2)
    (tw_, _), _ = cv2.getTextSize(r.command, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.putText(out, r.command, (w // 2 - tw_ // 2, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # fly's-eye thumbnail
    if eye.low_res is not None:
        thumb = cv2.resize(eye.low_res, (160, int(160 * eye.low_res.shape[0] / eye.low_res.shape[1])),
                           interpolation=cv2.INTER_NEAREST)
        th, tw = thumb.shape
        out[10:10 + th, w - tw - 10:w - 10] = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(out, (w - tw - 11, 9), (w - 9, 11 + th), (255, 255, 255), 1)
        cv2.putText(out, "fly's eye (~5 deg/px)", (w - tw - 10, 25 + th),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # bars: inputs (0..1) and outputs (Hz)
    y = 20
    def bar(label, val, vmax, col):
        nonlocal y
        cv2.putText(out, label, (10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        cv2.rectangle(out, (150, y), (150 + int(120 * min(val / vmax, 1)), y + 12), col, -1)
        cv2.rectangle(out, (150, y), (270, y + 12), (200, 200, 200), 1)
        y += 17
    for f in FlyEye.FEATURES:
        for side in ("left", "right"):
            bar(f"in {f[:12]} {side[0].upper()}", feats[f][side], 1.0, (255, 180, 0))
    y += 6
    for name, sides in r.rates.items():
        for side in ("left", "right"):
            bar(f"DN {name} {side[0].upper()} {sides[side]:.0f}Hz", sides[side], 100.0, (0, 200, 0))

    cv2.rectangle(out, (0, h - 22), (w, h), (0, 0, 0), -1)
    cv2.putText(out, info, (10, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # scrolling comparison trace
    if len(trace_fly) > 1:
        ph, px0, pw = 80, w - 330, 320
        py0 = h - ph - 30
        cv2.rectangle(out, (px0, py0), (px0 + pw, py0 + ph), (40, 40, 40), -1)
        cv2.line(out, (px0, py0 + ph // 2), (px0 + pw, py0 + ph // 2), (100, 100, 100), 1)
        def plot(vals, vmax, col):
            vals = list(vals)[-pw:]
            pts = [(px0 + i, int(py0 + ph / 2 - np.clip(v / vmax, -1, 1) * ph / 2 * 0.9))
                   for i, v in enumerate(vals)]
            cv2.polylines(out, [np.array(pts, np.int32)], False, col, 1)
        plot(trace_fly, max_hz, (0, 255, 255))
        cv2.putText(out, "fly turn (R-L Hz)", (px0 + 4, py0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        if trace_ap:
            plot(trace_ap, 60.0, (255, 0, 255))
            cv2.putText(out, "autopilot yaw", (px0 + 150, py0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
    return out


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="drum",
                    help="webcam index (0), video path, or drum/drum-left/loom-left/loom-right/blank")
    ap.add_argument("--data-dir", default="fly_data", type=Path)
    ap.add_argument("--fov", type=float, default=90.0, help="camera horizontal field of view, degrees")
    ap.add_argument("--brain-ms-per-frame", type=float, default=None,
                    help="brain time simulated per video frame (default: 1000/fps, i.e. tick-locked)")
    ap.add_argument("--dt", type=float, default=0.1, help="integration step, ms (0.1 matches Shiu et al.)")
    ap.add_argument("--mirror", action="store_true", help="flip video left-right (bias check)")
    ap.add_argument("--compare", type=Path, help="autopilot CSV: time_s,yaw_rate")
    ap.add_argument("--log", type=Path, help="write per-frame readout CSV here")
    ap.add_argument("--out", type=Path, help="write annotated video here")
    ap.add_argument("--headless", action="store_true", help="no window")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ensure_data(args.data_dir)
    ctrl = FlyController(args.data_dir, args.dt, args.seed)
    cap, live = open_source(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    eye = FlyEye(args.fov, fps)
    brain_ms = args.brain_ms_per_frame or 1000.0 / fps

    ap_t, ap_yaw = load_autopilot(args.compare) if args.compare else (None, None)
    trace_fly, trace_ap = collections.deque(maxlen=400), collections.deque(maxlen=400)

    writer, log_f, log_w = None, None, None
    if args.log:
        log_f = open(args.log, "w", newline="")
        log_w = csv.writer(log_f)
        header = ["frame", "video_t_s", "brain_t_ms", "turn_hz", "command"]
        header += [f"in_{f}_{s}" for f in FlyEye.FEATURES for s in ("left", "right")]
        header += [f"dn_{n}_{s}" for n in READOUT_CELL_TYPES for s in ("left", "right")]
        log_w.writerow(header)

    frame_i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)
            t_wall = time.perf_counter()

            feats = eye(frame)
            r = ctrl.step(feats, brain_ms)

            wall = time.perf_counter() - t_wall
            speed = (brain_ms / 1000.0) / max(wall, 1e-6)
            video_t = frame_i / fps
            trace_fly.append(r.turn_hz)
            if ap_t is not None:
                trace_ap.append(float(np.interp(video_t, ap_t, ap_yaw)))

            info = (f"frame {frame_i}  video {video_t:5.2f}s  brain {ctrl.brain.t_ms/1000:5.2f}s  "
                    f"{speed:4.2f}x realtime")
            vis = draw_overlay(frame, eye, feats, r, info, trace_fly, trace_ap)

            if log_w:
                row = [frame_i, f"{video_t:.3f}", f"{ctrl.brain.t_ms:.1f}", f"{r.turn_hz:.2f}", r.command]
                row += [f"{feats[f][s]:.3f}" for f in FlyEye.FEATURES for s in ("left", "right")]
                row += [f"{r.rates[n][s]:.1f}" for n in READOUT_CELL_TYPES for s in ("left", "right")]
                log_w.writerow(row)
            if args.out:
                if writer is None:
                    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"),
                                             fps, (vis.shape[1], vis.shape[0]))
                writer.write(vis)
            if not args.headless:
                cv2.imshow("fly navigator", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("r"):
                    ctrl.reset()
            elif frame_i % 10 == 0:
                print(info, "|", r.command, f"turn {r.turn_hz:+.1f} Hz")

            frame_i += 1
            if args.max_frames and frame_i >= args.max_frames:
                break
    finally:
        cap.release()
        if writer:
            writer.release()
        if log_f:
            log_f.close()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
