# fly_navigator

Feed a webcam, a video file, or a synthetic test stimulus into a whole-brain
fruit fly model and show which way the fly "wants" to move.

## Setup

    pip install -r requirements.txt
    python fly_navigator.py --source drum

The first run downloads ~130 MB into `./fly_data/`:

- `Completeness_783.csv`, `Connectivity_783.parquet`: FlyWire v783 connectome
  as packaged by Shiu et al. 2024 (github.com/philshiu/Drosophila_brain_model)
- `neuron_annotations.tsv`: FlyWire cell types (github.com/flyconnectome/flywire_annotations)

It then builds a sparse-matrix cache (`connectome_cache.npz`) once.

## Sources

| `--source` | What it is |
|---|---|
| `0`, `1`, ... | webcam index |
| `path/to/video.mp4` | recorded video |
| `drum` / `drum-left` | textured panorama drifting right/left (optomotor test) |
| `loom-left` / `loom-right` | expanding dark disc (escape test) |
| `blank` | uniform grey (should produce no activity) |

## Useful flags

    --fov 90                      camera horizontal field of view (degrees)
    --mirror                      flip video left/right (bias check)
    --compare autopilot.csv       CSV with time_s,yaw_rate (deg/s, + = right)
    --log readout.csv             per-frame inputs and descending-neuron rates
    --out annotated.mp4           save the overlay video
    --brain-ms-per-frame 33.3     brain time per frame (default 1000/fps)
    --headless                    no window (print to terminal)

Keys: `q`/Esc quit, `r` reset brain state.

## What to edit

Everything at the top of `fly_navigator.py`:

- `FEATURE_TO_CELL_TYPES`: which visual neuron types each video feature drives
- `READOUT_CELL_TYPES`: which descending neurons are read as steer/escape/etc.
- `FEATURE_GAIN`, `FEATURE_FLOOR`: input sensitivity
- `STEER_DEADBAND_HZ`, `ESCAPE_THRESHOLD_HZ`: how rates become commands

Both mappings are hand-designed guesses. The connectome in between is real
wiring, but the model gives every synapse the same strength.

## Sanity results (8 s synthetic stimuli, dt 0.1 ms, seed 0)

- `loom-left`: giant fibers fire, DNa steering goes right (turn away). Good.
- `loom-right`: giant fibers fire, steering goes left. Good.
- `blank`: silent. Good.
- `drum-left` / `drum-right`: little or no steering. The optomotor response
  does NOT come through with the default HS-cell input. Treat that as an
  open problem, not a bug you forgot to fix.

## Speed

On CPU this runs at roughly 0.1x real time (10 s of wall clock per second of
video). Fine for recorded video; too slow for smooth live webcam use. The
`Brain` class is small and self-contained, so porting it to PyTorch/CUDA is
the natural next step for live use.
