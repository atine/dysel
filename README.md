# Learning the Supports for Categorical Critic in Reinforcement Learning

This repository contains the official implementation of **DySEL** (Dynamic Support
Endpoint Learning), from our paper *"Learning the Supports for Categorical Critic in
Reinforcement Learning"* (Reinforcement Learning Conference / Reinforcement Learning
Journal, 2026).

**Project page:** http://isatine.xyz/projects/dysel

Classification-based value learning with the Gaussian Histogram Loss (HL-Gauss)
reframes value estimation as classification over a categorical support, but requires a
pre-defined support interval `[v_min, v_max]` that is rarely known a priori and is poorly
suited to the non-stationary returns seen during RL training. DySEL instead **dynamically
learns the lower and upper bounds of the support** jointly with the categorical
representation, framed as a constrained optimisation (adversarial min-max) problem that
minimises the support width while keeping the target distribution covered.


## Install
We use [uv](https://docs.astral.sh/uv/) to manage the environment. Install it with:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
We also provide an environment variable control script for a faster start.

 - `uv venv` (creates `.venv`)
 - `source setup_dmcontrol.sh` (edits may be required, depending on your system)
 - `uv pip install -r requirements.txt`


## Experiments tracking
We use wandb to track our experiments.
To disable wandb tracking, add `no_track=True` at the end of any command.


## Usage
Applicable tasks are listed in `cfgs/task/`, and applicable algorithms are listed in
`cfgs/algo/`. If you disable wandb tracking, results are written under `logs/`.
Below we show example commands, varying the task and disabling wandb tracking for
convenience.

- TD3 (scalar MSE critic, baseline)
```bash
python main.py task=dmc_quadrupedrun algo=td3 no_track=True no_disc_logging=False
```

- TD3 + HL-Gauss (fixed-support categorical critic, baseline)
```bash
python main.py task=dmc_cheetahrun algo=td3_hlg no_track=True no_disc_logging=False
```

- TD3 + DySEL (learned support, **ours**)
```bash
python main.py task=dmc_hopperhop algo=td3_dysel no_track=True no_disc_logging=False
```


## Helper scripts
Two small scripts are included for quick sanity checks:
- `python test_jax_devices.py` — check that JAX detects the devices (e.g. a GPU) available in the current compute setup.
- `python test_all_algos.py` — run every algorithm in `cfgs/algo/` on a dummy batch to confirm they all initialise and compile.


## Citation
If you find this work useful, please cite:
```bibtex
@inproceedings{formosan-landlocked-salmon,
author    = {Jen-Yen Chang and Takayuki Osa and Tatsuya Harada},
title     = {Learning the Supports for Categorical Critic in Reinforcement Learning},
booktitle = {Reinforcement Learning Conference (RLC)},
year      = {2026},
}
```
