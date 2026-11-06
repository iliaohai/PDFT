# PDFT

Implementation of **Pre-deployment and fine-tuning strategies of Wi-Fi for irregular coverage** ([paper](https://doi.org/10.1016/j.ins.2026.123605)). PDFT places Wi-Fi access points using pre-deployment followed by coordinate fine-tuning.

## Code tree

```text
PDFT/
├── proposed_run.py       # Original experiment entry
├── proposed_common.py    # PDFT optimization and evaluation
├── common.py             # Map, walls, and radio parameters
├── util.py               # Wall-intersection utilities
├── foa_run.py            # FOA baseline
├── zoa_run.py            # ZOA baseline
├── pd+foa_run.py          # PD + FOA
├── pd+zoa_run.py          # PD + ZOA
├── grid.data             # Recorded results
├── more ap.data          # AP-count results
├── coverage data         # Coverage results
├── run_tutorial.py       # Added runnable entry
├── requirements-tutorial.txt
├── citation.bib
├── TUTORIAL.md
└── README.md
```

## Setup

```bash
git clone https://github.com/iliaohai/PDFT.git
cd PDFT
git checkout ea58cef34d05674ba57b73db3a6b86044c35d6cb
```

Copy the accompanying `run_tutorial.py` and `requirements-tutorial.txt` into the repository root. These files are additions to the upstream code.

```bash
conda create -n pdft python=3.8.5 -y
conda activate pdft
python -m pip install -r requirements-tutorial.txt
```

## Run

```bash
# Verified example
python run_tutorial.py --aps 3 --seed 0 --output results/aps3.json

# Nine-AP deployment
python run_tutorial.py --aps 9 --seed 0 --output results/aps9.json
```

The runner uses the original PDFT functions and resolves missing project imports. It skips plotting, whose background assets are absent from the repository. No external dataset is required; the map is defined in `common.py`.

The JSON stores initial, pre-deployment, and final AP coordinates, RSSI, and coverage. `rss[4]` is mean RSSI across R1–R3; `coverage[3]` is their covered-point fraction. The verified three-AP run returned **−51.36 dBm** and **0.99**, respectively. This verifies code execution, rather than all paper results.

Change `--aps` and `--seed` for additional runs. Radio parameters, grid spacing, and walls are in `common.py`; regional priorities are in `proposed_common.py`.

## Citation

```bibtex
@article{chen2026pdft,
  title   = {Pre-deployment and fine-tuning strategies of {Wi-Fi} for irregular coverage},
  author  = {Chen, Song and Liao, Hai and Liang, Yan and Liu, Tu and Min, Fan},
  journal = {Information Sciences},
  volume  = {753},
  pages   = {123605},
  year    = {2026},
  doi     = {10.1016/j.ins.2026.123605}
}
```
