# Senko Evaluation via OpenBench
This directory contains scripts to evaluate the Senko diarization pipeline using the [OpenBench](https://github.com/argmaxinc/OpenBench) benchmarking framework by Argmax. This allows for standardized, reproducible evaluation and comparison with other diarization systems.

## Setup
Create a Python virtual environment and activate it
```
uv venv --python 3.11.13 .venv
source .venv/bin/activate
```
Then install dependancies
```sh
# For NVIDIA GPUs with CUDA compute capability >= 7.5 (~GTX 16 series and newer)
uv pip install -r requirements.txt "git+https://github.com/narcotic-sh/senko.git[nvidia]"

# For NVIDIA GPUs with CUDA compute capability < 7.5 (~GTX 10 series and older)
uv pip install -r requirements.txt "git+https://github.com/narcotic-sh/senko.git[nvidia-old]"

# For Mac (macOS 14+) and CPU execution on all other platforms
uv pip install -r requirements.txt "git+https://github.com/narcotic-sh/senko.git"
```

## Run Evaluation
The script will auto-download the datasets.
```sh
# Evaluate specific dataset
python evaluate.py --dataset voxconverse

# All datasets
python evaluate.py --all-datasets

# Only a few samples
python evaluate.py --dataset voxconverse --num-samples 10

# [--device, --vad, --clustering] options also available; see Senko DOCS.md
```