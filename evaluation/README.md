# Senko Evaluation via OpenBench
This directory contains scripts to evaluate the Senko diarization pipeline using the [OpenBench](https://github.com/argmaxinc/OpenBench) benchmarking framework by Argmax. This allows for standardized, reproducible evaluation and comparison with other diarization systems.

## Setup
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
uv run evaluate.py --dataset voxconverse

# All datasets
uv run evaluate.py --all-datasets

# Only a few samples
uv run evaluate.py --dataset voxconverse --num-samples 10

# [--device, --vad, --clustering] options also available; see Senko DOCS.md
```