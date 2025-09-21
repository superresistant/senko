# Pyannote `segmentation-3.0` CoreML
This directory contains code for interfacing with the Pyannote `segmentation-3.0` CoreML converted model, using it to run VAD (voice activity detection).

The model conversion as well as all the code is taken from the excellent FluidAudio project by Fluid Inference:
- CoreML model: [https://huggingface.co/FluidInference/speaker-diarization-coreml](https://huggingface.co/FluidInference/speaker-diarization-coreml)
- Interfacing code: [https://github.com/FluidInference/FluidAudio](https://github.com/FluidInference/FluidAudio)

I have simply made the Swift code accessible from Python through a compiled/shared lib, and lightly rearranged it to isolate out only the VAD function.