# Autonomous Driving V2X Preprocessing Pipeline
**Master's Thesis Research**

This repository contains the data engineering pipeline required to convert the **Waymo Open Motion Dataset** into multi-dimensional, deep-learning-ready tensors optimized for Controllable Diffusion Models (MotionDiffuser).

## Features
1. **Raw Ingestion:** Deserializes Google Protocol Buffers (`.tfrecord`).
2. **Ego-Centric Normalization:** Translates global GPS coordinate systems to an Ego-centric reference frame `(0,0)`, applying rotational matrices to align agent trajectory vectors.
3. **Tensor Engineering:** Pads variable traffic scenarios into uniform `[64, 91, 6]` tensors, integrating binary Valid Masks for Transformer attention layers.

## Execution
To generate the final `.npy` tensors:
```bash
cd src/
python3 build_tensors.py