# AI-ML based Intelligent Dead Reckoning System (SIH 26168)

## Overview
This repository contains the core Machine Learning pipeline and Edge-Deployable Dead Reckoning system. It predicts vehicle Speed and Heading using only smartphone IMU data when GNSS (GPS) signals are lost.

## Architecture
1. **TCN (Temporal Convolutional Network):** Extracts local features and filters out engine vibration noise.
2. **Causal Transformer:** Learns long-term temporal dynamics without looking into the future.
3. **EKF (Extended Kalman Filter):** Fuses AI predictions with raw gyroscope physics for smooth trajectories.
4. **Map Matching:** Uses OpenStreetMap (OSM) and Viterbi algorithm to snap the drifting path back to the road.

## Results (Test Set on Driver A - Suburban S1)
* **Speed MAE:** 2.37 km/h
* **Heading MAE:** 5.6°
* **Model Size:** ~40k parameters (Perfect for mobile TFLite deployment).

## Files
* `best_tcn_transformer_v4.pt`: The trained PyTorch model weights.
* `dead-sih.ipynb`: The complete training and evaluation notebook.
* `idr-demo.html`: A lightweight frontend demo of the system.

## Usage
```python
import torch

# Load the model architecture (Ensure TCN_Transformer_Small is defined)
model = TCN_Transformer_Small(in_features=9)
model.load_state_dict(torch.load("best_tcn_transformer_v4.pt", map_location='cpu'))
model.eval()
print("AI Model successfully loaded and ready for inference!")
