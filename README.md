# AI-ML based Intelligent Dead Reckoning System (SIH 26168)

## Overview
This repository contains the core Machine Learning pipeline for a lightweight, Edge-Deployable Dead Reckoning system. It predicts vehicle Speed and Heading using only smartphone IMU data when GNSS (GPS) signals are lost, allowing seamless navigation in tunnels, urban canyons, and underground parking.

## Key Results (Test Set)
*   **Speed MAE:** 2.37 km/h
*   **Heading MAE:** 5.6°
*   **Model Size:** ~40k parameters (Perfect for mobile TFLite/ONNX deployment).

## Core Architecture
1.  **TCN (Temporal Convolutional Network):** Filters engine vibration noise.
2.  **Causal Transformer:** Learns long-term temporal dynamics.
3.  **Extended Kalman Filter (EKF):** Fuses AI predictions with Gyroscope physics.
4.  **Map Matching:** Snaps the drifting path to OpenStreetMap (OSM) roads.

**[Read the full step-by-step Project Workflow here (WORKFLOW.md)](./WORKFLOW.md)**

## 📂 Repository Structure
*   `best_tcn_transformer_v4.pt`: The trained PyTorch model weights.
*   `dead-sih.ipynb`: The complete Kaggle pipeline (Data cleaning, AI training, EKF, and Map Matching).
*   `idr-demo.html`: A frontend demo of the system.
*   `WORKFLOW.md`: Detailed technical breakdown of how the project was built.

##  Quick Start (How to load the AI)
```python
import torch

# 1. Define the model architecture (TCN + Transformer)
# (See notebook cells for the full TCN_Transformer_Small class)
model = TCN_Transformer_Small(in_features=9)

# 2. Load the trained weights
model.load_state_dict(torch.load("best_tcn_transformer_v4.pt", map_location='cpu'))
model.eval()

print("AI Model successfully loaded and ready for inference!")
