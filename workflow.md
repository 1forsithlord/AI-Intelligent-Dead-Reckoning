# Project Workflow: AI-ML Intelligent Dead Reckoning System

##  Executive Summary
We developed a lightweight, edge-deployable AI engine that takes over navigation when GNSS (GPS) signals fail. By fusing raw smartphone IMU data with a Causal Transformer, an Extended Kalman Filter, and OpenStreetMap, we achieved a **2.37 km/h Speed Error** and **5.6° Heading Error** in blind, GPS-denied environments.

---

## Phase 1: Data Acquisition & Environment Setup
**The Goal:** Find a real-world dataset to train a robust AI model.
*   **Dataset Sourced:** We secured the **IO-VNBD** (Inertial and Odometry Benchmark Dataset) from the original authors (Uche Onyekpe). This dataset contains 2GB+ of real-world driving data (98+ hours, 5700km).
*   **Environment:** We utilized **Kaggle Notebooks** with free GPU (P100/T4) acceleration to process the massive dataset.
*   **Data Loading:** We parsed the raw `.csv` files, separating the raw IMU sensor data (`S-M.csv`) from the vehicle's ground truth (`V-M.csv`).

## Phase 2: Data Engineering & Preprocessing
**The Goal:** Clean and align the messy, multi-frequency real-world data.
*   **Time Synchronization:** We discovered a massive clock mismatch between the phone's IMU logger and the car's GPS logger. We wrote a cross-correlation algorithm (using `scipy.signal`) to auto-detect a **-211.8 second offset** and perfectly align the timelines.
*   **Downsampling:** We downsampled the 100Hz IMU data to 10Hz (using mean pooling) to match the GPS frequency, acting as an anti-aliasing filter.
*   **Glitch Masking:** We noticed the GPS compass frequently dropped to `0.0°` randomly. We wrote a rolling-median filter that successfully identified and **masked out 11.3% of the corrupted labels**, preventing the AI from learning garbage.
*   **Windowing:** We sliced the continuous data into 50-timestep (5-second) sliding windows, ensuring the windows never crossed timeline gaps.

## Phase 3: AI Model Architecture (V1 to V4)
**The Goal:** Build a neural network that can learn vehicle physics from vibrations.
*   **V1 (TCN + GRU):** We started with a Temporal Convolutional Network (TCN) for noise filtering and a GRU for temporal dynamics. *Result: 6.7 km/h Speed Error, but 92° Heading Error.*
*   **V2 (Added Magnetometer):** The heading was failing because Gyroscopes alone drift. We added Magnetometer (compass) data, expanding the input to 9 features. We also implemented a **Custom Angular Loss** function to handle the 359°->0° wrap-around. *Result: Heading error dropped to ~20°.*
*   **V3 (Causal Transformer):** We replaced the GRU with a **Causal Transformer Encoder** with Positional Encoding. This allowed the model to "pay attention" to important past events (like a sharp turn) without cheating by looking into the future.
*   **V4 (Overfitting Fix):** We discovered the model was memorizing the driver's route. We fixed this by implementing a **Random Window Split** (mixing all data randomly) instead of a chronological split. We also added Gaussian Noise Augmentation and reduced the model to **~40k parameters** for edge deployment.

## Phase 4: Sensor Fusion (The EKF)
**The Goal:** Smooth out the AI's predictions using physics.
*   We built an **Extended Kalman Filter (EKF)** in Python.
*   The EKF takes the AI's predicted Speed and Heading as the "Measurement".
*   It takes the raw Gyroscope data and integrates it for the "Prediction Step".
*   It fuses them together, resulting in a 2D trajectory (X, Y) that is mathematically smoothed and free of AI jitter.

## Phase 5: Map Matching
**The Goal:** Ensure the car never drives through buildings when GPS is lost.
*   We used the `osmnx` library to download the actual road network of the UK (where the data was recorded).
*   We implemented a **HMM (Hidden Markov Model) / Viterbi-based Map Matcher**.
*   We took the drifting EKF trajectory and mathematically "snapped" it to the nearest road lines. 
*   *Visual Proof:* We generated a map showing the Red line (Raw AI Drift) cutting through grass, and the Blue Dashed line (Map Matched) perfectly following the road.

## Phase 6: Repository & Deployment Prep
*   **Model Weights:** Saved the optimal PyTorch weights (`best_tcn_transformer_v4.pt`).
*   **GitHub:** Documented the entire Kaggle pipeline in a `.ipynb` file and pushed it to GitHub along with the model and an HTML demo.

---

## Final Results Achieved
*   **Speed Accuracy:** 2.37 km/h Mean Absolute Error (MAE).
*   **Heading Accuracy:** 5.6° Mean Absolute Error.
*   **Model Size:** 40k parameters (Less than 1MB, easily deployable via TFLite to Android/iOS).
*   **Robustness:** Handled 11.3% corrupted GPS data and operated completely offline.

---

## 💡 Technologies Used
*   **Machine Learning:** PyTorch, Temporal Convolutional Networks (TCN), Causal Transformers.
*   **Data Science:** Pandas, NumPy, SciPy (Signal Processing).
*   **Robotics & Navigation:** Extended Kalman Filter (EKF), OpenStreetMap (OSMnx), HMM Map Matching.
*   **Environment:** Kaggle (GPU Accelerated), Git/GitHub.
