# Cloud Metrics Predictive Alerting

## Design Decisions & Methodology

### 1. Problem Formulation
Instead of a simple anomaly detection, I've formulated this as a **binary classification task over a sliding window**. 
* **Window (W):** We observe the last 60 minutes of data.
* **Horizon (H):** We predict if an incident will occur in the next 15 minutes.
* **Why?** This gives SRE teams enough lead time to react before the system actually fails.

### 2. Data Strategy
I chose to generate **synthetic data** instead of using a public dataset. 
* **Why?** This allows for full control over the "ground truth". In real-world cloud environments, incidents are rare (class imbalance). By generating data, I can simulate specific failure modes like "Memory Leaks" and "Traffic Spikes" to test the model's robustness systematically.

#### File: `src/data_gen.py`
This script acts as the foundational data source, generating synthetic CPU telemetry to simulate a production cloud environment. It is designed to create a "controlled lab" for testing predictive alerting logic.
* **Synthetic Data Generation:** It produces a 1D time-series incorporating daily seasonality, Gaussian noise, and specific failure modes like "Traffic Spikes" and "Memory Leaks."
* **Forced Incident Injection:** To address class imbalance, the script deterministically injects 20 incidents at regular intervals, ensuring the dataset contains at least 550 incident-impacted points for training.
* **Operational Realism:** All generated metrics are clipped between 0-100% to mimic real-world system constraints, providing a robust and portable dataset without relying on external APIs.

#### File: `src/preprocessing.py`
This module handles the transformation of raw time-series metrics into a supervised learning dataset. 

**Technical Approach:**
* **Windowing Strategy:** Implements a sliding window of size $W=60$. This provides the model with one hour of historical context to identify patterns preceding an incident.
* **Labeling Logic:** Uses a look-ahead horizon of $H=15$. A window is labeled as "positive" ($1$) if an incident occurs at any point within the next 15 steps.
* **Format:** Outputs data as NumPy arrays, optimized for direct input into Deep Learning frameworks (like PyTorch or TensorFlow).
By framing the problem this way, we shift from "detecting an ongoing crash" to "predicting an upcoming incident," which is the core requirement for an effective alerting system.


### 3. Problem Formulation: Sliding Window Strategy
This section describes the logical framework used to transform the continuous stream of CPU metrics into a structured dataset for supervised learning. By framing the problem as a predictive classification task, the system shifts from reactive detection to proactive alerting.
* **Observation Window (W=60):** The model analyzes a 60-minute window of historical data. This period is long enough to capture significant patterns, such as gradual resource exhaustion or seasonal usage trends.
* **Prediction Horizon (H=15):** The system evaluates the status of the system for the next 15 minutes. This specific timeframe serves as a "Lead Time," giving engineers enough room to intervene before a potential system failure.
* **Predictive Labeling:** A window is labeled as "Warning" (1) if any incident or anomaly is detected within the 15-minute horizon. If the horizon remains stable, the window is labeled as "Safe" (0).
* **Data Transformation:** This approach converts the metric sequence into a fixed-size feature matrix (X) and a target vector (y), ensuring the dataset is ready for benchmarking both classical and deep learning architectures.

#### File: `src/model.py`
This script defines the primary neural network architecture using PyTorch. I have implemented a **1D Convolutional Neural Network (1D-CNN)** to leverage my background in Computer Vision for time-series analysis.
* **Spatial Pattern Recognition:** Using 1D kernels, the model learns to identify the "shape" of incoming incidents (e.g., sudden spikes or exponential growth) rather than just looking at raw numerical values.
* **Translation Invariance:** The convolutional layers allow the model to detect pre-incident signatures regardless of their exact position within the 60-minute window.
* **Computational Efficiency:** The architecture is designed to be lightweight, ensuring low-latency inference suitable for real-time monitoring environments.

#### File: `src/baseline.py`
This module establishes a performance benchmark using a **Random Forest Classifier**. In industrial AI applications, having a classical machine learning baseline is essential to validate the added value of deeper architectures.
* **Baseline Purpose:** It treats each of the 60 timestamps as an independent feature, providing a "performance floor" for model comparison.
* **Robustness:** Random Forest is naturally resilient to outliers and requires minimal feature scaling, making it a reliable sanity check for our 1D-CNN results.

### 4. Training and Evaluation Pipeline
This section defines the methodology used to validate the predictive alerting hypothesis. The focus is on ensuring the system can generalize to unseen telemetry while maintaining high sensitivity to critical failures.
* **Stratified Validation:** I utilized an 80/20 train-test split with stratified sampling. This is essential for imbalanced datasets, ensuring that the rare "Warning" windows are proportionately represented in both the training and evaluation phases.
* **Standardization:** All input sequences were normalized using a `StandardScaler` fitted on the training data. This ensures the 1D-CNN's gradient descent converges efficiently without being skewed by raw CPU value magnitudes.
* **Cost-Sensitive Optimization:** Both models were configured to prioritize the minority class. I implemented `class_weight='balanced'` for the baseline and a custom loss-weighting strategy for the neural network to specifically penalize missed detections.
* **Advanced Metrics:** Beyond simple accuracy, the pipeline tracks **Precision**, **Recall**, and **F1-Score**. This provides a transparent look at the trade-off between "Alert Fatigue" (false positives) and "System Reliability" (detection rate).

#### File: `src/train.py`
This is the main orchestration script that executes the end-to-end machine learning workflow. It serves as the primary benchmark for the project, comparing classical and deep learning approaches.
* **Unified Benchmarking:** The script trains both the Random Forest and the 1D-CNN under identical conditions, outputting a side-by-side performance table to justify the selection of the final model.
* **Weighted Loss Function:** I implemented a manual weight calculation in the PyTorch loop, multiplying the loss of positive samples by the negative/positive ratio. This prevents the model from defaulting to a "safe" majority-class prediction.
* **Threshold Calibration:** The script utilizes a 0.3 sensitivity threshold. This calibration is designed to prioritize **Recall**, ensuring that the system acts as a reliable early-warning mechanism for SRE teams.
* **Comprehensive Reporting:** It generates a detailed classification report, allowing for the analysis of how the model handles the specific morphologies of traffic spikes versus memory leaks.

### 5. Conclusion: Results and Analysis
The benchmarking results provide a clear technical justification for using Deep Learning in predictive infrastructure monitoring.
* **Detection Dominance:** The 1D-CNN achieved a **Recall of 1.0000**, successfully identifying 100% of the incidents in the test set. In contrast, the Random Forest baseline failed to identify any incidents (Recall: 0.0000), proving it cannot capture the temporal dependencies in raw sequences.
* **Performance Trade-offs:** While the CNN achieved an **F1-Score of 0.2449**, its low precision (0.1395) indicates a high number of false positives. This "paranoiac" behavior is a direct result of the high sensitivity required to ensure no system failure goes undetected.
* **Engineering Verdict:** The experiment confirms that 1D-CNNs are superior at recognizing the "shape" of impending failures. For a production environment, this model serves as a robust "safety net" that guarantees incident visibility, though further threshold tuning is recommended to reduce alert noise.
* **Future Scalability:** This modular pipeline can be easily extended to multi-variate data (RAM, Disk, Network). Adding more dimensions would likely improve the Precision, as the model would have more context to distinguish between normal load and genuine failure signatures.


| METRIC | RANDOM FOREST | 1D-CNN |
| :--- | :---: | :---: |
| Precision | 0.0000 | 0.1395 |
| Recall | 0.0000 | 1.0000 |
| F1-Score | 0.0000 | 0.2449 |


#### Configuration B: Enhanced Feature Engineering (Optimized)
By enabling statistical feature extraction (Mean, Std Dev, Max) and refining thresholds, the Random Forest becomes significantly more balanced.
| METRIC | RANDOM FOREST | 1D-CNN |
| :--- | :---: | :---: |
| Precision | 0.8750 | 0.1224 |
| Recall | 0.5833 | 1.0000 |
| F1-Score | 0.7000 | 0.2182 |

#### Final Verdict
* **Model Flexibility:** The 1D-CNN consistently achieves a **1.0000 Recall**, serving as a perfect "safety net" that never misses an incident.
* **Operational Stability:** The Random Forest, when provided with statistical features, offers a much higher **F1-Score (0.7000)**, making it more suitable for production environments where minimizing false alarms is a priority.
* **Engineering Insight:** The ability to pivot between these profiles demonstrates that the modular pipeline is ready to be fine-tuned based on real-world infrastructure needs.

### 6. Technical Defense and Future Adaptation
This section addresses the design decisions and scalability of the proposed solution for a production environment.
* **Why 1D-CNN?** Unlike classical models that treat time-steps as independent variables, 1D-CNNs utilize kernels to recognize the "morphology" of a signal. This allows the system to detect the specific geometric patterns of memory leaks and traffic spikes.
* **Known Limitations:** The reliance on synthetic data means the model may require fine-tuning on real-world telemetry to handle non-Gaussian noise. The current trade-off prioritizes Recall (1.00) over Precision to ensure zero-downtime visibility.
* **Real-World Adaptation:** To deploy this in a live infrastructure, the input vector would be expanded to include multivariate metrics (RAM, I/O). The system would be integrated into a Prometheus/Grafana pipeline, utilizing a feedback loop where SRE "False Positive" reports act as new training labels for continuous threshold optimization.

## 🚀 How to Run
To reproduce the results and the benchmark comparison, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/IraitzSantafosta/cloud-metrics-predictive-alerting.git
   cd cloud-metrics-predictive-alerting

2. **Set up the environment:**
   ```bash
   python -m venv .venv
2.1 **Windows**
   ```bash
   .\.venv\Scripts\activate
2.2 **Linux/macOS**
   ```bash
   source .venv/bin/activate
   pip install -r requirements.txt

3. **Execute the pipeline:**
3.1 Generate the synthetic data
   ```bash
   python src/data_gen.py
3.2 Preprocess into sliding windows
   ```bash
   python src/preprocessing.py
3.3 Train and compare models
   ```bash
   python src/train.py
