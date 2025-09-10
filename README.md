# Multimodal BD Detection

## Overview  
A state-of-the-art Python pipeline for classification of mental health states—**Depression, Mania, Euthymia**—leveraging multimodal data (text, audio, video, physiological signals) via deep learning. The transformer-based architecture incorporates temporal sequence modeling and advanced data augmentation techniques to achieve near-perfect detection accuracy.

---

## Current Status (September 2025) — Deployment Ready!

| Metric             | Score  | Status      |
|--------------------|--------|-------------|
| Test Accuracy      | 99.79% | 🎉 Excellent |
| Weighted F1-Score  | 99.79% | 🎉 Excellent |
| Macro F1-Score     | 99.25% | 🎉 Excellent |
| Weighted Precision | 99.79% | 🎉 Excellent |
| Weighted Recall    | 99.79% | 🎉 Excellent |

### Per-Class Performance

| Class       | Precision | Recall  | F1-Score | Clinical Significance               |
|-------------|-----------|---------|----------|-----------------------------------|
| Depression  | 98.9%     | 97.2%   | 98.0%    | High sensitivity for detection    |
| Mania       | 99.6%     | 99.8%   | 99.7%    | Exceptional specificity           |
| Euthymia    | 100%      | 100%    | 100%     | Perfect rare-class detection      |

---

## Goal  
Develop a clinically validated, robust multimodal AI system for mental health state detection, suitable for publication and real-world clinical deployment.

---

## Features

- 🔄 Multimodal input handling: text (Twitter, Kaggle), audio & video (CMU-MOSEI), physiological signals (OBF Psychiatrist)  
- 🧠 Temporal sequence modeling with transformer architecture  
- 🔧 Rigorous data cleaning ensuring stability (NaN/Inf handling)  
- 📈 Adaptive data augmentation for robust training  
- ⚖️ Balanced loss functions and sampling addressing class imbalance, especially for rare Euthymia class  
- 📊 Comprehensive evaluation including confusion matrices, per-class metrics, and sequence-level analysis  
- 🛑 Early stopping and model checkpointing for training efficiency  
- 📱 High-confidence prediction system reducing uncertain outputs  
- 🔍 Detailed error and interpretability analysis  

---

## Project Structure

multimodal-BD-Detection/
├── data/ # Input datasets (JSONL files)
├── configs/ # Configuration settings
├── models/ # Model definitions
├── dataset/ # Custom Dataset class and preprocessing
├── quality/ # Data quality scripts
├── evaluator/ # Evaluation and visualization
├── scripts/ # Main training and evaluation pipeline
├── trainer/ # Model training logic
├── requirements.txt # Python dependencies
├── README.md # This file
├── evaluation_results.png # Evaluation plots
├── training_history.png # Training progress visualization
└── improved_results_summary.pkl # Serialized evaluation summary

text

---

## Installation & Setup

1. Clone the repository:  
git clone https://github.com/yourusername/multimodal-BD-Detection.git
cd multimodal-BD-Detection

text

2. Create and activate a virtual environment:  
python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate

text

3. Install dependencies:  
pip install -r requirements.txt

text

4. Prepare your multimodal data following the schema documented in `configs/config.py` and place in `data/`.

5. Adjust hyperparameters and paths in `configs/config.py` as needed.

---

## Usage

Run the training and evaluation pipeline:  
python scripts/run_pipeline.py

text

This will:  
- Train model with early stopping on validation data.  
- Evaluate on test set with detailed metrics and plots.  
- Save training history and best model checkpoint.

---

## Clinical Significance

- **Depression** detection with 97.2% sensitivity and 98.9% precision minimizes missed cases while controlling false alarms.  
- **Mania** detected with near-perfect precision and recall helps accurately identify episodes.  
- **Euthymia (stable mood)** none misclassified, critical for tracking remission states.  
- Overall high-confidence (95.8%) predictions provide reliable outputs for clinical review.

---

## Roadmap & Next Steps

- Clinical validation and prospective study design  
- Regulatory and deployment pathway consultation  
- Integration of additional physiological data sources (e.g., WESAD)  
- Multi-site external validation for generalizability  
- Real-time model deployment and monitoring systems  

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork & clone repo  
2. Create a feature branch  
3. Follow PEP 8 coding standards  
4. Add tests and update documentation  
5. Submit a Pull Request with a detailed description  

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

Lead Researcher: [d15645415@gmail.com](mailto:d15645415@gmail.com)  
Issues & Feature Requests: [GitHub Issues](https://github.com/yourusername/multimodal-BD-Detection/issues)  
Clinical Collaboration: Email for partnership discussions

---

_Last Updated: September 10, 2025_  
_Model Version: v2.0 - Production Ready_

---

Feel free to ask for any formatting adjustments or additional sections!```
# Multimodal BD Detection

## Overview  
A state-of-the-art Python pipeline for classification of mental health states—**Depression, Mania, Euthymia**—leveraging multimodal data (text, audio, video, physiological signals) via deep learning. The transformer-based architecture incorporates temporal sequence modeling and advanced data augmentation techniques to achieve near-perfect detection accuracy.

***

## Current Status (September 2025) — Deployment Ready!

| Metric             | Score  | Status      |
|--------------------|--------|-------------|
| Test Accuracy      | 99.79% | 🎉 Excellent |
| Weighted F1-Score  | 99.79% | 🎉 Excellent |
| Macro F1-Score     | 99.25% | 🎉 Excellent |
| Weighted Precision | 99.79% | 🎉 Excellent |
| Weighted Recall    | 99.79% | 🎉 Excellent |

### Per-Class Performance

| Class       | Precision | Recall  | F1-Score | Clinical Significance               |
|-------------|-----------|---------|----------|-----------------------------------|
| Depression  | 98.9%     | 97.2%   | 98.0%    | High sensitivity for detection    |
| Mania       | 99.6%     | 99.8%   | 99.7%    | Exceptional specificity           |
| Euthymia    | 100%      | 100%    | 100%     | Perfect rare-class detection      |

***

## Goal  
Develop a clinically validated, robust multimodal AI system for mental health state detection, suitable for publication and real-world clinical deployment.

***

## Features

- 🔄 Multimodal input handling: text (Twitter, Kaggle), audio & video (CMU-MOSEI), physiological signals (OBF Psychiatrist)  
- 🧠 Temporal sequence modeling with transformer architecture  
- 🔧 Rigorous data cleaning ensuring stability (NaN/Inf handling)  
- 📈 Adaptive data augmentation for robust training  
- ⚖️ Balanced loss functions and sampling addressing class imbalance, especially for rare Euthymia class  
- 📊 Comprehensive evaluation including confusion matrices, per-class metrics, and sequence-level analysis  
- 🛑 Early stopping and model checkpointing for training efficiency  
- 📱 High-confidence prediction system reducing uncertain outputs  
- 🔍 Detailed error and interpretability analysis  

***

## Project Structure

multimodal-BD-Detection/
├── data/ # Input datasets (JSONL files)
├── configs/ # Configuration settings
├── models/ # Model definitions
├── dataset/ # Custom Dataset class and preprocessing
├── quality/ # Data quality scripts
├── evaluator/ # Evaluation and visualization
├── scripts/ # Main training and evaluation pipeline
├── trainer/ # Model training logic
├── requirements.txt # Python dependencies
├── README.md # This file
├── evaluation_results.png # Evaluation plots
├── training_history.png # Training progress visualization
└── improved_results_summary.pkl # Serialized evaluation summary

text

***

## Installation & Setup

1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/multimodal-BD-Detection.git
   cd multimodal-BD-Detection
Create and activate a virtual environment:

bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
Install dependencies:

bash
pip install -r requirements.txt
Prepare your multimodal data following the schema documented in configs/config.py and place in data/.

Adjust hyperparameters and paths in configs/config.py as needed.

Usage
Run the training and evaluation pipeline:

bash
python scripts/run_pipeline.py
This will:

Train model with early stopping on validation data.

Evaluate on test set with detailed metrics and plots.

Save training history and best model checkpoint.

Clinical Significance
Depression detection with 97.2% sensitivity and 98.9% precision minimizes missed cases while controlling false alarms.

Mania detected with near-perfect precision and recall helps accurately identify episodes.

Euthymia (stable mood) none misclassified, critical for tracking remission states.

Overall high-confidence (95.8%) predictions provide reliable outputs for clinical review.

Roadmap & Next Steps
Clinical validation and prospective study design

Regulatory and deployment pathway consultation

Integration of additional physiological data sources (e.g., WESAD)

Multi-site external validation for generalizability

Real-time model deployment and monitoring systems

Contributing
Contributions are welcome! Please follow these guidelines:

Fork & clone repo

Create a feature branch

Follow PEP 8 coding standards

Add tests and update documentation

Submit a Pull Request with a detailed description

License
This project is licensed under the MIT License. See LICENSE for details.

Contact
Lead Researcher: d15645415@gmail.com
Issues & Feature Requests: GitHub Issues
Clinical Collaboration: Email for partnership discussions

Last Updated: September 10, 2025
Model Version: v2.0 - Production Ready