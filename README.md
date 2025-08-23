# Multimodal Mental Health Classification (multimodal-BD-Detection)

A **deep learning pipeline** for classifying mental health states—**Depression, Mania, Euthymia**—using **multimodal data** (text, audio, video).  
The project is built with **PyTorch** and designed for **research and clinical validation**.

---

## 🚀 Overview
This repository provides a Python-based pipeline that integrates:
- Transformer-based **temporal sequence modeling**
- **Data augmentation** and quality analysis
- Handling of **class imbalance**
- Comprehensive **evaluation metrics**

**Current Status (Aug 23, 2025):**
- ✅ Functional pipeline
- 📊 Test Accuracy: **0.7565**
- 📊 Macro F1-score: **0.6918**
- ⚠️ Gaps: Low depression recall (**0.241**) and low euthymia precision (**0.637**)

**Goal:** Develop a robust, clinically validated model for mental health detection suitable for **publication** and **real-world application**.

---

## ✨ Features
- Multimodal input (text, audio, video)  
- Transformer-based temporal modeling  
- Data cleaning, augmentation, and quality checks  
- Weighted loss for class imbalance  
- Evaluation: F1-scores, confusion matrices, sequence-level metrics  
- Early stopping & model checkpointing  

---

## ⚙️ Requirements
- **Python**: 3.8+
- **Dependencies**:
  - `torch>=1.13.0`
  - `numpy>=1.21.0`
  - `scikit-learn>=1.0.0`
  - `matplotlib>=3.5.0`
  - `seaborn>=0.11.0`
  - `scipy>=1.7.0`
- **Hardware**: GPU recommended (NVIDIA + CUDA)

---

## 📥 Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/multimodal-mental-health.git
   cd multimodal-mental-health
   ```

2. **Set Up Virtual Environment**
   ```bash
   python -m venv myenv
   source myenv/bin/activate   # Windows: myenv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare Data**
   - Place your dataset in `data/` (JSONL format with `user_id`, `text`, `audio`, `video`, `label`, `timestamp`).
   - Ensure the format matches the schema in `configs/config.py`.

5. **Configure Settings**
   - Edit `configs/config.py` for hyperparameters (e.g., sequence length, batch size, device).

---

## ▶️ Usage

### Train & Evaluate
```bash
python scripts/run_pipeline.py
```
- Trains, validates, and evaluates the model.
- Results saved as:
  - `improved_results_summary.pkl`
  - `improved_results_summary.json`
  - Evaluation plots: `evaluation_results.png`

### Expected Outputs
- Training logs (loss, F1-scores per epoch)
- Evaluation metrics & confusion matrices
- Console + saved summaries

---

## 🛠 Example Config (`configs/config.py`)
```python
class Config:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 32
    sequence_length = 8
    min_sequence_length = 3
    num_classes = 3

    text_dim = 768
    audio_dim = 128
    video_dim = 512
    hidden_dim = 256

    num_heads = 4
    num_transformer_layers = 2
    dropout_rate = 0.5

    use_data_augmentation = True
    noise_factor = 0.1
    dropout_augmentation_rate = 0.2
```

---

## 📂 File Structure
```bash
multimodal-mental-health/
├── data/                  # Input dataset
├── configs/               # Config files (e.g., config.py)
├── models/                # Model definitions
│   └── multimodal_model.py
├── dataset/               # Dataset handling
│   └── dataset.py
├── quality/               # Data quality checks
│   └── quality.py
├── evaluator/             # Evaluation utilities
│   └── evaluator.py
├── scripts/               # Training scripts
│   └── run_pipeline.py
├── trainer/               # Training logic
│   └── trainer.py
├── requirements.txt
├── README.md
├── improved_results_summary.pkl
└── evaluation_results.png
```

---

## 🤝 Contributing

1. **Fork the Repository**
   ```bash
   git checkout -b feature/your-feature
   ```

2. **Make Changes**
   - Follow PEP 8.
   - Add tests where possible.

3. **Commit & Push**
   ```bash
   git commit -m "Add feature description"
   git push origin feature/your-feature
   ```

4. **Open a Pull Request**

---

## ⚠️ Known Issues
- **Data Loss**: ~58.4% removed due to aggressive cleaning (0 users with ≥2 posts remain).
- **Class Imbalance**: Depression recall (0.241) and euthymia precision (0.637) remain low.
- **Scalability**: Assumes static dataset; real-time support pending.

---

## 🔮 Future Work
- Improve cleaning to preserve multi-post users.
- Class-specific augmentation (esp. for depression).
- Temporal feature engineering (e.g., time gaps).
- Learning rate scheduling for convergence.
- Clinical validation with experts.

---

## 📜 License
This project is licensed under the MIT License. See the LICENSE file.

---

## 🙏 Acknowledgments
- Built with support from xAI and community contributions.
- Inspired by recent advances in multimodal mental health research.

---

## 📧 Contact
For questions or collaborations: d15645415@gmail.com