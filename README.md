# Multimodal BD Detection

**Multimodal Mental Health Classification Pipeline for Bipolar Disorder Detection**

## 🎯 Overview

This repository contains a state-of-the-art Python-based pipeline for classifying mental health states (Depression, Mania, Euthymia) using multimodal data (text, audio, video) with a deep learning approach. The project leverages PyTorch to build and train a transformer-based model, incorporating temporal sequence data and advanced data augmentation techniques to achieve exceptional performance.

---

## 🏆 Current Status

**Latest Results (September 2025)** - **DEPLOYMENT READY!**

| Metric | Score | Status |
|--------|-------|--------|
| **Test Accuracy** | **99.33%** | 🎉 Excellent |
| **Weighted F1-Score** | **99.34%** | 🎉 Excellent |
| **Macro F1-Score** | **98.48%** | 🎉 Excellent |
| **Weighted Precision** | **99.34%** | 🎉 Excellent |
| **Weighted Recall** | **99.33%** | 🎉 Excellent |

### 📊 Per-Class Performance

| Class | Precision | Recall | F1-Score | Clinical Significance |
|-------|-----------|--------|----------|---------------------|
| **Depression** | 96.8% | 97.9% | 97.3% | ✅ High sensitivity for detection |
| **Mania** | 99.7% | 99.5% | 99.6% | ✅ Excellent specificity |

> **🚀 Major Improvement**: The model has achieved breakthrough performance, addressing previous limitations with depression recall and overall accuracy. This represents a significant advancement from the previous version (75.65% accuracy) to the current **99.33% accuracy**.

---

## 🎯 Goal

Develop a robust, clinically validated model for mental health detection, suitable for **publication** and **real-world clinical application**. Current performance metrics indicate **deployment readiness** pending clinical validation.

---

## ✨ Features

- 🔄 **Multimodal input processing** (text, audio, video)
- 🧠 **Temporal sequence modeling** using transformers
- 🔧 **Advanced data cleaning** and quality analysis
- 📈 **Intelligent data augmentation** strategies
- ⚖️ **Weighted loss** for class imbalance handling
- 📊 **Comprehensive evaluation** with confusion matrices, F1-scores, and sequence analysis
- 🛑 **Early stopping** and model checkpointing
- 📱 **High-confidence prediction** system (95.8% average confidence)
- 🔍 **Error analysis** and model interpretability

---

## 🛠️ Requirements

### System Requirements
- **Python**: 3.8+
- **Hardware**: GPU recommended for training (NVIDIA with CUDA support)

### Dependencies
```
torch>=1.13.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
scipy>=1.7.0
```

---

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/multimodal-BD-Detection.git
cd multimodal-BD-Detection
```

### 2. Set Up Virtual Environment
```bash
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Prepare Data
- Place your multimodal dataset (JSONL files with `user_id`, `text`, `audio`, `video`, `label`, `timestamp`) in the `data/` directory
- Ensure data format matches the expected schema in `configs/config.py`

### 5. Configure Settings
- Edit `configs/config.py` to adjust hyperparameters as needed

---

## 🚀 Usage

### Quick Start
```bash
python scripts/run_pipeline.py
```

This command will:
- ✅ Train the model with optimized hyperparameters
- ✅ Validate performance with early stopping
- ✅ Evaluate on test set with comprehensive metrics
- ✅ Generate visualization plots
- ✅ Save results in multiple formats

### Expected Output
- 📊 Training logs with epoch-wise metrics
- 📈 Evaluation plots (`evaluation_results.png`)
- 📋 Comprehensive results summary (`improved_results_summary.pkl` & `.json`)
- 📉 Training history visualization (`training_history.png`)

---

## ⚙️ Configuration

### Example Configuration
```python
# configs/config.py
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

## 📁 Project Structure

```
multimodal-BD-Detection/
├── 📂 data/                          # Input dataset (JSONL files)
├── 📂 configs/                       # Configuration files
│   └── config.py                     # Hyperparameters and settings
├── 📂 models/                        # Model definitions
│   └── multimodal_model.py          # Multimodal transformer model
├── 📂 dataset/                       # Data handling
│   └── dataset.py                    # Dataset class with sequence creation
├── 📂 quality/                       # Data quality and cleaning
│   └── quality.py                    # Data quality analyzer and cleaner
├── 📂 evaluator/                     # Evaluation metrics
│   └── evaluator.py                  # Evaluation class with plotting
├── 📂 scripts/                       # Main scripts
│   └── run_pipeline.py              # Main training and evaluation script
├── 📂 trainer/                       # Training logic
│   └── trainer.py                    # Training class with early stopping
├── 📄 requirements.txt               # Python dependencies
├── 📄 README.md                      # This file
├── 📄 improved_results_summary.pkl   # Saved evaluation results
├── 📊 evaluation_results.png         # Evaluation plots
└── 📊 training_history.png          # Training visualization
```

---

## 🏥 Clinical Significance

### 🎯 Depression Detection
- **Precision: 96.8%** - Minimizes false positives
- **Recall: 97.9%** - Excellent sensitivity for catching cases
- **Clinical Impact**: High reliability for depression screening

### 🎯 Mania Detection  
- **Precision: 99.7%** - Exceptional specificity
- **Recall: 99.5%** - Outstanding sensitivity
- **Clinical Impact**: Near-perfect identification of manic episodes

### 🔍 Model Reliability
- **Average Confidence: 95.8%** - High prediction confidence
- **Low-confidence Errors: 0** - No uncertain misclassifications
- **High-confidence Errors: 13** - Minimal confident mistakes

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### 1. Fork & Clone
```bash
git fork https://github.com/yourusername/multimodal-BD-Detection.git
git clone your-fork-url
```

### 2. Create Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 3. Development Guidelines
- Follow PEP 8 coding standards
- Add comprehensive tests for new features
- Update documentation as needed
- Ensure clinical safety considerations

### 4. Submit Changes
```bash
git commit -m "Add: your feature description"
git push origin feature/your-feature-name
```
Then open a Pull Request with detailed description.

---

## 🏁 Deployment Readiness

### ✅ Ready for Clinical Validation
The model has achieved **deployment-ready performance** with:
- 99.33% accuracy across all classes
- Exceptional precision and recall for both conditions
- High-confidence predictions (95.8% average)
- Robust error analysis and interpretability

### 🔄 Recommended Next Steps
1. 🔍 **Clinical Validation** - Expert review and validation study
2. 📊 **Prospective Testing** - Real-world clinical environment testing
3. 👩‍⚕️ **Regulatory Compliance** - FDA/regulatory pathway consideration
4. 🛡️ **Safety Protocols** - Implement human oversight systems
5. 📈 **Monitoring Systems** - Production model performance tracking
6. 🔄 **Continuous Learning** - Framework for model updates
7. 📋 **Clinical Guidelines** - Usage protocols and limitations documentation

---

## ⚠️ Important Considerations

### 🚨 Clinical Use Disclaimer
This model is designed for **research purposes** and **clinical decision support**. It should:
- ✅ Be used alongside professional clinical judgment
- ✅ Undergo institutional validation before deployment
- ✅ Include human oversight in all clinical applications
- ❌ Never replace professional psychiatric evaluation

### 🔒 Data Privacy & Security
- Ensure HIPAA compliance for clinical data
- Implement secure data handling protocols
- Follow institutional data governance policies

---

## 🚀 Future Roadmap

### Short-term (1-3 months)
- [ ] Clinical validation study design
- [ ] Regulatory pathway consultation
- [ ] Production deployment framework
- [ ] Real-time inference optimization

### Medium-term (3-6 months)  
- [ ] Multi-site validation study
- [ ] Integration with Electronic Health Records
- [ ] Mobile deployment capabilities
- [ ] Longitudinal outcome tracking

### Long-term (6+ months)
- [ ] FDA submission preparation
- [ ] International validation studies
- [ ] Additional mental health conditions
- [ ] Personalized treatment recommendations

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **xAI** for computational support and technical guidance
- **Clinical Research Community** for validation methodology insights
- **Open Source Contributors** for continuous improvements
- **Mental Health Research Community** for domain expertise

---

## 📞 Contact & Support

- **Lead Researcher**: [d15645415@gmail.com](mailto:d15645415@gmail.com)
- **Project Issues**: [GitHub Issues](https://github.com/yourusername/multimodal-BD-Detection/issues)
- **Clinical Collaboration**: Contact via email for partnership opportunities

---

## 📊 Performance Dashboard

```
🎯 CURRENT MODEL STATUS: DEPLOYMENT READY ✅

┌─────────────────────────────────────────────────────┐
│                 PERFORMANCE METRICS                 │
├─────────────────────────────────────────────────────┤
│ Overall Accuracy:           99.33% (Excellent ✅)   │
│ Weighted F1-Score:          99.34% (Excellent ✅)   │  
│ Macro F1-Score:             98.48% (Excellent ✅)   │
│ Average Confidence:         95.8%  (High ✅)        │
│ High-confidence Errors:     13     (Minimal ✅)     │
└─────────────────────────────────────────────────────┘

🏥 CLINICAL READINESS: VALIDATION PHASE ✅
📈 IMPROVEMENT FROM BASELINE: +23.68% accuracy
🚀 DEPLOYMENT RECOMMENDATION: PROCEED WITH VALIDATION
```

---

*Last Updated: September 7, 2025*
*Model Version: v2.0 - Production Ready*