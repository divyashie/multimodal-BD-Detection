# multimodal-BD-Detection

Multimodal Mental Health Classification Pipeline

Overview

This repository contains a Python-based pipeline for classifying mental health states (Depression, Mania, Euthymia) using multimodal data (text, audio, video) with a deep learning approach. The project leverages PyTorch to build and train a transformer-based model, incorporating temporal sequence data and data augmentation to improve performance. The pipeline is designed for research purposes and aims to support clinical validation and deployment with continuous improvements.





Current Status: As of August 23, 2025, the pipeline is functional, achieving a test accuracy of 0.7565 and a macro F1-score of 0.6918. However, it requires enhancements to address low depression recall (0.241) and euthymia precision (0.637).



Goal: Develop a robust, clinically validated model for mental health detection, suitable for publication and real-world application.

Features





Multimodal input processing (text, audio, video).



Temporal sequence modeling using transformers.



Data cleaning, augmentation, and quality analysis.



Weighted loss for class imbalance handling.



Comprehensive evaluation with confusion matrices, F1-scores, and sequence analysis.



Early stopping and model checkpointing.

Requirements





Python: 3.8+



Dependencies:





torch>=1.13.0



numpy>=1.21.0



scikit-learn>=1.0.0



matplotlib>=3.5.0



seaborn>=0.11.0



scipy>=1.7.0



Hardware: GPU recommended for training (e.g., NVIDIA with CUDA support).

Installation





Clone the Repository:

git clone https://github.com/yourusername/multimodal-mental-health.git
cd multimodal-mental-health



Set Up a Virtual Environment:

python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate



Install Dependencies:

pip install -r requirements.txt

(Create a requirements.txt file with the listed dependencies if not already present.)



Prepare Data:





Place your multimodal dataset (e.g., JSONL files with user_id, text, audio, video, label, timestamp) in the data/ directory.



Ensure the data format matches the expected schema in configs/config.py.



Configure Settings:





Edit configs/config.py to adjust hyperparameters (e.g., sequence_length, batch_size, device).

Usage

Running the Pipeline





Train and Evaluate the Model:

python scripts/run_pipeline.py





This script trains the model, validates it, and evaluates performance on the test set.



Results are saved as improved_results_summary.pkl and improved_results_summary.json.



Expected Output:





Training logs (e.g., epoch-wise loss, F1-scores).



Evaluation plots (evaluation_results.png).



Summary metrics in the console and saved files.

Example Configuration

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

File Structure

multimodal-mental-health/
├── data/                  # Input dataset (e.g., JSONL files)
├── configs/               # Configuration files
│   └── config.py          # Hyperparameters and settings
├── models/                # Model definitions
│   └── multimodal_model.py # Multimodal transformer model
├── dataset/               # Data handling
│   └── dataset.py         # Dataset class with sequence creation
├── quality/               # Data quality and cleaning
│   └── quality.py         # Data quality analyzer and cleaner
├── evaluator/             # Evaluation metrics
│   └── evaluator.py       # Evaluation class with plotting
├── scripts/               # Main scripts
│   └── run_pipeline.py    # Main training and evaluation script
├── trainer/               # Training logic
│   └── trainer.py         # Training class with early stopping
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── improved_results_summary.pkl  # Saved evaluation results
└── evaluation_results.png  # Evaluation plots

Contributing





Fork the Repository:





Create a personal fork and clone it locally.



Create a Branch:

git checkout -b feature/your-feature-name



Make Changes:





Follow the coding style (PEP 8) and add tests if applicable.



Commit Changes:

git commit -m "Add your feature description"



Push and Submit a Pull Request:





Push to your fork and open a PR against the main repository.



Code Review:





Expect feedback and iterate as needed.

Known Issues and Limitations





Data Loss: 58.4% data removal due to aggressive cleaning, resulting in 0 users with 2+ posts.



Class Imbalance: Low depression recall (0.241) and low euthymia precision (0.637) indicate performance gaps.



Scalability: Current implementation assumes a fixed dataset; real-time updates need integration.

Future Work





Enhance data cleaning to retain more multi-post users.



Implement class-specific augmentation for depression.



Add temporal feature engineering (e.g., time gaps).



Integrate learning rate scheduling for better convergence.



Conduct clinical validation with mental health experts.

License

This project is licensed under the MIT License. See the LICENSE file for details (create one if not present).

Acknowledgments





Built with support from xAI and community feedback.



Inspired by advances in multimodal learning for mental health research.

Contact

For questions or collaboration, contact d15645415@gmail.com.