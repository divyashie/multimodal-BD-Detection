import pickle
import logging
import torch

from configs.config import Config
from data.dataset import ImprovedTemporalDataset
from data.utils import (
    create_improved_user_split,
    create_improved_data_loader,
    analyze_data_distribution,
    suggest_hyperparameters,
    create_data_quality_dashboard,
)
from models.multimodal_model import ImprovedMultimodalModel
from training.trainer import ImprovedTrainer
from training.evaluator import ImprovedEvaluator

logger = logging.getLogger(__name__)


def main():
    Config.validate()
    logger.info(f"Using device: {Config.device}")

    # Load data
    with open(Config.data_path, 'rb') as f:
        full_data = pickle.load(f)

    # Data analysis
    data_analysis = analyze_data_distribution(full_data)
    create_data_quality_dashboard(data_analysis)
    suggest_hyperparameters(data_analysis)

    # Splits
    train_data, val_data, test_data = create_improved_user_split(full_data)

    # Datasets
    train_dataset = ImprovedTemporalDataset(train_data, Config, mode='train')
    val_dataset = ImprovedTemporalDataset(val_data, Config, mode='val')
    test_dataset = ImprovedTemporalDataset(test_data, Config, mode='test')

    # Loaders
    train_loader = create_improved_data_loader(train_dataset, Config.batch_size, shuffle=True, use_sampler=True)
    val_loader = create_improved_data_loader(val_dataset, Config.batch_size)
    test_loader = create_improved_data_loader(test_dataset, Config.batch_size)

    # Model
    model = ImprovedMultimodalModel(Config).to(Config.device)

    # Trainer
    trainer = ImprovedTrainer(model, Config)
    history = trainer.train(train_loader, val_loader, train_dataset.get_class_weights())
    trainer.plot_training_history()
    trainer.load_best_model()

    # Evaluator
    evaluator = ImprovedEvaluator(model, Config)
    evaluator.evaluate(test_loader)

if __name__ == "__main__":
    main()
