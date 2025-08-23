import pickle
import logging
import torch
import json
import numpy as np
import torch.optim as optim
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_to_serializable(obj):
    """Recursively convert non-serializable objects to serializable ones."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

def main():
    # Validate config
    Config.validate()
    logger.info(f"Using device: {Config.device}")

    # Load data
    with open(Config.data_path, 'rb') as f:
        full_data = pickle.load(f)
    logger.info(f"Total samples loaded: {len(full_data)}")

    # Data analysis
    data_analysis = analyze_data_distribution(full_data)
    create_data_quality_dashboard(data_analysis)
    suggestions = suggest_hyperparameters(data_analysis)
    logger.info(f"Hyperparameter suggestions: {suggestions}")

    # Splits
    train_data, val_data, test_data = create_improved_user_split(full_data)
    logger.info(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}, Test samples: {len(test_data)}")

    # Datasets
    train_dataset = ImprovedTemporalDataset(train_data, Config, mode='train')
    val_dataset = ImprovedTemporalDataset(val_data, Config, mode='val')
    test_dataset = ImprovedTemporalDataset(test_data, Config, mode='test')

    # Loaders
    train_loader = create_improved_data_loader(train_dataset, Config.batch_size, shuffle=True, use_sampler=True)
    val_loader = create_improved_data_loader(val_dataset, Config.batch_size)
    test_loader = create_improved_data_loader(test_dataset, Config.batch_size)

    logger.info(f"DataLoaders created: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)}")

    # Model
    model = ImprovedMultimodalModel(Config).to(Config.device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model Parameters: Total={total_params:,}, Trainable={trainable_params:,}")

    # Trainer
    trainer = ImprovedTrainer(model, Config)
    history = trainer.train(train_loader, val_loader, train_dataset.get_class_weights())
    trainer.plot_training_history()
    trainer.load_best_model()

    # Evaluator
    evaluator = ImprovedEvaluator(model, Config)
    evaluation_results = evaluator.evaluate(test_loader)

    # ======================
    # Final summary logs ✅
    # ======================
    logger.info("\n" + "="*60)
    logger.info("FINAL PERFORMANCE SUMMARY")
    logger.info("="*60)
    logger.info(f"🎯 Final Test Accuracy: {evaluation_results['accuracy']:.4f}")
    logger.info(f"🎯 Final Test F1 (Weighted): {evaluation_results['f1_weighted']:.4f}")
    logger.info(f"🎯 Final Test F1 (Macro): {evaluation_results['f1_macro']:.4f}")

    if evaluation_results['f1_weighted'] >= 0.9:
        logger.info("🎉 EXCELLENT performance! Model is ready for deployment consideration.")
    elif evaluation_results['f1_weighted'] >= 0.8:
        logger.info("👍 GOOD performance! Some improvements may still be beneficial.")
    elif evaluation_results['f1_weighted'] >= 0.7:
        logger.info("👌 ACCEPTABLE performance, but significant improvements needed.")
    else:
        logger.info("⚠️  POOR performance. Major improvements required.")

    # Clinical considerations
    logger.info("\n🏥 CLINICAL CONSIDERATIONS:")
    precision_per_class = evaluation_results['precision_per_class']
    recall_per_class = evaluation_results['recall_per_class']
    class_names = ['Depression', 'Mania', 'Euthymia']

    for i, class_name in enumerate(class_names):
        if i < len(precision_per_class):
            if recall_per_class[i] < 0.8:
                logger.info(f"⚠️  Low sensitivity for {class_name} ({recall_per_class[i]:.3f}) - may miss cases")
            if precision_per_class[i] < 0.8:
                logger.info(f"⚠️  Low precision for {class_name} ({precision_per_class[i]:.3f}) - may cause false alarms")

    # Next steps
    logger.info("\n📋 NEXT STEPS:")
    logger.info("1. Review confusion matrix for specific error patterns")
    logger.info("2. Analyze high-confidence errors for model insights")
    logger.info("3. Consider clinical validation with expert review")
    logger.info("4. Implement model monitoring in production")
    logger.info("5. Plan for continuous model updates with new data")

    # Save results
    with open("improved_results_summary.pkl", "wb") as f:
        pickle.dump(evaluation_results, f)
    with open("improved_results_summary.json", "w") as f:
        json.dump(convert_to_serializable(evaluation_results), f, indent=4)

    logger.info("📂 Results saved to improved_results_summary.pkl and improved_results_summary.json")

    return evaluation_results

if __name__ == "__main__":
    results = main()
    logger.info("✅ Enhanced pipeline completed successfully!")
    logger.info(f"Final evaluation results: {results}")
    logger.info("You can now proceed with clinical validation and deployment considerations.")