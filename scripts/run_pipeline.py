"""
Enhanced BD Detection Pipeline
Complete implementation integrating all improvements
"""

import pickle
import logging
import torch
import json
import numpy as np
import os
from datetime import datetime
import torch.optim as optim
from typing import Dict, Tuple, Any, List

from configs.config import Config
from data.dataset import (
    ImprovedTemporalDataset, 
    create_improved_data_loader, 
    create_improved_user_split,
    analyze_data_distribution,
    suggest_hyperparameters,
    create_data_quality_dashboard
)
from models.multimodal_model import ImprovedMultimodalModel
from training.trainer import ImprovedTrainer
from training.evaluator import ImprovedEvaluator

# Setup enhanced logging
def setup_logging():
    """Setup comprehensive logging system"""
    log_filename = f'enhanced_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    # Create logger with specific name
    logger = logging.getLogger('BD_Detection_Pipeline')
    logger.info(f"Logging initialized. Log file: {log_filename}")
    
    return logger

def convert_to_serializable(obj):
    """Convert non-serializable objects to serializable ones"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif hasattr(obj, '__dict__'):
        return convert_to_serializable(obj.__dict__)
    return obj

def setup_environment(logger):
    """Setup environment and validate configuration"""
    logger.info("="*60)
    logger.info("🚀 ENHANCED BD DETECTION PIPELINE INITIALIZATION")
    logger.info("="*60)
    
    try:
        # Validate configuration
        Config.validate()
        logger.info("✅ Configuration validation passed")
    except Exception as e:
        logger.error(f"❌ Configuration validation failed: {e}")
        raise
    
    # Create necessary directories
    directories = ['results', 'models', 'plots', 'logs', 'checkpoints', 'visualizations']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"📁 Created/verified directory: {directory}")
    
    # Log system information
    logger.info(f"🖥️  Device: {Config.device}")
    logger.info(f"📊 Data path: {Config.data_path}")
    logger.info(f"🔄 Batch size: {Config.batch_size}")
    logger.info(f"🎯 Learning rate: {Config.learning_rate}")
    logger.info(f"📈 Max epochs: {Config.num_epochs}")
    
    if torch.cuda.is_available():
        logger.info(f"🎮 CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"💾 CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        logger.info(f"🔥 PyTorch version: {torch.__version__}")
    else:
        logger.info("🖥️  Running on CPU")
    
    return True

def load_and_analyze_data(logger) -> Tuple[List, Dict]:
    """Load data and perform comprehensive analysis"""
    logger.info("\n" + "="*50)
    logger.info("📊 DATA LOADING AND ANALYSIS")
    logger.info("="*50)
    
    try:
        # Load data
        logger.info(f"Loading data from: {Config.data_path}")
        with open(Config.data_path, 'rb') as f:
            full_data = pickle.load(f)
        
        logger.info(f"✅ Successfully loaded {len(full_data)} samples")
        
        # Perform comprehensive data analysis
        logger.info("🔍 Analyzing data distribution...")
        data_analysis = analyze_data_distribution(full_data)
        
        # Create data quality dashboard
        logger.info("📈 Creating data quality dashboard...")
        create_data_quality_dashboard(data_analysis)
        
        # Log key statistics
        logger.info("\n📊 DATA STATISTICS:")
        if 'class_distribution' in data_analysis:
            for class_name, count in data_analysis['class_distribution'].items():
                logger.info(f"   {class_name}: {count} samples")
        
        if 'temporal_stats' in data_analysis:
            stats = data_analysis['temporal_stats']
            logger.info(f"   Average sequence length: {stats.get('avg_length', 'N/A'):.2f}")
            logger.info(f"   Min sequence length: {stats.get('min_length', 'N/A')}")
            logger.info(f"   Max sequence length: {stats.get('max_length', 'N/A')}")
        
        # Get hyperparameter suggestions
        logger.info("🎛️  Getting hyperparameter suggestions...")
        suggestions = suggest_hyperparameters(data_analysis)
        logger.info("💡 HYPERPARAMETER SUGGESTIONS:")
        for param, value in suggestions.items():
            logger.info(f"   {param}: {value}")
        
        return full_data, data_analysis
        
    except FileNotFoundError:
        logger.error(f"❌ Data file not found: {Config.data_path}")
        raise
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        raise

def create_datasets_and_loaders(full_data, logger) -> Tuple:
    """Create datasets and data loaders with improved splitting"""
    logger.info("\n" + "="*50)
    logger.info("🔄 DATASET CREATION AND DATA LOADING")
    logger.info("="*50)
    
    try:
        # Create user-based splits to prevent data leakage
        logger.info("👥 Creating user-based data splits...")
        train_data, val_data, test_data = create_improved_user_split(full_data)
        
        logger.info("✅ Data splits created:")
        logger.info(f"   📚 Train: {len(train_data)} samples ({len(train_data)/len(full_data)*100:.1f}%)")
        logger.info(f"   📖 Validation: {len(val_data)} samples ({len(val_data)/len(full_data)*100:.1f}%)")
        logger.info(f"   📋 Test: {len(test_data)} samples ({len(test_data)/len(full_data)*100:.1f}%)")
        
        # Create datasets
        logger.info("🏗️  Creating datasets...")
        train_dataset = ImprovedTemporalDataset(train_data, Config, mode='train')
        val_dataset = ImprovedTemporalDataset(val_data, Config, mode='val')
        test_dataset = ImprovedTemporalDataset(test_data, Config, mode='test')
        
        # Create data loaders
        logger.info("⚡ Creating data loaders...")
        train_loader = create_improved_data_loader(
            train_dataset, 
            Config.batch_size, 
            shuffle=True, 
            use_sampler=True
        )
        val_loader = create_improved_data_loader(val_dataset, Config.batch_size)
        test_loader = create_improved_data_loader(test_dataset, Config.batch_size)
        
        logger.info("✅ Data loaders created:")
        logger.info(f"   🚂 Train: {len(train_loader)} batches")
        logger.info(f"   🚃 Validation: {len(val_loader)} batches")
        logger.info(f"   🚄 Test: {len(test_loader)} batches")
        
        # Get class weights for handling imbalanced data
        logger.info("⚖️  Computing class weights...")
        class_weights = train_dataset.get_class_weights()
        logger.info(f"   Class weights: {class_weights}")
        
        return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, class_weights
        
    except Exception as e:
        logger.error(f"❌ Error creating datasets: {e}")
        raise

def create_and_setup_model(logger) -> ImprovedMultimodalModel:
    """Create and setup the enhanced model"""
    logger.info("\n" + "="*50)
    logger.info("🧠 MODEL CREATION AND SETUP")
    logger.info("="*50)
    
    try:
        # Create model
        logger.info("🏗️  Creating enhanced multimodal model...")
        model = ImprovedMultimodalModel(Config).to(Config.device)
        
        # Log model information
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info("✅ Model created successfully:")
        logger.info(f"   📊 Total parameters: {total_params:,}")
        logger.info(f"   🎯 Trainable parameters: {trainable_params:,}")
        logger.info(f"   💾 Model size (MB): {total_params * 4 / (1024**2):.2f}")
        logger.info(f"   🎮 Device: {next(model.parameters()).device}")
        
        # Log model architecture summary if available
        if hasattr(model, 'get_architecture_summary'):
            summary = model.get_architecture_summary()
            logger.info("🏗️  Model Architecture:")
            for component, details in summary.items():
                logger.info(f"   {component}: {details}")
        
        return model
        
    except Exception as e:
        logger.error(f"❌ Error creating model: {e}")
        raise

def train_model(model, train_loader, val_loader, class_weights, logger) -> Tuple[ImprovedTrainer, Dict]:
    """Train the model with enhanced trainer"""
    logger.info("\n" + "="*50)
    logger.info("🚀 MODEL TRAINING")
    logger.info("="*50)
    
    try:
        # Initialize trainer
        logger.info("🏋️  Initializing enhanced trainer...")
        trainer = ImprovedTrainer(model, Config)
        
        # Start training
        logger.info("🚀 Starting training process...")
        logger.info(f"   📈 Max epochs: {Config.num_epochs}")
        logger.info(f"   📚 Training batches: {len(train_loader)}")
        logger.info(f"   📖 Validation batches: {len(val_loader)}")
        
        # Train model with class weights
        history = trainer.train(train_loader, val_loader, class_weights)
        
        # Plot training curves
        logger.info("📊 Creating training visualization...")
        trainer.plot_training_history()
        
        # Load best model
        logger.info("🏆 Loading best performing model...")
        trainer.load_best_model()
        
        logger.info("✅ Training completed successfully!")
        
        return trainer, history
        
    except Exception as e:
        logger.error(f"❌ Error during training: {e}")
        raise

def evaluate_model(model, test_loader, logger) -> Dict:
    """Evaluate the trained model"""
    logger.info("\n" + "="*50)
    logger.info("🎯 MODEL EVALUATION")
    logger.info("="*50)
    
    try:
        # Initialize evaluator
        logger.info("🔍 Initializing enhanced evaluator...")
        evaluator = ImprovedEvaluator(model, Config)
        
        # Perform evaluation
        logger.info("📊 Evaluating model performance...")
        evaluation_results = evaluator.evaluate(test_loader)
        
        return evaluation_results
        
    except Exception as e:
        logger.error(f"❌ Error during evaluation: {e}")
        raise

def log_final_results(evaluation_results, logger):
    """Log comprehensive final results"""
    logger.info("\n" + "="*60)
    logger.info("🎉 FINAL PERFORMANCE SUMMARY")
    logger.info("="*60)
    
    # Main metrics
    logger.info(f"🎯 Final Test Accuracy: {evaluation_results['accuracy']:.4f}")
    logger.info(f"🎯 Final Test F1 (Weighted): {evaluation_results['f1_weighted']:.4f}")
    logger.info(f"🎯 Final Test F1 (Macro): {evaluation_results['f1_macro']:.4f}")
    
    # Performance assessment
    f1_weighted = evaluation_results['f1_weighted']
    if f1_weighted >= 0.9:
        logger.info("🎉 EXCELLENT performance! Model is ready for deployment consideration.")
    elif f1_weighted >= 0.8:
        logger.info("👍 GOOD performance! Some improvements may still be beneficial.")
    elif f1_weighted >= 0.7:
        logger.info("👌 ACCEPTABLE performance, but significant improvements needed.")
    else:
        logger.info("⚠️  POOR performance. Major improvements required.")
    
    # Clinical considerations
    logger.info("\n🏥 CLINICAL CONSIDERATIONS:")
    precision_per_class = evaluation_results.get('precision_per_class', [])
    recall_per_class = evaluation_results.get('recall_per_class', [])
    class_names = ['Depression', 'Mania', 'Euthymia']
    
    for i, class_name in enumerate(class_names):
        if i < len(precision_per_class) and i < len(recall_per_class):
            precision = precision_per_class[i]
            recall = recall_per_class[i]
            logger.info(f"   {class_name}:")
            logger.info(f"     Precision: {precision:.3f}")
            logger.info(f"     Recall: {recall:.3f}")
            
            if recall < 0.8:
                logger.info(f"     ⚠️  Low sensitivity - may miss {class_name.lower()} cases")
            if precision < 0.8:
                logger.info(f"     ⚠️  Low precision - may cause false {class_name.lower()} alarms")
    
    # Next steps
    logger.info("\n📋 RECOMMENDED NEXT STEPS:")
    logger.info("1. 🔍 Review confusion matrix for specific error patterns")
    logger.info("2. 📊 Analyze high-confidence errors for model insights")
    logger.info("3. 👩‍⚕️ Consider clinical validation with expert review")
    logger.info("4. 📈 Implement model monitoring in production environment")
    logger.info("5. 🔄 Plan for continuous model updates with new data")
    logger.info("6. 🛡️  Establish safety protocols and human oversight")
    logger.info("7. 📋 Document model limitations and usage guidelines")

def save_results(comprehensive_results, history, logger: logging.Logger):
    """
    Safely save evaluation results and training history to disk.
    Converts any class attributes (mappingproxy) to regular dicts to avoid pickling errors.
    """

    safe_results = comprehensive_results.copy() if isinstance(comprehensive_results, dict) else comprehensive_results

    # Convert Config object to dict if present
    if 'config' in safe_results:
        cfg = safe_results['config']
        # Only include non-dunder attributes
        safe_results['config'] = {k: v for k, v in cfg.__dict__.items() if not k.startswith('__')}

    # Save evaluation results
    try:
        with open('evaluation_results.pkl', 'wb') as f:
            pickle.dump(safe_results, f)
        logger.info("✅ Evaluation results saved successfully")
    except Exception as e:
        logger.error(f"❌ Failed to save evaluation results: {e}")

    # Save training history
    try:
        with open('training_history.pkl', 'wb') as f:
            pickle.dump(history, f)
        logger.info("✅ Training history saved successfully")
    except Exception as e:
        logger.error(f"❌ Failed to save training history: {e}")


def main():
    """Main pipeline execution function"""
    # Setup logging
    logger = setup_logging()
    
    try:
        # 1. Setup environment
        setup_environment(logger)
        
        # 2. Load and analyze data
        full_data, data_analysis = load_and_analyze_data(logger)
        
        # 3. Create datasets and loaders
        (train_dataset, val_dataset, test_dataset, 
         train_loader, val_loader, test_loader, class_weights) = create_datasets_and_loaders(full_data, logger)
        
        # 4. Create model
        model = create_and_setup_model(logger)
        
        # 5. Train model
        trainer, history = train_model(model, train_loader, val_loader, class_weights, logger)
        
        # 6. Evaluate model
        evaluation_results = evaluate_model(model, test_loader, logger)
        
        # 7. Log final results
        log_final_results(evaluation_results, logger)
        
        # 8. Save results
        save_results(evaluation_results, history, logger)
        
        # Final success message
        logger.info("\n" + "="*60)
        logger.info("✅ ENHANCED BD DETECTION PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info("🎯 You can now proceed with clinical validation and deployment considerations.")
        logger.info("📊 Check the generated plots and saved results for detailed analysis.")
        
        return evaluation_results, history
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        logger.error("💡 Please check the logs for detailed error information.")
        raise
    
    finally:
        logger.info("🏁 Pipeline execution finished.")

if __name__ == "__main__":
    results, history = main()
    print("\n" + "="*60)
    print("🎉 ENHANCED BD DETECTION PIPELINE EXECUTION COMPLETE")
    print("="*60)
    print(f"📊 Final Results Summary:")
    print(f"   Accuracy: {results.get('accuracy', 'N/A'):.4f}")
    print(f"   F1 (Weighted): {results.get('f1_weighted', 'N/A'):.4f}")
    print(f"   F1 (Macro): {results.get('f1_macro', 'N/A'):.4f}")
    print("📂 Check saved files for detailed results and visualizations.")