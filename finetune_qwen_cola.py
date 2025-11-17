import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    Qwen2Tokenizer, 
    Qwen2ForSequenceClassification,
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, matthews_corrcoef
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_model_and_tokenizer(model_name, num_labels=2):
    """Initialize tokenizer and model for linear probing."""
    tokenizer = Qwen2Tokenizer.from_pretrained(model_name)
    
    # Load model first
    model = Qwen2ForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="single_label_classification"
    )
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Ensure we have a valid pad token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Set pad token id in model config BEFORE resizing
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Resize token embeddings if needed (this must be done AFTER setting pad_token_id)
    if len(tokenizer) != model.config.vocab_size:
        logger.info(f"Resizing token embeddings from {model.config.vocab_size} to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
        model.config.vocab_size = len(tokenizer)
    
    # Freeze all parameters in the base model (everything except the classification head)
    for name, param in model.named_parameters():
        if 'score' not in name:  # 'score' is the classification head in Qwen2ForSequenceClassification
            param.requires_grad = False
            logger.debug(f"Frozen parameter: {name}")
        else:
            logger.info(f"Trainable parameter: {name} - Shape: {param.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    logger.info(f"Linear Probe Setup:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters (classification head): {trainable_params:,}")
    logger.info(f"  Frozen parameters (pre-trained model): {frozen_params:,}")
    logger.info(f"  Trainable ratio: {trainable_params/total_params*100:.2f}%")
    
    return tokenizer, model

def preprocess_function(examples, tokenizer):
    """Tokenize the sentences."""
    return tokenizer(
        examples["sentence"], 
        truncation=True, 
        padding=False,  # Will be done by data collator
        max_length=512
    )

def compute_metrics(eval_pred):
    """Compute accuracy and Matthews correlation coefficient."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    mcc = matthews_corrcoef(labels, predictions)
    
    return {
        "accuracy": accuracy,
        "matthews_correlation": mcc
    }

def main():
    parser = argparse.ArgumentParser(description="Linear probe Qwen2.5-0.5B on CoLA")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B", help="Model name or path")
    parser.add_argument("--output_dir", default="./qwen_cola_linear_probe", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate (higher for linear probe)")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every N steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    logger.info(f"Loading CoLA dataset...")
    # Load CoLA dataset
    dataset = load_dataset("glue", "cola")
    
    logger.info(f"Dataset info:")
    logger.info(f"Train: {len(dataset['train'])} samples")
    logger.info(f"Validation: {len(dataset['validation'])} samples")
    logger.info(f"Test: {len(dataset['test'])} samples")
    
    # Initialize tokenizer and model
    logger.info(f"Loading model and tokenizer: {args.model_name}")
    tokenizer, model = setup_model_and_tokenizer(args.model_name, num_labels=2)
    
    # Preprocess datasets
    logger.info("Tokenizing datasets...")
    tokenized_datasets = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=[col for col in dataset["train"].column_names if col != "label"]
    )
    
    # Rename label column to match model expectations
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    
    # Validate tokenized data - check for out-of-bounds token IDs
    vocab_size = len(tokenizer)
    logger.info(f"Validating tokenized data (vocab_size={vocab_size})...")
    for split in ["train", "validation"]:
        max_token_id = max([max(ids) for ids in tokenized_datasets[split]["input_ids"]])
        if max_token_id >= vocab_size:
            raise ValueError(f"Found token ID {max_token_id} >= vocab_size {vocab_size} in {split} set")
    logger.info("Token ID validation passed!")
    
    # Data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments - optimized for linear probing
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        eval_strategy="epoch",
        eval_steps=args.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="matthews_correlation",
        greater_is_better=True,
        report_to=None,  # Disable wandb/tensorboard logging
        fp16=args.fp16,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        push_to_hub=False,
        save_total_limit=2,  # Only keep best 2 checkpoints
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Verify that only classification head is trainable
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Verification - Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    # Train the model (linear probe)
    logger.info("Starting linear probing...")
    trainer.train()
    
    # Evaluate the model on validation set
    logger.info("Evaluating model on validation set...")
    eval_results = trainer.evaluate(eval_dataset=tokenized_datasets["validation"])
    logger.info(f"Validation Results:")
    logger.info(f"  Accuracy: {eval_results['eval_accuracy']:.4f}")
    logger.info(f"  Matthews Correlation: {eval_results['eval_matthews_correlation']:.4f}")
    logger.info(f"  Loss: {eval_results['eval_loss']:.4f}")
    
    # Save the final model
    logger.info(f"Saving linear probe model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # Save training and evaluation results
    results_file = os.path.join(args.output_dir, "linear_probe_results.txt")
    with open(results_file, "w") as f:
        f.write(f"Linear Probe Results on CoLA Dataset\n")
        f.write(f"=" * 40 + "\n\n")
        f.write(f"Methodology: Linear Probing\n")
        f.write(f"  - Pre-trained model: {args.model_name}\n")
        f.write(f"  - Frozen parameters: {total_params - trainable_params:,}\n")
        f.write(f"  - Trainable parameters: {trainable_params:,} (classification head only)\n")
        f.write(f"  - Trainable ratio: {trainable_params/total_params*100:.2f}%\n\n")
        
        f.write(f"Evaluation Results:\n")
        f.write(f"  Accuracy: {eval_results['eval_accuracy']:.4f}\n")
        f.write(f"  Matthews Correlation: {eval_results['eval_matthews_correlation']:.4f}\n")
        f.write(f"  Loss: {eval_results['eval_loss']:.4f}\n\n")
        
        f.write(f"Training Configuration:\n")
        f.write(f"  Training samples: {len(tokenized_datasets['train'])}\n")
        f.write(f"  Validation samples: {len(tokenized_datasets['validation'])}\n")
        f.write(f"  Epochs: {args.num_epochs}\n")
        f.write(f"  Batch size: {args.batch_size}\n")
        f.write(f"  Learning rate: {args.learning_rate}\n")
        f.write(f"  Method: Linear Probing (frozen backbone)\n")
        
        f.write(f"\nAll Arguments:\n")
        for key, value in vars(args).items():
            f.write(f"  {key}: {value}\n")
    
    logger.info(f"Results saved to {results_file}")
    logger.info("Linear probing completed!")

if __name__ == "__main__":
    main()