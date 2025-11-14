#!/usr/bin/env python3
"""
Fine-tune Qwen2.5-0.5B on CoLA dataset for sequence classification.
CoLA (Corpus of Linguistic Acceptability) is a binary classification task.
"""

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
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, matthews_corrcoef
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_model_and_tokenizer(model_name, num_labels=2, use_lora=True, lora_r=16, lora_alpha=32):
    """Initialize tokenizer and model for sequence classification with LoRA."""
    tokenizer = Qwen2Tokenizer.from_pretrained(model_name)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Ensure we have a valid pad token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = Qwen2ForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="single_label_classification"
    )
    
    # Set pad token id in model config
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Resize token embeddings if needed
    model.resize_token_embeddings(len(tokenizer))
    
    if use_lora:
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_r,  # rank
            lora_alpha=lora_alpha,  # scaling factor
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        # Apply LoRA to the model
        model = get_peft_model(model, lora_config)
        logger.info(f"LoRA applied with rank {lora_r} and alpha {lora_alpha}")
        
        # Print trainable parameters
        model.print_trainable_parameters()
    
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
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-0.5B on CoLA")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B", help="Model name or path")
    parser.add_argument("--output_dir", default="./qwen_cola_finetuned", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every N steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    parser.add_argument("--use_lora", action="store_true", default=True, help="Use LoRA for efficient fine-tuning")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha scaling factor")
    
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
    tokenizer, model = setup_model_and_tokenizer(
        args.model_name, 
        num_labels=2, 
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha
    )
    
    # Preprocess datasets
    logger.info("Tokenizing datasets...")
    tokenized_datasets = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=[col for col in dataset["train"].column_names if col != "label"]
    )
    
    # Rename label column to match model expectations
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    
    # Data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        eval_strategy="epoch",  # Evaluate at the end of each epoch
        eval_steps=args.eval_steps,
        report_to=None,  # Disable wandb/tensorboard logging
        fp16=args.fp16,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        push_to_hub=False,
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
    
    # Print model info
    logger.info(f"Model parameters: {model.num_parameters():,}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Evaluate the model on validation set
    logger.info("Evaluating model on validation set...")
    eval_results = trainer.evaluate(eval_dataset=tokenized_datasets["validation"])
    logger.info(f"Validation Results:")
    logger.info(f"  Accuracy: {eval_results['eval_accuracy']:.4f}")
    logger.info(f"  Matthews Correlation: {eval_results['eval_matthews_correlation']:.4f}")
    logger.info(f"  Loss: {eval_results['eval_loss']:.4f}")
    
    # Save the final model
    logger.info(f"Saving model to {args.output_dir}")
    if args.use_lora:
        # Save LoRA adapter
        model.save_pretrained(args.output_dir)
        logger.info("LoRA adapter saved")
        
        # Also save merged model (traditional full model without adapter)
        merged_output_dir = args.output_dir + "_merged"
        logger.info(f"Merging LoRA weights and saving traditional model to {merged_output_dir}")
        
        # Merge LoRA weights into base model
        merged_model = model.merge_and_unload()
        
        # Save the merged model
        merged_model.save_pretrained(merged_output_dir)
        tokenizer.save_pretrained(merged_output_dir)
        logger.info("Traditional merged model saved")
        
        # Also save config file for easier loading
        merged_model.config.save_pretrained(merged_output_dir)
        
    else:
        trainer.save_model()
    
    # Save tokenizer in main output dir
    tokenizer.save_pretrained(args.output_dir)
    
    # Save training and evaluation results
    results_file = os.path.join(args.output_dir, "results.txt")
    with open(results_file, "w") as f:
        f.write(f"Fine-tuning Results on CoLA Dataset\n")
        f.write(f"=" * 40 + "\n\n")
        f.write(f"Evaluation Results:\n")
        f.write(f"  Accuracy: {eval_results['eval_accuracy']:.4f}\n")
        f.write(f"  Matthews Correlation: {eval_results['eval_matthews_correlation']:.4f}\n")
        f.write(f"  Loss: {eval_results['eval_loss']:.4f}\n\n")
        f.write(f"Saved Models:\n")
        if args.use_lora:
            f.write(f"  LoRA adapter: {args.output_dir}\n")
            f.write(f"  Traditional merged model: {args.output_dir}_merged\n")
        else:
            f.write(f"  Full model: {args.output_dir}\n")
        f.write(f"\n")
        f.write(f"Training Configuration:\n")
        f.write(f"  Training samples: {len(tokenized_datasets['train'])}\n")
        f.write(f"  Validation samples: {len(tokenized_datasets['validation'])}\n")
        f.write(f"  Epochs: {args.num_epochs}\n")
        f.write(f"  Batch size: {args.batch_size}\n")
        f.write(f"  Learning rate: {args.learning_rate}\n")
        f.write(f"  Use LoRA: {args.use_lora}\n")
        if args.use_lora:
            f.write(f"  LoRA rank: {args.lora_r}\n")
            f.write(f"  LoRA alpha: {args.lora_alpha}\n")
        f.write(f"\nAll Arguments:\n")
        for key, value in vars(args).items():
            f.write(f"  {key}: {value}\n")
    
    logger.info(f"Results saved to {results_file}")
    logger.info("Training completed!")

if __name__ == "__main__":
    main()