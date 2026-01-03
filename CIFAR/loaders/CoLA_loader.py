from transformers import Qwen2Tokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset

def CoLALoaders(batch_size=32, num_workers=8, args=None, val_ratio=0.1, seed=42):
    dataset = load_dataset("glue", "sst2")
    tokenizer = Qwen2Tokenizer.from_pretrained(args.pretrained_dir)
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # Ensure we have a valid pad token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    def tokenize_fn(batch):
        return tokenizer(batch['sentence'], truncation=False, padding='max_length', max_length=64)
    
    tokenized = dataset.map(tokenize_fn, batched=True)
    tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    # train_val = tokenized['train'].train_test_split(test_size=val_ratio, seed=seed)
    train_dataset = tokenized['train']
    val_dataset = tokenized['validation']
    test_dataset = tokenized['validation']

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
    