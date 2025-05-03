import pandas as pd
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader

class IMDbDataset(Dataset):
    """Custom Dataset class for IMDB movie reviews"""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }
        return item
    
    def __len__(self):
        return len(self.labels)

def load_dataframes(train_path, val_path, test_path):
    """Load raw data from CSV files"""
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    return train_df, val_df, test_df

def tokenize_data(tokenizer, texts):
    """Tokenize text data using the provided tokenizer"""
    return tokenizer(
        list(texts),
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors='pt'
    )

def create_dataloaders(train_df, val_df, test_df, batch_size=32):
    """Create dataloaders for train, validation and test sets"""
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Extract texts and labels
    train_texts, train_labels = train_df['text'].values, train_df['sentiment'].values
    val_texts, val_labels = val_df['text'].values, val_df['sentiment'].values
    test_texts, test_labels = test_df['text'].values, test_df['sentiment'].values
    
    # Tokenize data
    train_encodings = tokenize_data(tokenizer, train_texts)
    val_encodings = tokenize_data(tokenizer, val_texts)
    test_encodings = tokenize_data(tokenizer, test_texts)
    
    # Create datasets
    train_dataset = IMDbDataset(train_encodings, train_labels)
    val_dataset = IMDbDataset(val_encodings, val_labels)
    test_dataset = IMDbDataset(test_encodings, test_labels)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader, test_loader, tokenizer

def get_imdb_data(data_dir, batch_size=32):
    """Main function to get IMDB dataloaders"""
    train_path = f"{data_dir}/train/train.csv"
    val_path = f"{data_dir}/val/val.csv"
    test_path = f"{data_dir}/test/test.csv"
    
    train_df, val_df, test_df = load_dataframes(train_path, val_path, test_path)
    return create_dataloaders(train_df, val_df, test_df, batch_size)