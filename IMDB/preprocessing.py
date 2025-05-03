import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

# Path to raw dataset
raw_train_path = 'aclImdb/train'
raw_test_path = 'aclImdb/test'

# Create output directories if they don't exist
os.makedirs('data/train', exist_ok=True)
os.makedirs('data/val', exist_ok=True)
os.makedirs('data/test', exist_ok=True)

def load_raw_data(folder_path, sentiment):
    """Load data from raw directory and return DataFrame"""
    data = []
    for filename in os.listdir(folder_path):
        with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
            text = file.read()
            data.append({'text': text, 'sentiment': sentiment})
    return pd.DataFrame(data)

# Create DataFrames from raw data
train_pos = load_raw_data(os.path.join(raw_train_path, 'pos'), 'pos')
train_neg = load_raw_data(os.path.join(raw_train_path, 'neg'), 'neg')
test_pos = load_raw_data(os.path.join(raw_test_path, 'pos'), 'pos')
test_neg = load_raw_data(os.path.join(raw_test_path, 'neg'), 'neg')

# Combine data
train_df = pd.concat([train_pos, train_neg]).sample(frac=1, random_state=42).reset_index(drop=True)
test_df = pd.concat([test_pos, test_neg]).sample(frac=1, random_state=42).reset_index(drop=True)

# Convert labels
train_df['sentiment'] = train_df['sentiment'].map({'pos': 1, 'neg': 0})
test_df['sentiment'] = test_df['sentiment'].map({'pos': 1, 'neg': 0})

# Split train into train and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df['text'].values, 
    train_df['sentiment'].values, 
    test_size=0.2, 
    random_state=42,
    stratify=train_df['sentiment'].values
)

# Create DataFrames for each set
processed_train_df = pd.DataFrame({'text': train_texts, 'sentiment': train_labels})
processed_val_df = pd.DataFrame({'text': val_texts, 'sentiment': val_labels})
processed_test_df = test_df.copy()

# Save datasets as CSV files
processed_train_df.to_csv('data/train/train.csv', index=False)
processed_val_df.to_csv('data/val/val.csv', index=False)
processed_test_df.to_csv('data/test/test.csv', index=False)

print("Data processing and saving completed successfully!")
print(f"Train set size: {len(processed_train_df)}")
print(f"Validation set size: {len(processed_val_df)}")
print(f"Test set size: {len(processed_test_df)}")