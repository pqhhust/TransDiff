# preprocessing.py
import os
import pickle
import random
import torch
from torchtext.legacy import data
from transformers import BertTokenizer
from glob import glob

def create_data_folders(base_path="data"):
    for folder in ["train", "val", "test"]:
        os.makedirs(os.path.join(base_path, folder), exist_ok=True)

def load_imdb_from_directory(imdb_dir, TEXT, LABEL):
    examples = []
    for split in ["train", "test"]:
        for label in ["pos", "neg"]:
            for txt_file in glob(os.path.join(imdb_dir, split, label, "*.txt")):
                with open(txt_file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                example = data.Example.fromlist([text, label], fields=[('text', TEXT), ('label', LABEL)])
                examples.append(example)
    return examples

def preprocess_and_save_imdb(seed=42, base_path="data", imdb_dir="./aclImdb"):
    try:
        random.seed(seed)
        torch.manual_seed(seed)

        # Initialize BERT tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        max_input_length = 512

        def tokenize_and_cut(sentence):
            tokens = tokenizer.tokenize(sentence)
            return tokens[:max_input_length-2]

        # Define fields
        TEXT = data.Field(batch_first=True, use_vocab=False, tokenize=tokenize_and_cut,
                         preprocessing=tokenizer.convert_tokens_to_ids, init_token=tokenizer.cls_token_id,
                         eos_token=tokenizer.sep_token_id, pad_token=tokenizer.pad_token_id,
                         unk_token=tokenizer.unk_token_id)
        LABEL = data.LabelField(dtype=torch.float)

        # Load IMDB from local directory
        examples = load_imdb_from_directory(imdb_dir, TEXT, LABEL)
        all_data = data.Dataset(examples, fields=[('text', TEXT), ('label', LABEL)])

        # Split data
        train_data, test_data = all_data.split(split_ratio=0.8)
        train_data, valid_data = train_data.split(split_ratio=0.875)

        # Build label vocabulary
        LABEL.build_vocab(train_data)

        # Save datasets
        create_data_folders(base_path)
        with open(os.path.join(base_path, "train", "train_data.pkl"), "wb") as f:
            pickle.dump(train_data, f)
        with open(os.path.join(base_path, "val", "valid_data.pkl"), "wb") as f:
            pickle.dump(valid_data, f)
        with open(os.path.join(base_path, "test", "test_data.pkl"), "wb") as f:
            pickle.dump(test_data, f)

        print(f"Training examples: {len(train_data)}")
        print(f"Validation examples: {len(valid_data)}")
        print(f"Testing examples: {len(test_data)}")
        print(f"Datasets saved to {base_path}/{{train, val, test}}/")

    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    preprocess_and_save_imdb(imdb_dir="./aclImdb")