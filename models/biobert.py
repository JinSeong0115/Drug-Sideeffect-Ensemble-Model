import argparse
import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW, get_cosine_schedule_with_warmup
from transformers import DataCollatorForTokenClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report

# BIO Tag Mapping
tag2id = {"O": 0, "B-DRUG": 1, "I-DRUG": 2, "B-EFFECT": 3, "I-EFFECT": 4}
id2tag = {v: k for k, v in tag2id.items()}

class BIODataset(Dataset):
    def __init__(self, texts, tags, tokenizer, max_len):
        self.texts = texts
        self.tags = tags
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.texts[idx]
        tags = self.tags[idx]
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        labels = [-100] * self.max_len
        word_ids = encoding.word_ids()
        for i, word_id in enumerate(word_ids):
            if word_id is not None and word_id < len(tags):
                labels[i] = tag2id[tags[word_id]]
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

def load_data(data_path, tokenizer, max_len=512, batch_size=16):
    data = pd.read_csv(data_path)
    texts = data['tokens'].apply(eval).tolist()
    tags = data['bio_tags'].apply(eval).tolist()
    dataset = BIODataset(texts, tags, tokenizer, max_len)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)
    return data_loader

def train_model(model, train_loader, valid_loader, optimizer, scheduler, loss_fn, device, epochs=10, patience=5, model_path="best_model.pt"):
    best_loss = float("inf")
    no_improve = 0
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(valid_loader)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            no_improve = 0
            torch.save(model.state_dict(), model_path)
            print(f"Model saved at {model_path}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping triggered.")
                break
    model.load_state_dict(torch.load(model_path))
    return model

def evaluate_model(model, test_loader, device, output_logits):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits.cpu().numpy()
            all_logits.append(logits)
            all_labels.extend(labels)
    logits_array = np.concatenate(all_logits, axis=0)
    labels_array = np.array(all_labels)
    
    flat_preds, flat_labels = [], []
    for pred, lab in zip(np.argmax(logits_array, axis=-1), labels_array):
        for p, l in zip(pred, lab):
            if l != -100:
                flat_preds.append(p)
                flat_labels.append(l)
    print("\nClassification Report:")
    print(classification_report([id2tag[l] for l in flat_labels], [id2tag[p] for p in flat_preds]))
    
    np.save(output_logits, logits_array)
    print(f"Logits saved to {output_logits}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, required=True, help='Path to training data')
    parser.add_argument('--valid_data', type=str, required=True, help='Path to validation data')
    parser.add_argument('--test_data', type=str, required=True, help='Path to test data')
    parser.add_argument('--model_path', type=str, default='biobert_best_model.pt', help='Path to save the best model')
    parser.add_argument('--output_logits', type=str, default='biobert_logits.npy', help='Path to save logits')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    train_loader = load_data(args.train_data, tokenizer)
    valid_loader = load_data(args.valid_data, tokenizer)
    test_loader = load_data(args.test_data, tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForTokenClassification.from_pretrained("dmis-lab/biobert-base-cased-v1.1", num_labels=len(tag2id)).to(device)
    
    optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * len(train_loader) * 100), num_training_steps=len(train_loader) * 100)
    loss_fn = torch.nn.CrossEntropyLoss()

    trained_model = train_model(model, train_loader, valid_loader, optimizer, scheduler, loss_fn, device, epochs=10, patience=5, model_path=args.model_path)
    
    evaluate_model(trained_model, test_loader, device, args.output_logits)

if __name__ == "__main__":
    main()
