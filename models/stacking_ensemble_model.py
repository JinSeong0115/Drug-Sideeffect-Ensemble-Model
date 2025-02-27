import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer
from sklearn.metrics import classification_report
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW

# Custom collate function for padding
def custom_collate_fn(batch):
    features = [item["features"] for item in batch]
    labels = [item["labels"] for item in batch]

    # Pad sequences to the same length
    padded_features = pad_sequence(features, batch_first=True)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)  # Use -100 for ignored labels

    return {"features": padded_features, "labels": padded_labels}

# Train Dataset 정의
class TrainDataset(Dataset):
    def __init__(self, logits_paths, labels_path):
        self.logits = [np.load(path) for path in logits_paths]
        self.labels = np.load(labels_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        meta_features = np.concatenate([logit[idx] for logit in self.logits], axis=-1)
        label = self.labels[idx]
        return {
            "features": torch.tensor(meta_features, dtype=torch.float),
            "labels": torch.tensor(label, dtype=torch.long)
        }

tag2id = {"O": 0, "B-DRUG": 1, "I-DRUG": 2, "B-EFFECT": 3, "I-EFFECT": 4}
id2tag = {v: k for k, v in tag2id.items()}

# Test Dataset 정의
class TestDataset(Dataset):
    def __init__(self, csv_path):
        data = pd.read_csv(csv_path)
        self.tokens = data["tokens"].apply(eval).tolist()
        self.bio_tags = data["bio_tags"].apply(eval).tolist()

    def __len__(self):
        return len(self.bio_tags)

    def __getitem__(self, idx):
        bio_tag_ids = [tag2id[tag] for tag in self.bio_tags[idx]]
        return {
            "tokens": self.tokens[idx],  
            "bio_tags": bio_tag_ids      
        }

# Transformer 기반 Meta Model
class MetaTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads=4, num_layers=2, hidden_dim=128):
        super(MetaTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 2)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        output = self.classifier(x)
        return output

# 모델 로드 함수
def load_model(model_path, model_name, num_labels):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, tokenizer

# 로짓 추출 함수
def extract_logits(test_loader, model, tokenizer, device):
    logits_list = []
    with torch.no_grad():
        for batch in test_loader:
            tokens = batch["tokens"]  
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits.cpu().numpy()
            logits_list.append(logits)
    return np.concatenate(logits_list, axis=0)

# Custom collate function for Test Data
def custom_test_collate_fn(batch):
    tokens_list = [item["tokens"] for item in batch]
    bio_tags_list = [item["bio_tags"] for item in batch]

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    encoding = tokenizer(
        tokens_list,
        is_split_into_words=True,
        padding="longest",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    max_len = input_ids.size(1)
    labels = []
    for bio_tags in bio_tags_list:
        padded_labels = [-100] * max_len
        for i, tag in enumerate(bio_tags):
            if i < max_len:
                padded_labels[i] = tag
        labels.append(padded_labels)

    labels = torch.tensor(labels, dtype=torch.long)
    return {
        "tokens": tokens_list,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_logits', nargs='+', required=True, help='Paths to training logits')
    parser.add_argument('--train_labels', type=str, required=True, help='Path to training labels')
    parser.add_argument('--test_csv', type=str, required=True, help='Path to test CSV file')
    parser.add_argument('--meta_model_path', type=str, default='meta_model.pt', help='Path to save the meta model')
    parser.add_argument('--test_model_paths', nargs='+', required=True, help='List of test model paths (model_path, model_name)')
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Meta Model 초기화
    meta_model = MetaTransformer(input_dim=len(args.train_logits) * 5, num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = AdamW(meta_model.parameters(), lr=1e-4, weight_decay=0.01)

    # Train 데이터 로드
    train_dataset = TrainDataset(logits_paths=args.train_logits, labels_path=args.train_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)

    # Train Meta Model
    meta_model.train()
    epochs = 10
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            outputs = meta_model(features)
            logits = outputs.view(-1, outputs.size(-1))
            targets = labels.view(-1)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}")

    torch.save(meta_model.state_dict(), args.meta_model_path)

    # Test 데이터 로드 및 평가
    test_dataset = TestDataset(csv_path=args.test_csv)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=custom_test_collate_fn)

    test_logits = []
    for model_path, model_name in zip(args.test_model_paths[::2], args.test_model_paths[1::2]):
        model, tokenizer = load_model(model_path, model_name, num_labels=5)
        model = model.to(device)
        logits = extract_logits(test_loader, model, tokenizer, device)
        test_logits.append(logits)

    test_logits_combined = np.concatenate(test_logits, axis=-1)
    meta_model.eval()
    meta_features = torch.tensor(test_logits_combined, dtype=torch.float).to(device)
    with torch.no_grad():
        outputs = meta_model(meta_features)
        preds = torch.argmax(outputs, dim=-1)

if __name__ == "__main__":
    main()
