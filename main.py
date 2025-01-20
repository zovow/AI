import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from torchvision import models, transforms
from torch import nn, optim
from PIL import Image
import chardet
import os
from sklearn.model_selection import train_test_split


# 定义数据集类
class MultiModalDataset(Dataset):
    def __init__(self, data_list, data_dir, tokenizer, max_len=128, transform=None):
        self.data_list = data_list
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transform
        self.label_tag = {'negative': 0, 'neutral': 1, 'positive': 2}

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        guid, label_str = self.data_list[idx]
        txt_path = os.path.join(self.data_dir, f"{guid}.txt")
        text = self.read_text_file(txt_path).strip()

        img_path = os.path.join(self.data_dir, f"{guid}.jpg")
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_len
        )
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)

        if label_str is not None and label_str != 'null':
            label_id = self.label_tag[label_str]
        else:
            label_id = -1

        return input_ids, attention_mask, image, torch.tensor(label_id, dtype=torch.long)

    def read_text_file(self, txt_path):
        with open(txt_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            if encoding is None:
                encoding = 'utf-8'
            text = raw_data.decode(encoding, errors='replace')
        return text


# 数据预处理
def load_data(train_file, test_file):
    train_data_list = []
    with open(train_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            guid, label = line.split(",")
            train_data_list.append((guid, label))

    test_data_list = []
    with open(test_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            guid, label = line.split(",")
            test_data_list.append((guid, label))

    return train_data_list, test_data_list


# 模型定义
class MultiModalModel(nn.Module):
    def __init__(self, num_classes=3, text_model_name='bert-base-uncased'):
        super(MultiModalModel, self).__init__()
        self.bert = BertModel.from_pretrained(text_model_name)
        self.text_hidden_size = self.bert.config.hidden_size
        self.resnet = models.resnet50(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        fusion_hidden = 256
        self.fusion = nn.Linear(self.text_hidden_size + num_ftrs, fusion_hidden)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(fusion_hidden, num_classes)

    def forward(self, input_ids, attention_mask, images):
        text_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_text = text_outputs.last_hidden_state[:, 0, :]
        img_features = self.resnet(images)
        fusion_input = torch.cat([pooled_text, img_features], dim=1)
        fusion_output = self.fusion(fusion_input)
        fusion_output = nn.ReLU()(fusion_output)
        fusion_output = self.dropout(fusion_output)
        logits = self.classifier(fusion_output)
        return logits


# 训练和评估
def train_and_evaluate(train_data, val_data, tokenizer, transform):
    train_dataset = MultiModalDataset(train_data, 'data', tokenizer, max_len=128, transform=transform)
    val_dataset = MultiModalDataset(val_data, 'data', tokenizer, max_len=128, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    model = MultiModalModel(num_classes=3, text_model_name='bert-base-uncased').to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)

    for epoch in range(5):
        model.train()
        total_train_loss = 0.0
        for input_ids, attention_mask, images, labels in train_loader:
            input_ids, attention_mask, images, labels = input_ids.to(device), attention_mask.to(device), images.to(
                device), labels.to(device)
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0.0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for input_ids, attention_mask, images, labels in val_loader:
                input_ids, attention_mask, images, labels = input_ids.to(device), attention_mask.to(device), images.to(
                    device), labels.to(device)
                logits = model(input_ids, attention_mask, images)
                loss = criterion(logits, labels)
                total_val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        accuracy = total_correct / total_samples if total_samples else 0.0
        print(f'Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {accuracy:.4f}')

# 预测并更新文件
def predict_and_update(test_data, tokenizer, transform, test_file):
    test_dataset = MultiModalDataset(test_data, 'data', tokenizer, max_len=128, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = MultiModalModel(num_classes=3, text_model_name='bert-base-uncased').to(device)
    model.eval()
    predictions = []

    id2label = {0: 'negative', 1: 'neutral', 2: 'positive'}

    with torch.no_grad():
        for input_ids, attention_mask, images, guids in test_loader:
            input_ids, attention_mask, images = input_ids.to(device), attention_mask.to(device), images.to(device)
            logits = model(input_ids, attention_mask, images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)

    with open(test_file, 'w') as f:
        f.write("guid,tag\n")
        for (guid, _), pred in zip(test_data, predictions):
            label_str = id2label[pred]
            f.write(f"{guid},{label_str}\n")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_data, test_data = load_data('train.txt', 'test_without_label.txt')

    # 使用 train_test_split 进行随机划分
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

    train_and_evaluate(train_data, val_data, tokenizer, transform)
    predict_and_update(test_data, tokenizer, transform, 'predications.txt')