import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import re
import time
import gc
import os
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

logging.set_verbosity_error()
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
gc.collect()
torch.cuda.empty_cache()

caminho_csv = "C:/TCC2/data/usernames_20000_balanceado_embaralhado.csv"

inicio = time.time()

print("📥 Carregando dados...")
df = pd.read_csv(caminho_csv)

def normalizar_nome(name):
    name = name.lower()
    name = re.sub(r'[^a-z0-9]', '', name)
    return name

print("🔧 Pré-processando dados...")
df["Name"] = df["Name"].astype(str).apply(normalizar_nome)
df["Username"] = df["Username"].astype(str).str.strip()

print("🧪 Dividindo dados em treino e teste...")
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df[["Name", "Username"]].values.tolist(), df["Label"].values.tolist(), test_size=0.2, random_state=42
)

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

class UserProfileDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        name, username = self.texts[idx]
        inputs = tokenizer(name, username, padding="max_length", truncation=True, max_length=32, return_tensors="pt")
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float)
        }

train_dataset = UserProfileDataset(train_texts, train_labels)
test_dataset = UserProfileDataset(test_texts, test_labels)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 Dispositivo utilizado: {device}")

model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2).to(device)

optimizer = optim.AdamW(model.parameters(), lr=2e-5)
# Peso maior para a classe 1 (match), para reduzir falsos negativos
pesos = torch.tensor([1.0, 1.0]).to(device)
criterion = nn.CrossEntropyLoss(weight=pesos)

def train_model(model, train_loader, optimizer, criterion, test_loader, epochs=10, patience=2):
    print(f"📚 Iniciando treinamento por até {epochs} épocas (paciente: {patience})...")
    best_loss = float('inf')
    patience_counter = 0
    # Antes do loop de treinamento, por segurança
    print(f"🔎 Labels únicos no treino: {set(train_labels)}")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            try:
                optimizer.zero_grad()
                input_ids = batch.get("input_ids").to(device)
                attention_mask = batch.get("attention_mask").to(device)
                labels = batch.get("labels").to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels.long())

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                del input_ids, attention_mask, labels, outputs, loss
                torch.cuda.empty_cache()

            except RuntimeError as e:
                print("⚠️ Erro CUDA detectado. Limpando cache...")
                try:
                    torch.cuda.empty_cache()
                except Exception as clear_error:
                    print("Erro ao limpar cache:", clear_error)
                raise e

        avg_loss = total_loss / len(train_loader)
        print(f"🔷 Época {epoch + 1} - Perda Média: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("🔴 Early stopping ativado.")
                break

print("🚀 Iniciando treinamento...")
train_model(model, train_loader, optimizer, criterion, test_loader, epochs=10)

# model.save_pretrained("modelo_roberta_20000_balanceado_treinado")
# tokenizer.save_pretrained("modelo_roberta_20000_balanceado_treinado")
# print("✅ Modelo RoBERTa treinado salvo com sucesso!")

# Avaliação do modelo
print("🔍 Avaliando modelo...")
model.eval()
predictions, true_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()  # probabilidade da classe 1 (match)

        threshold = 0.78  # Define teu threshold aqui
        preds = (probs >= threshold).astype(int)  # Classe 1 só se for >= threshold

        predictions.extend(preds.tolist())
        true_labels.extend(labels.cpu().numpy().tolist())
        
fim = time.time()
tempo_total = fim - inicio
print(f"\n⏱️  Tempo total de treinamento: {tempo_total:.2f} segundos")
        
print("\n📊 Classification Report (RoBERTa):")
print(classification_report(true_labels, predictions))

total_acertos = sum([1 for t, p in zip(true_labels, predictions) if t == p])
total_fp = sum([1 for t, p in zip(true_labels, predictions) if t == 0 and p == 1])
total_fn = sum([1 for t, p in zip(true_labels, predictions) if t == 1 and p == 0])
print(f"\n✅ Acertos totais: {total_acertos}")
print(f"❌ Falsos positivos totais: {total_fp}")
print(f"🟧 Falsos negativos totais: {total_fn}")
print(f"📏 Tamanho total: {len(true_labels)} labels / {len(predictions)} preds")

# Gerar gráfico de acertos, falsos negativos e falsos positivos por grupo
print("\n📈 Gerando gráfico de acertos, falsos negativos e falsos positivos por grupo...")

acertos = []
falsos_positivos = []
falsos_negativos = []
grupos = []

for i in range(0, len(true_labels), 500):
    grupo_true = true_labels[i:i+500]
    grupo_pred = predictions[i:i+500]

    corretos = sum([1 for t, p in zip(grupo_true, grupo_pred) if t == p])
    fp = sum([1 for t, p in zip(grupo_true, grupo_pred) if t == 0 and p == 1])
    fn = sum([1 for t, p in zip(grupo_true, grupo_pred) if t == 1 and p == 0])

    print(f"🔹 Grupo A{len(grupos)+1} - Acertos: {corretos} | Falsos Positivos: {fp} | Falsos Negativos: {fn}")

    acertos.append(corretos)
    falsos_positivos.append(fp)
    falsos_negativos.append(fn)
    grupos.append(f"A{len(grupos) + 1}")

# Gráfico
plt.figure(figsize=(12, 6))
x = np.arange(len(grupos))
largura = 0.3
plt.bar(x - largura, acertos, width=largura, color="blue", label="Acertos")
plt.bar(x, falsos_negativos, width=largura, color="orange", label="Falsos Negativos")
plt.bar(x + largura, falsos_positivos, width=largura, color="red", label="Falsos Positivos")
plt.xlabel("Amostras (500 perfis cada)")
plt.ylabel("Quantidade")
plt.title("Acertos vs Falsos Negativos vs Falsos Positivos (RoBERTa)")
plt.xticks(x, grupos, rotation=45)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.savefig("C:/TCC2/pdf_graficos/roberta_20k_4k.pdf")

plt.show()
