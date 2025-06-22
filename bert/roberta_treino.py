# ------------------------------------------------------------
# RoBERTa – Treino + validação + busca de limiar automaticamente
# ------------------------------------------------------------
import pandas as pd
import torch
import numpy as np
import random
import re
import time
import gc
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import (RobertaTokenizer,
                          RobertaForSequenceClassification,
                          logging)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import time

start_prog = time.time()
# ---------- 1. reproducibilidade ----------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
logging.set_verbosity_error()
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
gc.collect()
torch.cuda.empty_cache()

# ---------- 2. dados ----------
CSV_PATH = "C:/TCC2/data/dataset_v3.csv"
print("📥  Lendo csv…")
df = pd.read_csv(CSV_PATH)


def norm(s): return re.sub(r'[^a-z0-9]', '', str(s).lower())


df["Name"] = df["Name"].apply(norm)
df["Username"] = df["Username"].str.strip().str.lower()
df["Label"] = df["Label"].astype(int)

X = df[["Name", "Username"]].values
y = df["Label"].values

# ------ 2.1  split -> treino / validação / teste (80 / 10 / 10) ------
X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.10,
                                                random_state=SEED, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.1111,
                                                  random_state=SEED, stratify=y_tmp)
print(
    f"🔎  Tamanhos  train={len(y_train)}  val={len(y_val)}  test={len(y_test)}")

tok = RobertaTokenizer.from_pretrained("roberta-base")


class ProfileDS(Dataset):
    def __init__(self, feats, labels): self.X, self.y = feats, labels
    def __len__(self): return len(self.X)

    def __getitem__(self, i):
        n, u = self.X[i]
        enc = tok(n, u, truncation=True, max_length=32,
                  padding='max_length', return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": torch.tensor(self.y[i], dtype=torch.long)}


ds_train = ProfileDS(X_train, y_train)
ds_val = ProfileDS(X_val,   y_val)
ds_test = ProfileDS(X_test,  y_test)

# ------ 2.2  sampler balanceado  ------
class_counts = np.bincount(y_train)
sample_w = [1/class_counts[label] for label in y_train]
sampler = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

train_loader = DataLoader(ds_train, batch_size=8, sampler=sampler)
val_loader = DataLoader(ds_val,   batch_size=8, shuffle=False)
test_loader = DataLoader(ds_test,  batch_size=8, shuffle=False)

# ---------- 3. modelo ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("💻  Dispositivo:", device)
model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base", num_labels=2).to(device)
opt = optim.AdamW(model.parameters(), lr=2e-5)

# ------ 3.1  focal-loss ----------


def focal_loss(logits, targets, alpha=(1.0, 1.5), gamma=3):
    ce = nn.functional.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce)
    at = torch.tensor(alpha, device=logits.device)[targets]
    return (at*((1-pt)**gamma)*ce).mean()

# ---------- 4. treino ----------


def train(num_epochs=8, patience=3):
    best_val = 1e9
    wait = 0
    for ep in range(1, num_epochs+1):
        model.train()
        tot = 0
        for batch in train_loader:
            opt.zero_grad()
            inp = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labs = batch['labels'].to(device)
            loss = focal_loss(model(**inp).logits, labs)
            loss.backward()
            opt.step()
            tot += loss.item()
        print(f"🔹 Época {ep}  loss={tot/len(train_loader):.4f}", end="  ")

        # -------- validação loss ----------
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for b in val_loader:
                inp = {k: v.to(device) for k, v in b.items() if k != 'labels'}
                labs = b['labels'].to(device)
                v_loss += focal_loss(model(**inp).logits, labs).item()
        v_loss /= len(val_loader)
        print(f"val_loss={v_loss:.4f}")
        if v_loss < best_val:
            best_val = v_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("🛑 early-stopping")
                break


train()

# ---------- 5. achar melhor threshold na validação ----------
model.eval()
P, Y = [], []
with torch.no_grad():
    for b in val_loader:
        inp = {k: v.to(device) for k, v in b.items() if k != 'labels'}
        Y.extend(b['labels'].numpy())
        P.extend(torch.softmax(model(**inp).logits, dim=1)[:, 1].cpu().numpy())

best_t, best_f1 = 0.5, 0
for t in np.arange(0.5, 0.91, 0.01):
    pred = (np.array(P) >= t).astype(int)
    f1 = f1_score(Y, pred)
    if f1 > best_f1:
        best_f1, best_t = f1, t
print(f"⭐ Threshold escolhido {best_t:.2f} (F1 val={best_f1:.3f})")

# ---------- 6. avaliação no TESTE ----------


def evaluate(loader, threshold):
    pr, tl = [], []
    with torch.no_grad():
        for b in loader:
            inp = {k: v.to(device) for k, v in b.items() if k != 'labels'}
            probs = torch.softmax(model(**inp).logits,
                                  dim=1)[:, 1].cpu().numpy()
            pr.extend((probs >= threshold).astype(int))
            tl.extend(b['labels'].numpy())
    return np.array(pr), np.array(tl)


preds, true = evaluate(test_loader, best_t)

print("\n📊 Classification Report (RoBERTa):")
print(classification_report(true, preds, digits=2))

# matriz de confusão
cmatrix = confusion_matrix(true, preds, labels=[1, 0])  # [1] primeiro

fig, ax = plt.subplots(figsize=(5.2, 4.8))  # canvas levemente maior
disp = ConfusionMatrixDisplay(
    confusion_matrix=cmatrix,
    display_labels=["Match (1)", "Unmatch (0)"]
)
disp.plot(cmap=cm.Blues, ax=ax, colorbar=False)

plt.title("Matriz de Confusão – RoBERTa")
plt.tight_layout()

plt.savefig(
    "C:/TCC2/pdf_graficos/roberta_confusion_matrix.pdf",
    bbox_inches="tight",
    pad_inches=0.05,
    dpi=300
)
plt.show()
plt.close()

# Impressão dos totais de falsos positivos e falsos negativos
false_positives = np.sum((true == 0) & (preds == 1))
false_negatives = np.sum((true == 1) & (preds == 0))
print(f"Falsos Positivos: {false_positives}")
print(f"Falsos Negativos: {false_negatives}")

elapsed = time.time() - start_prog
print(
    f"⏱️  Tempo total de execução: {elapsed/60:.2f} min  ({elapsed:.1f} seg)")

model.save_pretrained("C:/TCC2/modelos_treinados/modelo_roberta_v3_treinado")
tok.save_pretrained("C:/TCC2/modelos_treinados/modelo_roberta_v3_treinado")
print("✅ Modelo e tokenizer salvos.")

# ---------- 7. gráfico ----------
groups = 8
true_groups = np.array_split(true, groups)
pred_groups = np.array_split(preds, groups)

acc, fp, fn, labs = [], [], [], []
for i, (gt, pr) in enumerate(zip(true_groups, pred_groups)):
    acc.append((gt == pr).sum())
    fp.append(((gt == 0) & (pr == 1)).sum())
    fn.append(((gt == 1) & (pr == 0)).sum())
    labs.append(f"A{i+1}")

plt.figure(figsize=(12, 6))
x = np.arange(len(labs))
w = .3
plt.bar(x - w, acc, width=w, color='blue', label='Acertos')
plt.bar(x, fn, width=w, color='orange', label='Falsos Negativos')
plt.bar(x + w, fp, width=w, color='red', label='Falsos Positivos')
plt.xticks(x, labs)
plt.xlabel("Amostras (≈ 362 perfis)")
plt.title("Acertos vs FN vs FP (RoBERTa)")
plt.legend()
plt.grid(axis='y', ls='--', alpha=.6)
plt.tight_layout()
plt.savefig("C:/TCC2/pdf_graficos/roberta_v3_2.pdf")
plt.show()
