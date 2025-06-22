# ------------------------------------------------------------
# Inference — RoBERTa pré-treinado em 28 000 pares sorteados
# ------------------------------------------------------------
import pandas as pd, numpy as np, re, gc, torch
import time
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix

inicio = time.time()
# ---------- configurações ----------
SEED      = 42
N_SAMPLES = 28000
MODEL_DIR = r"C:/TCC2/modelos_treinados/modelo_roberta_v3_treinado"
CSV_IN    = r"C:/TCC2/data/dataset_v3.csv"
CSV_OUT   = r"C:/TCC2/data/roberta_preds_28000.csv"
THRESHOLD = 0.57            # limiar validado
BATCH     = 16

# ---------- carregar modelo/tokenizer ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok   = RobertaTokenizer.from_pretrained(MODEL_DIR)
model = RobertaForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval();  gc.collect();  torch.cuda.empty_cache()

# ---------- ler CSV e sortear 28 000 linhas ----------
df_full = pd.read_csv(CSV_IN)
df      = df_full.sample(n=N_SAMPLES, random_state=SEED).reset_index(drop=True)

# ---------- normalização idêntica ao treino ----------
def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

df["Name"]     = df["Name"].astype(str).apply(norm)
df["Username"] = df["Username"].astype(str).str.strip().str.lower()

# ---------- Dataset / DataLoader ----------
class ProfileDS(Dataset):
    def __init__(self, frame): self.arr = frame[["Name","Username"]].values
    def __len__(self): return len(self.arr)
    def __getitem__(self, idx):
        n, u = self.arr[idx]
        enc  = tok(n, u, truncation=True, max_length=32,
                   padding="max_length", return_tensors="pt")
        return {k: v.squeeze(0) for k, v in enc.items()}

loader = DataLoader(ProfileDS(df), batch_size=BATCH, shuffle=False,
                    pin_memory=True)

# ---------- inferência ----------
probs, preds = [], []
with torch.no_grad():
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        p1 = torch.softmax(model(**batch).logits, dim=1)[:, 1].cpu()
        probs.extend(p1.numpy())
        preds.extend((p1 >= THRESHOLD).int().numpy())

# ---------- salvar resultados ----------
df["prob_match"] = probs
df["pred_label"] = preds
df.to_csv(CSV_OUT, index=False)
print("✅ Resultados salvos em", CSV_OUT)

print(f"\n⏱️  Tempo total: {time.time() - inicio:.2f} s")

# ---------- métricas (dataset_v3 já possui Label) ----------
y_true = df["Label"].values
print("\n📊 Classification report — RoBERTa (28 000 pares)")
print(classification_report(y_true, preds, digits=2, zero_division=0))

cm = confusion_matrix(y_true, preds, labels=[1, 0])
print("Matriz de confusão [ [TP FN]; [FP TN] ]:\n", cm)
