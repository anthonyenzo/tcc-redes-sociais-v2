# ------------------------------------------------------------
# Levenshtein — avaliação em 28 000 (mesmos sorteados p/ RoBERTa)
# ------------------------------------------------------------
import pandas as pd, numpy as np, matplotlib.pyplot as plt, re, time, random
from Levenshtein import opcodes
from sklearn.metrics import classification_report

# ---------- 0. reproducibilidade ---------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---------- 1. carregar e sortear --------------------------
inicio = time.time()
CSV_IN = r"C:/TCC2/data/dataset_v3.csv"

df_full = pd.read_csv(CSV_IN)
df = df_full.sample(n=28_000, random_state=SEED).reset_index(drop=True)  # mesmo conjunto do RoBERTa

# ---------- 2. pré-processamento ---------------------------
def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9_]", "", s.lower())

df["Name"]     = df["Name"].astype(str).apply(norm)
df["Username"] = df["Username"].astype(str).str.strip()

# ---------- 3. distância customizada -----------------------
def custom_levenshtein(a: str, b: str) -> float:
    dist = 0
    for op, i1, _, i2, _ in opcodes(a, b):
        if op == "replace":
            if i1 < len(a) and i2 < len(b) and a[i1] == b[i2]:
                dist += 1.5
            else:
                dist += 2
        else:                          # insert / delete
            dist += 1
    return dist

# ---------- 4. avaliação em lotes --------------------------
BATCH = 1000
grupos, acertos_lote, fn_lote, fp_lote = [], [], [], []
true_all, pred_all = [], []

for i in range(0, len(df), BATCH):
    sub = df.iloc[i:i+BATCH]
    ok = fn = fp = 0

    for _, r in sub.iterrows():
        n, u, y = r["Name"], r["Username"], r["Label"]
        thresh  = max(2, int(len(u) * 0.4))
        pred    = 1 if custom_levenshtein(n, u) <= thresh else 0

        true_all.append(y);  pred_all.append(pred)
        if pred == y:
            ok += 1
        elif y == 1:
            fn += 1
        else:
            fp += 1

    idx = len(grupos) + 1
    grupos.append(f"A{idx}")
    acertos_lote.append(ok); fn_lote.append(fn); fp_lote.append(fp)

    print(f"🔹 A{idx} – Acertos {ok}, FN {fn}, FP {fp}")

# ---------- 5. tempo ---------------------------------------
print(f"\n⏱️  Tempo total: {time.time() - inicio:.2f} s")

# ---------- 6. gráfico ------------------------------------
plt.figure(figsize=(10, 5))
x = np.arange(len(grupos)); w = 0.28
plt.bar(x - w, acertos_lote, width=w, color="blue",   label="Acertos")
plt.bar(x,     fn_lote,     width=w, color="orange", label="Falsos Negativos")
plt.bar(x + w, fp_lote,     width=w, color="red",    label="Falsos Positivos")
plt.xticks(x, grupos);  plt.grid(axis="y", ls="--", alpha=.6)
plt.xlabel("Amostras (≈ 362 pares)");  plt.ylabel("Quantidade")
plt.title("Acertos vs FN vs FP – Levenshtein (2 898 pares)")
plt.legend();  plt.tight_layout()
plt.savefig(r"C:/TCC2/pdf_graficos/lev_2898_barras.pdf")
plt.show()

# ---------- 7. métricas globais ----------------------------
print("\n📊 Classification Report – Levenshtein (28 000 pares)")
print(classification_report(true_all, pred_all, digits=2, zero_division=0))

print(f"\n✅ Acertos totais: {sum(acertos_lote)}")
print(f"❌ Falsos positivos: {sum(fp_lote)}")
print(f"🟧 Falsos negativos: {sum(fn_lote)}")

mean_dist = np.mean(
    [custom_levenshtein(r["Name"], r["Username"]) for _, r in df.iterrows()]
)
print(f"📉 Distância média ajustada: {mean_dist:.2f}")
