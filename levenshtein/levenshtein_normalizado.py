import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import re
from Levenshtein import opcodes
from sklearn.metrics import classification_report
import time

inicio = time.time()
caminho_csv = "C:/TCC2/data/usernames_20000_balanceado.csv"
df = pd.read_csv(caminho_csv)

df["Name"] = df["Name"].astype(str).str.strip()
df["Username"] = df["Username"].astype(str).str.strip()


def normalizar_nome(name):
    name = name.lower()
    name = re.sub(r'[^a-z0-9_]', '', name)
    return name


df["Name"] = df["Name"].apply(normalizar_nome)


def custom_levenshtein(name, username):
    ops = opcodes(name, username)
    distance = 0
    for op, i1, _, i2, _ in ops:
        if op == "replace":
            if i1 < len(name) and i2 < len(username):
                if name[i1].lower() == username[i2].lower():
                    distance += 1.5
                else:
                    distance += 2
            else:
                distance += 2
        else:
            distance += 1
    return distance


tamanho_amostra = 1000
resultados_corretos = []
falsos_negativos = []
falsos_positivos = []
grupos = []
todos_true_labels = []
todos_pred_labels = []

for i in range(0, len(df), tamanho_amostra):
    grupo_df = df.iloc[i:i + tamanho_amostra]
    acertos = 0
    fn = 0
    fp = 0

    for _, row in grupo_df.iterrows():
        nome = row["Name"]
        username = row["Username"]
        true_label = row["Label"]

        limiar_levenshtein = max(2, int(len(username) * 0.4))
        dist = custom_levenshtein(nome, username)
        pred_label = 1 if dist <= limiar_levenshtein else 0

        todos_true_labels.append(true_label)
        todos_pred_labels.append(pred_label)

        if pred_label == true_label:
            acertos += 1
        elif true_label == 1 and pred_label == 0:
            fn += 1
        elif true_label == 0 and pred_label == 1:
            fp += 1

    grupo_nome = f"A{len(grupos) + 1}"
    grupos.append(grupo_nome)
    resultados_corretos.append(acertos)
    falsos_negativos.append(fn)
    falsos_positivos.append(fp)

    if len(grupos) <= 10:
        print(f"🔹 {grupo_nome} - Acertos: {acertos} | Falsos Negativos: {fn}")
    else:
        print(f"🔹 {grupo_nome} - Acertos: {acertos} | Falsos Positivos: {fp}")

fim = time.time()
print(f"⏱️  Tempo total de treino: {fim - inicio:.2f} segundos")

# Gráfico final
plt.figure(figsize=(12, 6))
x = np.arange(len(grupos))
largura = 0.35
plt.bar(x - largura/2, resultados_corretos,
        width=largura, color="blue", label="Acertos")

# Falsos negativos (laranja) para A1–A10
plt.bar(x[:10] + largura/2, falsos_negativos[:10], width=largura,
        color="orange", label="Falsos Negativos (A1–A10)")

# Falsos positivos (vermelho) para A11–A20
plt.bar(x[10:] + largura/2, falsos_positivos[10:], width=largura,
        color="red", label="Falsos Positivos (A11–A20)")


plt.xlabel("Amostras (1000 perfis cada)")
plt.ylabel("Quantidade")
plt.title("Acertos vs Falsos Negativos (A1–A10) e Falsos Positivos (A11–A20)")
plt.xticks(x, grupos, rotation=45)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)

# plt.savefig("C:/TCC2/pdf_graficos/levenshtein_normalizado_20k.pdf")
plt.tight_layout()
plt.show()

print("\n📊 Classification Report (Levenshtein):")
print(classification_report(todos_true_labels, todos_pred_labels, zero_division=0))

# matriz de confusão
cmatrix = confusion_matrix(todos_true_labels, todos_pred_labels, labels=[1, 0])

fig, ax = plt.subplots(figsize=(5.2, 4.8))          # ← ajuste livre

disp = ConfusionMatrixDisplay(
    confusion_matrix=cmatrix,
    display_labels=["Match (1)", "Unmatch (0)"]
)
disp.plot(cmap=cm.Blues, ax=ax, colorbar=False)

plt.title("Matriz de Confusão – Levenshtein")
plt.tight_layout()                                   # encaixa texto e eixos

plt.savefig(
    "C:/TCC2/pdf_graficos/lev_20k_confusion_matrix.pdf",
    bbox_inches="tight",        # garante que nada fique fora da página
    pad_inches=0.05,            # fina borda branca opcional
    dpi=300                     # melhor nitidez em PDF/Overleaf
)
plt.show()

# Totais finais
total_acertos = sum(resultados_corretos)
total_fn = sum(falsos_negativos)
total_fp = sum(falsos_positivos)

print(f"\n✅ Acertos totais: {total_acertos}")
print(f"❌ Falsos positivos totais: {total_fp}")
print(f"🟧 Falsos negativos totais: {total_fn}")

distancia_media = np.mean(
    [custom_levenshtein(row["Name"], row["Username"]) for _, row in df.iterrows()])
print(f"📉 Distância média de Levenshtein (ajustada): {distancia_media:.2f}")
