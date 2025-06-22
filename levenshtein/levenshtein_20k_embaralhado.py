
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from Levenshtein import opcodes
from sklearn.metrics import classification_report
import time

inicio = time.time()
caminho_csv = "C:/TCC2/data/usernames_20000_balanceado_embaralhado.csv"
df = pd.read_csv(caminho_csv)
df = df.head(2898)

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

tamanho_amostra = 362
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
largura = 0.3

plt.bar(x - largura, acertos, width=largura, color="blue", label="Acertos")
plt.bar(x, falsos_negativos, width=largura, color="orange", label="Falsos Negativos")
plt.bar(x + largura, falsos_positivos, width=largura, color="red", label="Falsos Positivos")

plt.xlabel("Amostras (1000 perfis cada)")
plt.ylabel("Quantidade")
plt.title("Acertos vs Falsos Negativos vs Falsos Positivos (Levenshtein)")
plt.xticks(x, grupos, rotation=45)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.savefig("C:/TCC2/pdf_graficos/levenshtein_20k_embaralhado.pdf")
plt.show()

print("\n📊 Classification Report (Levenshtein):")
print(classification_report(todos_true_labels, todos_pred_labels, zero_division=0))
# Totais finais
total_acertos = sum(resultados_corretos)
total_fn = sum(falsos_negativos)
total_fp = sum(falsos_positivos)

print(f"\n✅ Acertos totais: {total_acertos}")
print(f"❌ Falsos positivos totais: {total_fp}")
print(f"🟧 Falsos negativos totais: {total_fn}")

distancia_media = np.mean([custom_levenshtein(row["Name"], row["Username"]) for _, row in df.iterrows()])
print(f"📉 Distância média de Levenshtein (ajustada): {distancia_media:.2f}")
