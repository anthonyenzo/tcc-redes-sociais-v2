import pandas as pd
from rapidfuzz import fuzz

# Caminhos dos arquivos
POSITIVOS_PATH = "C:/TCC2/data/usernames_20000_balanceado.csv"
BASE_COMPLETA_PATH = "C:/TCC2/data/perfis_base.csv"

# Carrega os datasets
df = pd.read_csv(POSITIVOS_PATH)
df_base = pd.read_csv(BASE_COMPLETA_PATH)

# Filtra os 10.000 positivos
positivos = df[df["Label"] == 1].copy()
positivos = positivos.reset_index(drop=True)

# Evita usernames duplicados
usernames_base = set(df_base["Username"].str.lower())

# Lista para armazenar os novos negativos difíceis
negativos_dificeis = []

for idx, row in positivos.iterrows():
    nome = str(row["Name"]).strip()
    username_real = str(row["Username"]).strip().lower()

    # Candidatos: usernames diferentes do verdadeiro
    candidatos = df_base[df_base["Username"].str.lower() != username_real].copy()
    candidatos["score"] = candidatos["Username"].apply(
        lambda x: fuzz.token_sort_ratio(x.lower(), username_real) / 100
    )

    # Filtra os usernames mais parecidos (ajuste o limiar conforme necessário)
    similares = candidatos[candidatos["score"] >= 0.65]
    similares = similares.sort_values(by="score", ascending=False)

    if not similares.empty:
        username_falso = similares.iloc[0]["Username"]
        negativos_dificeis.append({
            "Name": nome,
            "Username": username_falso,
            "Label": 0
        })

    # Para gerar exatamente 10.000 negativos difíceis
    if len(negativos_dificeis) >= 10000:
        break

# Converte para DataFrame
df_neg_dificil = pd.DataFrame(negativos_dificeis)

# Junta com os 20.000 pares originais
df_final = pd.concat([df, df_neg_dificil], ignore_index=True)
df_final = df_final.drop_duplicates(subset=["Name", "Username"])
df_final = df_final.sample(frac=1, random_state=42)  # embaralha

# Salva a nova base balanceada com negativos difíceis
df_final.to_csv("C:/TCC2/data/dataset_30000_com_negativos_dificeis.csv", index=False)

print("✅ Base final gerada com sucesso!")
print("Total de pares:", df_final.shape[0])
print("Total positivos (Label 1):", (df_final["Label"] == 1).sum())
print("Total negativos (Label 0):", (df_final["Label"] == 0).sum())
