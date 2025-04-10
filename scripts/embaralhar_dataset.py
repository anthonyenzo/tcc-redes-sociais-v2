import pandas as pd

# Caminho para o arquivo original
caminho_original = "C:/TCC2/data/usernames_20000_balanceado.csv"

# Ler o CSV
df = pd.read_csv(caminho_original)

# Embaralhar as linhas
df_embaralhado = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Salvar como novo CSV
df_embaralhado.to_csv("usernames_20000_balanceado_embaralhado.csv", index=False)

print("✅ CSV embaralhado salvo com sucesso!")
