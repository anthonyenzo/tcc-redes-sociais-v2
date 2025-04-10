import pandas as pd
import random

# Caminho do CSV original com 20.000 pares corretos
caminho_csv = "C:/TCC2/data/usernames_20000.csv"

# Carregar os dados
df = pd.read_csv(caminho_csv)
df["Name"] = df["Name"].astype(str).str.strip()
df["Username"] = df["Username"].astype(str).str.strip()

# Selecionar 10.000 pares corretos
df_match = df.sample(n=10000, random_state=42).copy()
df_match["Label"] = 1  # Match correto

# Criar pares incorretos sem repetição de Name ou Username dos pares corretos
names_usados = set(df_match["Name"])
usernames_usados = set(df_match["Username"])

# Listas disponíveis para embaralhar
names_disponiveis = df_match["Name"].tolist()
usernames_disponiveis = df_match["Username"].tolist()

# Embaralhar até garantir que os pares gerados não coincidam
pares_incorretos = []
tentativas = 0
max_tentativas = 100000

while len(pares_incorretos) < 10000 and tentativas < max_tentativas:
    nome = random.choice(names_disponiveis)
    username = random.choice(usernames_disponiveis)

    # Verificar se é um par incorreto válido
    if nome != username and not ((df_match["Name"] == nome) & (df_match["Username"] == username)).any():
        if (nome, username) not in pares_incorretos:
            pares_incorretos.append((nome, username))

    tentativas += 1

# Criar DataFrame dos pares incorretos
df_nao_match = pd.DataFrame(pares_incorretos, columns=["Name", "Username"])
df_nao_match["Label"] = 0

# Concatenar os dados e salvar
df_balanceado = pd.concat([df_match, df_nao_match], ignore_index=True)
df_balanceado.to_csv("C:/TCC2/data/usernames_20000_balanceado.csv", index=False)

print(f"✅ Dataset balanceado salvo com {len(df_balanceado)} pares (50% corretos e 50% incorretos).")
