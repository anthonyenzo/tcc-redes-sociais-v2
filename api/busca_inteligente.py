# api/busca_inteligente.py

import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# Caminho do CSV da base de perfis
CSV_PATH = "./data/perfis_base_modified.csv"  # ajuste conforme o seu

# Caminho do modelo RoBERTa treinado
MODEL_PATH = "./modelos_treinados/modelo_roberta_v3_treinado/"

# Carrega a base apenas uma vez ao importar o módulo
df_perfis = pd.read_csv(CSV_PATH)

# Carrega modelo e tokenizer uma vez só
tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def buscar_perfis_inteligente(query, top_n=20, batch_size=32):
    # considera apenas perfis rotulados como 1
    df_validos = df_perfis[df_perfis["Label"] == 1]
    # Filtra rapidamente no DataFrame por substring em Name ou Username
    mask =  (df_perfis["Label"] == 1) & (
        df_perfis["Name"].str.lower().str.contains(query.lower()) |
        df_perfis["Username"].str.lower().str.contains(query.lower())
    )
    candidatos = df_perfis[mask].copy()

    if candidatos.empty:
        return []

    # Preprocessamento para o modelo
    nomes = candidatos["Name"].astype(str).str.strip().str.lower().tolist()
    usernames = candidatos["Username"].astype(
        str).str.strip().str.lower().tolist()

    # Batch prediction com o RoBERTa
    scores = []
    for i in range(0, len(nomes), batch_size):
        batch_names = nomes[i:i+batch_size]
        batch_usernames = usernames[i:i+batch_size]
        inputs = tokenizer(
            batch_names, batch_usernames,
            padding="max_length", truncation=True,
            max_length=32, return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1)[
                :, 1].cpu().numpy()  # Prob da classe 1
        scores.extend(probs.tolist())

    # Adiciona a probabilidade ao DataFrame de candidatos
    candidatos["probabilidade"] = scores

    # Calcula prioridade: 2 = termo no início, 1 = termo em qualquer lugar
    termo_busca = query.lower().strip()

    def prioridade(row):
        if row["Name"].lower().startswith(termo_busca) or row["Username"].lower().startswith(termo_busca):
            return 2
        return 1

    candidatos["prioridade"] = candidatos.apply(prioridade, axis=1)

    # Ordena: primeiro pelos que começam com o termo, depois pela maior probabilidade
    candidatos = candidatos.sort_values(
        by=["prioridade", "probabilidade"], ascending=[False, False])

    # Seleciona os top N resultados
    top_resultados = candidatos.head(top_n)

    # Monta resposta para API
    resultados = [
        {
            "Name": row["Name"],
            "Username": row["Username"],
            "Probability": round(row["probabilidade"], 4)
        }
        for idx, row in top_resultados.iterrows()
    ]
    return resultados
