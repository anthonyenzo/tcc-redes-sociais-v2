# api/main.py
from fastapi import FastAPI, Query
from api.busca_inteligente import buscar_perfis_inteligente
from api import busca_inteligente, database
from pydantic import BaseModel
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import pandas as pd
import re
from Levenshtein import opcodes

app = FastAPI()

# Caminho do modelo treinado
MODEL_PATH = "./modelos_treinados/modelo_roberta_v3_treinado/"

# Carregar modelo e tokenizer
tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Threshold usado no predict
THRESHOLD = 0.57  # Ajuste conforme necessário

# funções auxiliares Levenshtein 
def _norm(text: str) -> str:
    """Mesma normalização usada nos scripts offline."""
    return re.sub(r"[^a-z0-9_]", "", text.lower())

def _custom_lev(a: str, b: str) -> float:
    """Levenshtein ponderado (1, 1, 1.5 ou 2)."""
    dist = 0
    for op, i1, _, i2, _ in opcodes(a, b):
        if op == "replace":
            dist += 1.5 if i1 < len(a) and i2 < len(b) and a[i1] == b[i2] else 2
        else:                                  # insert / delete
            dist += 1
    return dist


class InputData(BaseModel):
    name: str
    username: str


@app.post("/predict/")
def predict(data: InputData):
    name = data.name.strip().lower()
    username = data.username.strip().lower()

    inputs = tokenizer(
        name,
        username,
        padding="max_length",
        truncation=True,
        max_length=32,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prob = torch.softmax(logits, dim=1)[0, 1].item()

    prediction = int(prob >= THRESHOLD)
    return {
        "probability": prob,
        "prediction": prediction
    }
    
@app.post("/predict/levenshtein")
def predict_levenshtein(data: InputData):
    name_norm     = _norm(data.name)
    user_norm     = _norm(data.username)

    dist  = _custom_lev(name_norm, user_norm)
    thresh = max(2, int(len(user_norm) * 0.4))   # limiar dinâmico
    is_match = int(dist <= thresh)

    return {
        "distance": dist,
        "threshold": thresh,
        "prediction": is_match
    }


class NewProfile(BaseModel):
    name: str
    username: str


CSV_PATH = "./data/perfis_base_modified.csv"


@app.post("/add_profile")
def add_profile(profile: NewProfile):
    df = pd.read_csv(CSV_PATH)

    if "Label" not in df.columns:
        df["Label"] = 0          # garante a coluna

    # checa existência (case-insensitive)
    mask = (df["Name"].str.lower() == profile.name.lower()) & \
           (df["Username"].str.lower() == profile.username.lower())
    if mask.any():
        return {"success": False, "message": "Perfil já existe na base."}

    # insere com Label = 1
    new_row = pd.DataFrame([{
        "Name": profile.name,
        "Username": profile.username,
        "Label": 1
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(CSV_PATH, index=False)

    # mantém DataFrames globais atualizados
    busca_inteligente.df_perfis = df.copy()
    database.df_perfis = df.copy()

    return {"success": True, "message": "Perfil adicionado com sucesso!"}


@app.get("/search")
def search_perfis_inteligente(query: str = Query(..., min_length=1)):
    return buscar_perfis_inteligente(query)
