# api/database.py
import pandas as pd

# Caminho do seu CSV
CSV_PATH = "./data/perfis_base_modified.csv"

# Carrega o CSV uma única vez ao importar o módulo
df_perfis = pd.read_csv(CSV_PATH)

def buscar_perfis(termo_busca: str, top_n: int = 10):
    termo = termo_busca.lower().strip()
    # Busca por nome OU username contendo o termo
    candidatos = df_perfis[
        df_perfis["Name"].str.lower().str.contains(termo) |
        df_perfis["Username"].str.lower().str.contains(termo)
    ].copy()
    return candidatos.head(top_n).to_dict(orient="records")
