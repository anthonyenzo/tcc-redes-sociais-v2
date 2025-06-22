import streamlit as st
import requests
import pandas as pd
import math

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Busca de Perfis", page_icon="🔍")

st.title("🔎 Busca de Perfis")

# --- BARRA DE PESQUISA ---
query = st.text_input("Digite um nome ou username para buscar:")

if query:
    response = requests.get(f"{API_URL}/search", params={"query": query})
    if response.status_code == 200:
        perfis = response.json()
        if perfis:
            st.subheader("Perfis Encontrados:")
            for perfil in perfis:
                name = perfil.get("Name", "")
                username = perfil.get("Username", "")
                prob = perfil.get("Probability", None)
                prob_str = f" | Probabilidade: {prob:.2%}" if prob is not None else ""
                st.markdown(f"**{name}**  \n@{username}{prob_str}")
        else:
            st.info("Nenhum perfil encontrado.")
    else:
        st.error("Erro na busca.")

# --- BOTÃO PARA CADASTRAR NOVO PERFIL ---
st.markdown("---")
st.subheader("Cadastrar novo perfil")
with st.form("novo_perfil_form"):
    nome_novo = st.text_input("Nome do perfil")
    username_novo = st.text_input("Username do perfil")
    submit = st.form_submit_button("Cadastrar")

    if submit and nome_novo and username_novo:
        data = {"name": nome_novo, "username": username_novo}
        resp = requests.post(f"{API_URL}/add_profile", json=data)
        if resp.status_code == 200:
            rjson = resp.json()
            if rjson.get("success"):
                st.success(rjson.get("message"))
            else:
                st.warning(rjson.get("message"))
        else:
            st.error("Erro ao cadastrar o perfil.")


# --- TESTAR PREDICT ---
st.markdown("---")
st.subheader("Testar probabilidade de match")

with st.form("predict_form", clear_on_submit=False):
    nome_pred      = st.text_input("Nome do perfil para teste")
    username_pred  = st.text_input("Username do perfil para teste")

    modelo_escolhido = st.radio(
        "Escolha o modelo",
        ["RoBERTa", "Levenshtein"],
        horizontal=True,
    )

    submeter_pred = st.form_submit_button("Calcular probabilidade")

    if submeter_pred and nome_pred and username_pred:
        payload = {"name": nome_pred, "username": username_pred}

        # rota depende da seleção do usuário
        if modelo_escolhido.startswith("RoBERTa"):
            endpoint = f"{API_URL}/predict"                  # RoBERTa
        else:
            endpoint = f"{API_URL}/predict/levenshtein"      # puro Levenshtein

        resp_pred = requests.post(endpoint, json=payload)

        if resp_pred.status_code == 200:
            data = resp_pred.json()

            if modelo_escolhido.startswith("RoBERTa"):
                prob  = data.get("probability", 0.0)
                label = data.get("prediction", 0)

                st.success(
                    f"**RoBERTa**  \n"
                    f"Probabilidade de *match*: **{prob:.2%}**  \n"
                    f"Classificação (> τ ⇒ 1): **{label}**"
                )
            else:
                dist  = data["distance"]
                thr   = data["threshold"]
                label = data["prediction"]

                st.success(
                    f"**Levenshtein**  \n"
                    f"Distância ajustada: **{dist:.2f}**  \n"
                    f"Limiar τ: **{thr}**  \n"
                    f"Classificação (≤ τ ⇒ 1): **{label}**"
                )
        else:
            st.error("Erro ao consultar o serviço.")
