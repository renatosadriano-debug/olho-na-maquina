"""
Olho na Máquina - Interface web para previsão de falhas.
Execute com:  streamlit run app.py
(É preciso rodar antes: python salvar_modelo.py)
"""
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Olho na Máquina", page_icon="👁️")

st.title("👁️⚙️ Olho na Máquina")
st.write("Sistema de previsão de falhas em equipamentos industriais. "
         "Informe os dados dos sensores e clique em **Prever**.")

# Carrega o modelo, as colunas e as estatísticas de referência
modelo = joblib.load('modelo/modelo_arvore.joblib')
colunas = joblib.load('modelo/colunas.joblib')
estatisticas = joblib.load('modelo/estatisticas.joblib')

# Cada variável está ligada a um mecanismo de falha
MAPA_CAUSA = {
    'potencia': 'Falha de potência',
    'velocidade_rotacao_rpm': 'Falha de potência',
    'torque_nm': 'Sobrecarga mecânica',
    'esforco_mecanico': 'Sobrecarga mecânica',
    'desgaste_ferramenta_min': 'Desgaste da ferramenta',
    'diferenca_temperatura': 'Dissipação de calor (superaquecimento)',
    'temperatura_ar_k': 'Dissipação de calor (superaquecimento)',
    'temperatura_processo_k': 'Dissipação de calor (superaquecimento)',
}

st.header("Dados dos sensores")

col1, col2 = st.columns(2)
with col1:
    tipo = st.selectbox("Tipo do equipamento", ['L (Baixo)', 'M (Médio)', 'H (Alto)'])
    temp_ar = st.number_input("Temperatura do ar (K)", value=300.0, step=0.1)
    temp_proc = st.number_input("Temperatura do processo (K)", value=310.0, step=0.1)
with col2:
    velocidade = st.number_input("Velocidade de rotação (RPM)", value=1500.0, step=10.0)
    torque = st.number_input("Torque (Nm)", value=40.0, step=1.0)
    desgaste = st.number_input("Desgaste da ferramenta (min)", value=100, step=1)

if st.button("🔍 Prever", type="primary"):
    tipo_enc = {'L (Baixo)': 0, 'M (Médio)': 1, 'H (Alto)': 2}[tipo]
    valores = {
        'tipo': tipo_enc,
        'temperatura_ar_k': temp_ar,
        'temperatura_processo_k': temp_proc,
        'velocidade_rotacao_rpm': velocidade,
        'torque_nm': torque,
        'desgaste_ferramenta_min': desgaste,
        'potencia': velocidade * torque,
        'diferenca_temperatura': temp_proc - temp_ar,
        'esforco_mecanico': desgaste * torque,
    }
    entrada = pd.DataFrame([valores])[colunas]

    previsao = modelo.predict(entrada)[0]
    risco = modelo.predict_proba(entrada)[0][1] * 100

    st.divider()
    st.metric("Risco de falha", f"{risco:.1f}%")

    if previsao == 1:
        st.error("⚠️ ALERTA: esta máquina provavelmente VAI FALHAR!\n\n"
                 "Recomendação: agendar manutenção preventiva.")

        # --- Causas prováveis (específicas desta máquina) ---
        # peso = importância da variável no modelo x quão fora do normal está o valor
        importancias = pd.Series(modelo.feature_importances_, index=colunas)
        media = estatisticas['media']
        desvio = estatisticas['desvio']

        contrib_causa = {}
        for feat, causa in MAPA_CAUSA.items():
            if desvio[feat] > 0:
                extremidade = abs(entrada[feat].iloc[0] - media[feat]) / desvio[feat]
            else:
                extremidade = 0
            peso = importancias[feat] * extremidade
            contrib_causa[causa] = contrib_causa.get(causa, 0) + peso

        total = sum(contrib_causa.values())
        if total > 0:
            st.subheader("Causas prováveis para esta máquina")
            causas_pct = {c: (v / total * 100) for c, v in contrib_causa.items()}
            for causa, pct in sorted(causas_pct.items(), key=lambda x: x[1], reverse=True):
                if pct >= 1:   # esconde causas irrelevantes
                    st.write(f"**{causa}: {pct:.1f}%**")
                    st.progress(int(pct))
    else:
        st.success("✅ Máquina operando NORMALMENTE.")