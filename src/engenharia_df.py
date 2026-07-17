def criar_features(df, df_original):
    """
    Fase 3 - Feature Engineering.
    Cria novas variáveis numéricas a partir dos sensores e faz o encoding da 'tipo'.
    """
    # Potência mecânica (proporcional a: rotação x torque)
    df['potencia'] = df['velocidade_rotacao_rpm'] * df['torque_nm']

    # Diferença de temperatura (dissipação de calor: processo - ambiente)
    df['diferenca_temperatura'] = df['temperatura_processo_k'] - df['temperatura_ar_k']

    # Esforço mecânico acumulado (desgaste x torque)
    df['esforco_mecanico'] = df['desgaste_ferramenta_min'] * df['torque_nm']

    # Encoding ordinal da variável 'tipo' (L=0, M=1, H=2).
    # Mapeia a partir do df_original (idempotente: não corrompe se rodar mais de uma vez).
    df['tipo'] = df_original['tipo'].map({'L': 0, 'M': 1, 'H': 2})

    print("Novas variáveis criadas:")
    print(df[['potencia', 'diferenca_temperatura', 'esforco_mecanico']].head())
    return df