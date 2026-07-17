import pandas as pd


def carregar_dados(caminho='data/manutencao_preditiva.csv'):
    """
    Lê a base de dados a partir de um caminho relativo.
    Retorna a base original (backup intocável) e uma cópia de trabalho.
    """
    df_original = pd.read_csv(caminho)   # base original - nunca alterar
    df_ler = df_original.copy()          # cópia de trabalho - manipular esta
    return df_original, df_ler