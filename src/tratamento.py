import matplotlib.pyplot as plt
import seaborn as sns

# Colunas com outliers (definem a estratégia de imputação por mediana)
COLUNAS_COM_OUTLIERS = ['velocidade_rotacao_rpm', 'torque_nm']


def verificar_duplicados(df):
    """
    Fase 2 - Verifica duplicados pela definição de negócio:
    mesma linha em todas as colunas, ignorando apenas o 'udi' (índice).
    """
    qtd = df.duplicated(subset=df.columns.drop('udi')).sum()
    print("Duplicados (todas as colunas, menos udi):", qtd)
    # Resultado 0: não há equipamentos duplicados, nada é removido.
    return df


def imputar_nulos(df):
    """
    Fase 2 - Imputação dos valores ausentes.
    Média nas colunas simétricas (sem outliers) e mediana nas colunas com outliers.
    """
    # Média nas colunas simétricas
    df['temperatura_ar_k'] = df['temperatura_ar_k'].fillna(df['temperatura_ar_k'].mean())
    df['temperatura_processo_k'] = df['temperatura_processo_k'].fillna(df['temperatura_processo_k'].mean())

    # Mediana nas colunas com outliers (mais robusta a valores extremos)
    df['velocidade_rotacao_rpm'] = df['velocidade_rotacao_rpm'].fillna(df['velocidade_rotacao_rpm'].median())
    df['torque_nm'] = df['torque_nm'].fillna(df['torque_nm'].median())

    print("Valores ausentes após a imputação:")
    print(df.isnull().sum())
    return df


def boxplots_outliers(df):
    """Fase 2 - Boxplots para identificar outliers nas variáveis com valores extremos."""
    plt.figure(figsize=(12, 8))
    for i, coluna in enumerate(COLUNAS_COM_OUTLIERS):
        Q1 = df[coluna].quantile(0.25)
        Q3 = df[coluna].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        qtd_outliers = df[(df[coluna] < limite_inferior) | (df[coluna] > limite_superior)][coluna].count()

        plt.subplot(2, 1, i + 1)
        sns.boxplot(x=df[coluna])
        plt.title(f'{coluna} - {qtd_outliers} outliers')

    plt.subplots_adjust(hspace=0.4)
    plt.show()