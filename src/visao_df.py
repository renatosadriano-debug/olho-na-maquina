import matplotlib.pyplot as plt
import seaborn as sns

# Colunas de sensores (variáveis explicativas contínuas)
COLUNAS_SENSORES = ['temperatura_ar_k', 'temperatura_processo_k',
                    'velocidade_rotacao_rpm', 'torque_nm', 'desgaste_ferramenta_min']


def visao_geral(df):
    """Fase 1 - Dimensões, tipos de dados e resumo estatístico."""
    print("Dimensões do dataset (linhas, colunas):", df.shape)
    print("\nTipos de dados e valores não-nulos:")
    df.info()
    print("\nResumo estatístico:")
    print(df.describe().round(2))


def grafico_alvo(df):
    """Fase 1 - Gráfico de barras do desbalanceamento da variável alvo."""
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(data=df, x='falha_maquina')
    for container in ax.containers:      # escreve a contagem no topo de cada barra
        ax.bar_label(container)
    plt.title('Distribuição da variável alvo (Falha da máquina)')
    plt.xlabel('Falha (0 = normal, 1 = falha)')
    plt.ylabel('Quantidade de registros')
    plt.show()


def grafico_histogramas(df):
    """Fase 1 - Histogramas de distribuição das variáveis preditoras."""
    df[COLUNAS_SENSORES].hist(figsize=(12, 8))
    plt.tight_layout()
    plt.show()


def grafico_heatmap(df):
    """Fase 1 - Mapa de calor com a correlação de Pearson."""
    plt.figure(figsize=(12, 8))
    matriz_corr = df.corr(numeric_only=True)   # correlação de Pearson entre colunas numéricas
    sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Mapa de Calor - Correlação de Pearson')
    plt.show()


def analise_exploratoria(df):
    """Executa todas as etapas da Fase 1 (EDA)."""
    visao_geral(df)
    grafico_alvo(df)
    grafico_histogramas(df)
    grafico_heatmap(df)