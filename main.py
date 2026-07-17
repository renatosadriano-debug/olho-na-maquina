"""
Olho na Máquina - Pipeline de Manutenção Preditiva (versão modular .py)

Orquestra as 7 fases do pipeline, chamando as funções de cada módulo em ordem.
Execute com:  python main.py
"""

from src.leitor import carregar_dados
from src.visao_df import analise_exploratoria
from src.tratamento import verificar_duplicados, imputar_nulos, boxplots_outliers
from src.engenharia_df import criar_features
from src.modelos import (separar_x_y, dividir_e_balancear, escalonar,
                         ajuste_knn, ajuste_arvore,
                         treinar_melhor_knn, treinar_melhor_arvore)
from src.resultado import veredito_final, grafico_comparativo, grafico_importancia
from src.modelos_avancados import treinar_modelos_avancados, grafico_avancados


def main():
    # --- Fase 1: Leitura e Análise Exploratória (EDA) ---
    df_original, df = carregar_dados()
    analise_exploratoria(df)

    # --- Fase 2: Limpeza e Tratamento de Dados ---
    df = verificar_duplicados(df)
    df = imputar_nulos(df)
    boxplots_outliers(df)

    # --- Fase 3: Feature Engineering ---
    df = criar_features(df, df_original)

    # --- Fase 4: Divisão e Balanceamento ---
    X, y = separar_x_y(df)
    X_train_bal, X_test, y_train_bal, y_test = dividir_e_balancear(X, y)

    # --- Fase 5: Escalonamento (apenas para o KNN) ---
    X_train_scaled, X_test_scaled = escalonar(X_train_bal, X_test)

    # --- Fase 6: Ajuste de Parâmetros e Combate ao Overfitting ---
    ajuste_knn(X_train_scaled, y_train_bal, X_test_scaled, y_test)
    ajuste_arvore(X_train_bal, y_train_bal, X_test, y_test)

    # --- Fase 7: Avaliação e Veredito Final (KNN vs Árvore) ---
    melhor_knn = treinar_melhor_knn(X_train_scaled, y_train_bal, k=3)
    melhor_arvore = treinar_melhor_arvore(X_train_bal, y_train_bal, max_depth=5)

    acc_knn, acc_arvore = veredito_final(melhor_knn, melhor_arvore,
                                         X_test_scaled, X_test, y_test)
    grafico_comparativo(acc_knn, acc_arvore)
    grafico_importancia(melhor_arvore, X)

    # --- Exploração adicional (bônus): modelos avançados ---
    # Random Forest, XGBoost e LightGBM (baseados em árvores, usam dados crus)
    resultados = treinar_modelos_avancados(X_train_bal, y_train_bal, X_test, y_test)
    grafico_avancados(resultados)


if __name__ == '__main__':
    main()