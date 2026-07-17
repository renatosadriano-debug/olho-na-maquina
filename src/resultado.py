import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score
from src.saidas import salvar_figura


def veredito_final(melhor_knn, melhor_arvore, X_test_scaled, X_test, y_test):
    """Fase 7 - Acurácia final no teste dos melhores modelos e veredito (KNN vs Árvore)."""
    acc_knn = accuracy_score(y_test, melhor_knn.predict(X_test_scaled))
    acc_arvore = accuracy_score(y_test, melhor_arvore.predict(X_test))

    print(f"\nAcurácia Final KNN (k=3): {acc_knn:.3f}")
    print(f"Acurácia Final Árvore (depth=5): {acc_arvore:.3f}")

    if acc_arvore > acc_knn:
        print("\nVeredito Final: a Árvore de Decisão teve o melhor desempenho no teste.")
    else:
        print("\nVeredito Final: o KNN teve o melhor desempenho no teste.")

    return acc_knn, acc_arvore


def grafico_comparativo(acc_knn, acc_arvore):
    """Fase 7 - Barras comparando a acurácia de KNN e Árvore no teste."""
    modelos = ['KNN (K=3)', 'Árvore (depth=5)']
    acuracias = [acc_knn, acc_arvore]

    plt.figure(figsize=(6, 4))
    barras = plt.bar(modelos, acuracias, color=['skyblue', 'salmon'], width=0.6)
    plt.ylim(0, 1.15)
    plt.title('Acurácia no Teste - KNN vs Árvore', pad=8)
    for barra in barras:
        plt.text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 0.02,
                 f'{barra.get_height():.3f}', ha='center', va='bottom', fontweight='bold')
    salvar_figura('05_comparacao_knn_arvore.png')


def grafico_importancia(melhor_arvore, X):
    """Fase 7 - Importância das variáveis segundo a Árvore de Decisão."""
    importancias = (pd.Series(melhor_arvore.feature_importances_, index=X.columns) * 100).sort_values(ascending=False)

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(x=importancias.values, y=importancias.index)
    plt.title('Importância das Variáveis - Árvore de Decisão (depth=5)')
    plt.xlabel('Importância (%)')
    plt.ylabel('Variáveis')
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f%%', label_type='edge', fontweight='bold')
    salvar_figura('06_importancia_arvore.png')

    print(importancias.round(2))