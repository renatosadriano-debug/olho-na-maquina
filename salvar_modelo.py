"""
Treina a Árvore de Decisão e salva em disco para a interface (app.py):
- o modelo (prevê SE a máquina vai falhar);
- a ordem das colunas;
- as estatísticas de cada variável (média e desvio), usadas para medir
  o quão "fora do normal" está o valor de uma máquina ao estimar as causas.
Execute uma vez com:  python salvar_modelo.py
"""
import os
import joblib

from src.leitor import carregar_dados
from src.tratamento import imputar_nulos
from src.engenharia_df import criar_features
from src.modelos import separar_x_y, dividir_e_balancear, treinar_melhor_arvore


def main():
    df_original, df = carregar_dados()
    df = imputar_nulos(df)
    df = criar_features(df, df_original)

    X, y = separar_x_y(df)
    X_train_bal, X_test, y_train_bal, y_test = dividir_e_balancear(X, y)
    modelo = treinar_melhor_arvore(X_train_bal, y_train_bal, max_depth=5)

    os.makedirs('modelo', exist_ok=True)
    joblib.dump(modelo, 'modelo/modelo_arvore.joblib')
    joblib.dump(list(X.columns), 'modelo/colunas.joblib')
    # estatísticas de referência (média e desvio) para medir extremidade
    joblib.dump({'media': X.mean(), 'desvio': X.std()}, 'modelo/estatisticas.joblib')
    print("Modelo, colunas e estatisticas salvos na pasta modelo/")


if __name__ == '__main__':
    main()