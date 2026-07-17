from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Colunas removidas do X: identificadores, alvo e motivos de falha (data leakage)
COLUNAS_REMOVER = ['udi', 'id_produto', 'falha_maquina',
                   'falha_twf', 'falha_hdf', 'falha_pwf', 'falha_osf', 'falha_rnf']


def separar_x_y(df):
    """Fase 4 - Separa as variáveis preditoras (X) da variável alvo (y)."""
    X = df.drop(columns=COLUNAS_REMOVER)   # preditoras
    y = df['falha_maquina']                # alvo
    print("Colunas do X:", list(X.columns))
    return X, y


def dividir_e_balancear(X, y):
    """
    Fase 4 - Split treino/teste 80/20 estratificado e balanceamento com SMOTE
    aplicado somente no treino (evita data leakage).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)   # só no treino

    print("Antes do SMOTE:", y_train.value_counts().to_dict())
    print("Depois do SMOTE:", y_train_bal.value_counts().to_dict())
    return X_train_bal, X_test, y_train_bal, y_test


def escalonar(X_train_bal, X_test):
    """
    Fase 5 - StandardScaler apenas para o KNN.
    fit_transform no treino (aprende as estatísticas) e transform no teste (só aplica).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def ajuste_knn(X_train_scaled, y_train_bal, X_test_scaled, y_test):
    """Fase 6 - Testa o KNN variando K (3, 5, 7), medindo acurácia de treino e teste."""
    print("\n--- KNN (dados escalonados) ---")
    for k in [3, 5, 7]:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train_bal)
        acc_treino = accuracy_score(y_train_bal, knn.predict(X_train_scaled))
        acc_teste = accuracy_score(y_test, knn.predict(X_test_scaled))
        print(f"K={k}: Treino = {acc_treino:.3f} | Teste = {acc_teste:.3f}")


def ajuste_arvore(X_train_bal, y_train_bal, X_test, y_test):
    """Fase 6 - Testa a Árvore variando max_depth (3, 5, None), com dados crus."""
    print("\n--- Árvore de Decisão (dados crus) ---")
    for profundidade in [3, 5, None]:
        arvore = DecisionTreeClassifier(max_depth=profundidade, random_state=42)
        arvore.fit(X_train_bal, y_train_bal)
        acc_treino = accuracy_score(y_train_bal, arvore.predict(X_train_bal))
        acc_teste = accuracy_score(y_test, arvore.predict(X_test))
        print(f"max_depth={profundidade}: Treino = {acc_treino:.3f} | Teste = {acc_teste:.3f}")


def treinar_melhor_knn(X_train_scaled, y_train_bal, k=3):
    """Treina o melhor KNN (K=3) com os dados escalonados."""
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train_bal)
    return knn


def treinar_melhor_arvore(X_train_bal, y_train_bal, max_depth=5):
    """Treina a melhor Árvore (max_depth=5) com os dados crus."""
    arvore = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    arvore.fit(X_train_bal, y_train_bal)
    return arvore