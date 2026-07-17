import os
import matplotlib.pyplot as plt

PASTA_SAIDAS = 'outputs'


def salvar_figura(nome):
    """Salva a figura atual na pasta outputs/ (em vez de abrir janela) e fecha."""
    os.makedirs(PASTA_SAIDAS, exist_ok=True)
    caminho = os.path.join(PASTA_SAIDAS, nome)
    plt.savefig(caminho, bbox_inches='tight', dpi=100)
    plt.close()
    print(f"[grafico salvo] {caminho}")


def salvar_resumo(texto, nome='resumo_resultados.txt'):
    """Salva um resumo textual dos resultados na pasta outputs/."""
    os.makedirs(PASTA_SAIDAS, exist_ok=True)
    caminho = os.path.join(PASTA_SAIDAS, nome)
    with open(caminho, 'w', encoding='utf-8') as f:
        f.write(texto)
    print(f"[resumo salvo] {caminho}")
