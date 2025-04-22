import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict

# Função para processar e gerar gráfico
def gerar_grafico(pasta, nome_arquivo):
    # Caminho completo do arquivo JSON
    caminho_arquivo = os.path.join(pasta, nome_arquivo)

    # Abrir o arquivo JSON
    with open(caminho_arquivo, 'r') as f:
        dados = json.load(f)

    # Verificar se a chave 'experimentos' existe no arquivo JSON
    if "experimentos" not in dados:
        print(f"Erro: 'experimentos' não encontrado no arquivo {nome_arquivo}. Pulando...")
        return

    # Organizar dados por contexto
    resultados = defaultdict(lambda: {"batch_size": [], "acuracia": []})

    for exp in dados["experimentos"]:
        if "contexto" in exp:  # Ignora os que não têm contexto, como "seq_completas"
            contexto = exp["contexto"]
            resultados[contexto]["batch_size"].append(exp["batch_size"])
            resultados[contexto]["acuracia"].append(exp["acuracia"])

    # Plotar
    plt.figure(figsize=(10, 6))
    for contexto, valores in sorted(resultados.items()):
        plt.plot(valores["batch_size"], valores["acuracia"], marker='o', label=f'Contexto {contexto}')

    plt.xlabel('Batch Size')
    plt.ylabel('Acurácia')
    plt.title(f'Acurácia por Batch Size em cada Contexto - {pasta}')
    plt.legend()
    plt.grid(True)
    grafico_path = f'grafico_acuracia_{pasta}.png'
    plt.savefig(grafico_path)
    plt.close()

    print(f"Gráfico salvo como '{grafico_path}'")

# Função para encontrar o melhor modelo
def encontrar_melhor_modelo(pasta, nome_arquivo):
    # Caminho completo do arquivo JSON
    caminho_arquivo = os.path.join(pasta, nome_arquivo)

    # Abrir o arquivo JSON
    with open(caminho_arquivo, 'r') as f:
        dados = json.load(f)

    # Verificar se a chave 'experimentos' existe no arquivo JSON
    if "experimentos" not in dados:
        print(f"Erro: 'experimentos' não encontrado no arquivo {nome_arquivo}. Pulando...")
        return None

    melhor_acuracia = 0
    melhor_modelo = None

    for exp in dados["experimentos"]:
        if exp["acuracia"] > melhor_acuracia:
            melhor_acuracia = exp["acuracia"]
            melhor_modelo = exp

    return melhor_modelo

# Listar as pastas que você deseja analisar
pastas = ['300', '1000', '10000']

# Para cada pasta, verificar se existe, gerar gráfico e identificar o melhor modelo
for pasta in pastas:
    if os.path.exists(pasta):  # Verifica se a pasta existe
        print(f"Processando a pasta: {pasta}")
        nome_arquivo = 'todos_experimentos.json'  # Arquivo único
        gerar_grafico(pasta, nome_arquivo)
        melhor_modelo = encontrar_melhor_modelo(pasta, nome_arquivo)
        if melhor_modelo:  # Verifica se foi encontrado um modelo válido
            print(f"\nMelhor modelo para {pasta}/{nome_arquivo}:")
            print(f"Contexto: {melhor_modelo['contexto']}")
            print(f"Batch Size: {melhor_modelo['batch_size']}")
            print(f"Acurácia: {melhor_modelo['acuracia']}")
        else:
            print(f"Não foi possível encontrar um modelo válido em {pasta}/{nome_arquivo}.")
    else:
        print(f"Pasta '{pasta}' não encontrada. Pulando...")
