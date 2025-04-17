import asyncio
import websockets
import os
import json
import csv
import pandas as pd

PASTA_ARQUIVOS = "Dados"
os.makedirs(PASTA_ARQUIVOS, exist_ok=True)

usuarios_processados = set()                # IDs já tratados na sessão
dataframes_usuarios = {}                    # Armazena os DataFrames por ID


def verifica_ou_cria_arquivo_csv_por_id(user_id: str, dados: dict) -> None:
    """Verifica/cria CSV para o ID e insere os dados como nova linha."""
    caminho_arquivo = os.path.join(PASTA_ARQUIVOS, f"{user_id}.csv")
    arquivo_existe = os.path.exists(caminho_arquivo)

    dados["id"] = user_id  # Garante que o ID está presente

    with open(caminho_arquivo, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=dados.keys())

        if not arquivo_existe:
            writer.writeheader()

        writer.writerow(dados)


def carregar_csv_para_dataframe(user_id: str) -> pd.DataFrame:
    """Carrega o CSV de um usuário (se existir) em um DataFrame do pandas."""
    caminho_arquivo = os.path.join(PASTA_ARQUIVOS, f"{user_id}.csv")
    if os.path.exists(caminho_arquivo):
        df = pd.read_csv(caminho_arquivo)
        print(f"[INFO] Dados carregados para DataFrame do usuário {user_id}.")
        return df
    else:
        print(f"[INFO] Arquivo CSV do usuário {user_id} ainda não existe.")
        return pd.DataFrame()  # Retorna DataFrame vazio


async def handler(websocket):
    print('Cliente conectado')
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                user_id = str(data.get('id'))
                if not user_id:
                    await websocket.send("Erro: JSON deve conter campo 'id'.")
                    continue

                if user_id not in usuarios_processados:
                    usuarios_processados.add(user_id)
                    df = carregar_csv_para_dataframe(user_id)
                    dataframes_usuarios[user_id] = df  # Guarda o DataFrame para uso interno

                verifica_ou_cria_arquivo_csv_por_id(user_id, data)
                await websocket.send("Dados recebidos e armazenados com sucesso.")

            except json.JSONDecodeError:
                await websocket.send("Erro: JSON inválido.")
    except websockets.ConnectionClosed:
        print('Cliente desconectado')


async def main():
    async with websockets.serve(handler, "localhost", 8080):
        print("Servidor WebSocket rodando em ws://localhost:8080")
        await asyncio.Future()  # Roda para sempre


asyncio.run(main())
