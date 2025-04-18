import json
import csv
import io
import os
import websockets
import asyncio

usuarios_processados = set()
dataframes_usuarios = {}

# ===================== Função para achatar JSON =====================
def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())

        elif isinstance(v, list):
            if k == 'teclasPressionadasJogador1':
                nomes = [
                    'primeiraTeclaJogador1',
                    'segundaTeclaJogador1',
                    'terceiraTeclaJogador1',
                    'quartaTeclaJogador1'
                ]
                for i in range(4):
                    valor = v[i] if i < len(v) else ''
                    items.append((nomes[i], valor))
            else:
                for idx, val in enumerate(v):
                    items.append((f"{new_key}_{idx}", val))
        else:
            items.append((new_key, v))
    return dict(items)

# ===================== Salvar CSV =====================
def verifica_ou_cria_arquivo_csv_por_id(user_id, data):
    flattened = flatten_dict(data)
    flattened['type'] = 'gameState'
    flattened['id'] = user_id

    # Definir o caminho da pasta Dados
    folder = 'Dados'
    if not os.path.exists(folder):
        os.makedirs(folder)  # Cria a pasta se não existir

    filename = os.path.join(folder, f"{user_id}.csv")
    
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=flattened.keys())

        if not file_exists:
            writer.writeheader()
        writer.writerow(flattened)

# ===================== Carregar CSV (opcional) =====================
def carregar_csv_para_dataframe(user_id):
    import pandas as pd
    filename = f"{user_id}.csv"
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    return pd.DataFrame()

# ===================== WebSocket Handler =====================
async def handler(websocket):
    print('Cliente conectado')
    try:
        async for message in websocket:
            try:
                print(f"Mensagem recebida: {message}")

                # A mensagem já está em formato JSON, então apenas decodifique
                try:
                    data = json.loads(message)  # Aqui estamos assumindo que 'message' é um JSON válido
                except json.JSONDecodeError:
                    await websocket.send("Erro: JSON inválido.")
                    continue

                print("JSON decodificado:", data)

                user_id = str(data.get('id'))
                if not user_id:
                    await websocket.send("Erro: JSON deve conter campo 'id'.")
                    continue

                # Agora que temos o id do usuário, verificamos e carregamos o CSV
                if user_id not in usuarios_processados:
                    usuarios_processados.add(user_id)
                    df = carregar_csv_para_dataframe(user_id)
                    dataframes_usuarios[user_id] = df

                # Chama a função para salvar o estado do jogo no CSV
                verifica_ou_cria_arquivo_csv_por_id(user_id, data['state'])
                await websocket.send(json.dumps({
                    "type": "ack",
                    "message": "Dados recebidos e armazenados com sucesso."
                }))

                print(f"CSV salvo com sucesso para {user_id}")

            except Exception as e:
                print("Erro interno:", e)
                await websocket.send(f"Erro interno: {str(e)}")
    except websockets.ConnectionClosed:
        print('e desconectado')


# ===================== Iniciar servidor =====================
async def main():
    async with websockets.serve(handler, "localhost", 8080):
        print("Servidor WebSocket iniciado na porta 8080.")
        await asyncio.Future()  # Mantém o servidor rodando

if __name__ == "__main__":
    asyncio.run(main())
