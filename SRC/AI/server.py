import json
import csv
import os
import websockets
import asyncio
import pandas as pd
from aigame import prever_tecla_server

usuarios_processados = set()
dataframes_usuarios = {}
modelo_global = None
label_encoder_global = None

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

def verifica_ou_cria_arquivo_csv_por_id(user_id, data):
    flattened = flatten_dict(data)
    flattened['type'] = 'gameState'
    flattened['id'] = user_id
    folder = 'Dados'
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = os.path.join(folder, f"{user_id}.csv")
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=flattened.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(flattened)
    return filename

def carregar_csv_para_dataframe(user_id):
    import pandas as pd
    filename = os.path.join('Dados', f"{user_id}.csv")
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    return pd.DataFrame()

def extrair_ultimas_sequencias(df, janela=3):
    sequencias = []
    for _, row in df.iterrows():
        teclas = [
            row['primeiraTeclaJogador1'],
            row['segundaTeclaJogador1'],
            row['terceiraTeclaJogador1'],
            row['quartaTeclaJogador1']
        ]
        teclas = [t for t in teclas if pd.notna(t) and t != '']
        if teclas:
            sequencias.append(teclas)
    return sequencias[-janela:] if len(sequencias) >= janela else None

async def handler(websocket):
    global modelo_global, label_encoder_global
    print('Cliente conectado')
    try:
        async for message in websocket:
            try:
                print(f"Mensagem recebida: {message}")
                data = json.loads(message)
                user_id = str(data.get('id'))
                if not user_id:
                    await websocket.send(json.dumps({"type": "error", "message": "JSON deve conter campo 'id'."}))
                    continue

                csv_path = verifica_ou_cria_arquivo_csv_por_id(user_id, data['state'])
                df = carregar_csv_para_dataframe(user_id)
                ultimos_turnos = extrair_ultimas_sequencias(df, janela=3)
                
                if ultimos_turnos:
                    print(f"Chamando a LSTM para prever com a sequência: {ultimos_turnos}")
                    proxima_tecla, modelo_global, label_encoder_global = prever_tecla_server(
                        csv_path, modelo_global, label_encoder_global, ultimos_turnos, janela=3
                    )
                    if proxima_tecla:
                        print(f"Previsão para {user_id}: {proxima_tecla}")
                        await websocket.send(json.dumps({
                            "type": "prediction",
                            "proxima_tecla": proxima_tecla
                        }))
                
                await websocket.send(json.dumps({
                    "type": "ack",
                    "message": "Dados recebidos e armazenados com sucesso."
                }))
                print(f"CSV salvo com sucesso para {user_id}")

            except Exception as e:
                print("Erro interno:", e)
                await websocket.send(json.dumps({"type": "error", "message": str(e)}))
    except websockets.ConnectionClosed:
        print('Cliente desconectado')

async def main():
    async with websockets.serve(handler, "localhost", 8080):
        print("Servidor WebSocket iniciado na porta 8080.")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())