import json
import csv
import os
import websockets
import asyncio
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd

# ===================== One-Hot Encoders Globais =====================
turno_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
type_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
turno_encoder.fit(np.array([['ataque'], ['defesa']]))
type_encoder.fit(np.array([['gameState'], ['outro']]))

# ===================== Estruturas globais =====================
usuarios_processados = set()
dataframes_usuarios = {}

TECLA_MAP = {
    'q': 1, 'w': 2, 'e': 3, 'r': 4, 't': 5, 'a': 6, 's': 7,
    # Adicione mais teclas conforme necessário
}

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

# ===================== Tratamento dos dados =====================
def trata_dados_para_csv(user_id, data):
    flattened = flatten_dict(data)
    flattened['type'] = data.get('type', 'gameState')
    flattened['id'] = user_id

    # Padronização de teclas
    for key in ['primeiraTeclaJogador1', 'segundaTeclaJogador1', 'terceiraTeclaJogador1', 'quartaTeclaJogador1']:
        tecla = flattened.get(key, '').lower()
        flattened[key] = TECLA_MAP.get(tecla, 0)

    # One-hot encoding
    turno_val = [[flattened.get('turnoJogador1', '').lower()]]
    type_val = [[flattened.get('type', '').lower()]]

    turno_ohe = turno_encoder.transform(turno_val)[0]
    type_ohe = type_encoder.transform(type_val)[0]

    turno_cols = [f"turno_{cat}" for cat in turno_encoder.categories_[0]]
    type_cols = [f"type_{cat}" for cat in type_encoder.categories_[0]]

    flattened.update(dict(zip(turno_cols, turno_ohe)))
    flattened.update(dict(zip(type_cols, type_ohe)))

    # Padroniza valores vazios ou numéricos
    for k, v in flattened.items():
        if isinstance(v, str) and v.strip() == '':
            flattened[k] = 0
        elif isinstance(v, str) and v.replace('.', '', 1).isdigit():
            flattened[k] = float(v) if '.' in v else int(v)

    # Definindo todas as colunas esperadas
    all_columns = [
        'id', 'numeroInputs', 'vida_jogador1', 'vida_jogador2',
        'primeiraTeclaJogador1', 'segundaTeclaJogador1', 'terceiraTeclaJogador1', 'quartaTeclaJogador1',
        'turnoAtual', 'cooldown_jogador1_q', 'cooldown_jogador1_w', 'cooldown_jogador1_e',
        'cooldown_jogador1_t', 'cooldown_jogador1_r', 'cooldown_jogador1_a', 'cooldown_jogador1_s',
        'cooldown_jogador2_q', 'cooldown_jogador2_w', 'cooldown_jogador2_e',
        'cooldown_jogador2_t', 'cooldown_jogador2_r', 'cooldown_jogador2_a', 'cooldown_jogador2_s',
        *turno_cols, *type_cols,
        'turnoJogador1', 'type', 'partida'
    ]

    for col in all_columns:
        if col not in flattened:
            flattened[col] = 0

    return flattened, all_columns

# ===================== Salvar no CSV =====================
def salva_dados_csv_por_id(user_id, data):
    linha, colunas = trata_dados_para_csv(user_id, data)
    folder = 'Dados'
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"{user_id}.csv")

    file_exists = os.path.isfile(filename)
    print("Valor de 'turnoAtual' antes de salvar:", linha.get('turnoAtual'))

    with open(filename, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=colunas)
        if not file_exists:
            writer.writeheader()
        writer.writerow(linha)

    print("\nCSV atualizado:")
    df = pd.read_csv(filename)
    print(df['turnoAtual'].tail())
    print(df['partida'].tail())

# ===================== Carregar CSV =====================
def carregar_csv_para_dataframe(user_id):
    filename = os.path.join('Dados', f"{user_id}.csv")
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

                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    await websocket.send("Erro: JSON inválido.")
                    continue

                print("JSON decodificado:", data)

                user_id = str(data.get('id'))
                if not user_id:
                    await websocket.send("Erro: JSON deve conter campo 'id'.")
                    continue

                # Verifica e carrega CSV
                if user_id not in usuarios_processados:
                    usuarios_processados.add(user_id)
                    df = carregar_csv_para_dataframe(user_id)
                    dataframes_usuarios[user_id] = df

                # Verifica último turno e define a partida
                df = dataframes_usuarios[user_id]
                last_row = df.iloc[-1] if not df.empty else None
                current_turn = data['state'].get('turnoAtual')
                print("Última linha:", last_row)
                print("Turno atual:", current_turn)

                if last_row is not None:
                    last_turn = last_row['turnoAtual']
                    last_partida = last_row['partida']
                    if current_turn <= last_turn:
                        partida = last_partida + 1
                    else:
                        partida = last_partida
                else:
                    partida = 1

                data['state']['partida'] = partida

                # Salva dados
                salva_dados_csv_por_id(user_id, data['state'])

                # Atualiza o DataFrame na memória
                dataframes_usuarios[user_id] = carregar_csv_para_dataframe(user_id)

                await websocket.send("Dados recebidos e armazenados com sucesso.")
                print(f"CSV salvo com sucesso para {user_id}")

            except Exception as e:
                print("Erro interno:", e)
                await websocket.send(f"Erro interno: {str(e)}")
    except websockets.ConnectionClosed:
        print('Cliente desconectado')

# ===================== Iniciar servidor =====================
async def main():
    async with websockets.serve(handler, "localhost", 8080):
        print("Servidor WebSocket iniciado na porta 8080.")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
