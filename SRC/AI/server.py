import json
import csv
import os
import websockets
import asyncio
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
from modelador import *

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Activation, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import json
tf.keras.config.enable_unsafe_deserialization()


from tensorflow.keras import backend as K

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


    for key in ['primeiraTeclaJogador1', 'segundaTeclaJogador1', 'terceiraTeclaJogador1', 'quartaTeclaJogador1']:
        tecla = flattened.get(key, '').lower()
        flattened[key] = TECLA_MAP.get(tecla, 0)


    teclas_disponiveis_str = flattened.get('teclasDisponiveis', '')
    teclas_disponiveis_lista = [t.strip().lower() for t in teclas_disponiveis_str.split(',') if t.strip()]
    teclas_classes = [TECLA_MAP.get(t, 0) for t in teclas_disponiveis_lista]

    for tecla_val in TECLA_MAP.values():
        flattened[f"teclaDisponivel_{tecla_val}"] = 1 if tecla_val in teclas_classes else 0

    flattened['teclasDisponiveis'] = ','.join(str(t) for t in teclas_classes)

    turno_val = [[flattened.get('turnoJogador1', '').lower()]]
    type_val = [[flattened.get('type', '').lower()]]

    turno_ohe = turno_encoder.transform(turno_val)[0]
    type_ohe = type_encoder.transform(type_val)[0]

    turno_cols = [f"turno_{cat}" for cat in turno_encoder.categories_[0]]
    type_cols = [f"type_{cat}" for cat in type_encoder.categories_[0]]

    flattened.update(dict(zip(turno_cols, turno_ohe)))
    flattened.update(dict(zip(type_cols, type_ohe)))

    for k, v in flattened.items():
        if isinstance(v, str) and v.strip() == '':
            flattened[k] = 0
        elif isinstance(v, str) and v.replace('.', '', 1).isdigit():
            flattened[k] = float(v) if '.' in v else int(v)

    all_columns = [
        'id', 'numeroInputs', 'vida_jogador1', 'vida_jogador2',
        'primeiraTeclaJogador1', 'segundaTeclaJogador1', 'terceiraTeclaJogador1', 'quartaTeclaJogador1',
        'turnoAtual',
        *turno_cols,
        'turnoJogador1', 'partida'
    ] + [f"teclaDisponivel_{v}" for v in TECLA_MAP.values()]

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
        linha_filtrada = {k: v for k, v in linha.items() if k in colunas}
        writer.writerow(linha_filtrada)

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

                caminho = data.get('caminho', '').lower()
                if caminho == 'carregar':
                    model = data.get('modelo','').lower()
                    await websocket.send('modelo carregado')
                    return
                
                print("JSON decodificado:", data)

                user_id = str(data.get('id'))
                if not user_id:
                    await websocket.send("Erro: JSON deve conter campo 'id'.")
                    continue

               
                if user_id not in usuarios_processados:
                    usuarios_processados.add(user_id)
                    df = carregar_csv_para_dataframe(user_id)
                    dataframes_usuarios[user_id] = df

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

                if caminho == 'salvar':
                    
                    salva_dados_csv_por_id(user_id, data['state'])

                
                    dataframes_usuarios[user_id] = carregar_csv_para_dataframe(user_id)

                    await websocket.send(json.dumps({"type": "ok", "message": "Dados recebidos e armazenados com sucesso."}))
                
                elif caminho == 'inferir':
                    modelo_id = data.get('modelo')
                    if not modelo_id:
                        await websocket.send(json.dumps({"type": "erro", "message": "Nome do modelo não especificado"}))
                        continue
                    
                    try:
                      
                        caminho_modelo = f"modelos/{modelo_id}.keras"
                        
                        
                        if not os.path.exists(caminho_modelo):
                            await websocket.send(json.dumps({
                                "type": "erro", 
                                "message": f"Modelo '{modelo_id}' não encontrado"
                            }))
                            continue
                        
                     
                        modelo = Modelador.carregar_modelo(caminho_modelo)
                        
                    
                        estado_atual = data['state']
                        
                        estado_formatado = {
                            'vida_jogador1': estado_atual.get('vida_jogador1', 100),
                            'vida_jogador2': estado_atual.get('vida_jogador2', 100),
                            'turnoJogador1': 0 if estado_atual.get('turnoJogador1', '').lower() == 'ataque' else 1,
                        }
                        
                      
                        teclas_disponiveis_str = estado_atual.get('teclasDisponiveis', '')
                        teclas_disponiveis_lista = [t.strip().lower() for t in teclas_disponiveis_str.split(',') if t.strip()]
                        
                      
                        for tecla_key, tecla_val in TECLA_MAP.items():
                            estado_formatado[f'teclaDisponivel_{tecla_val}'] = 1 if tecla_key in teclas_disponiveis_lista else 0
                        
                       
                        estado_formatado['delta_vida_j1'] = 0  
                        estado_formatado['delta_vida_j2'] = 0  
                        estado_formatado['ratio_vida'] = estado_formatado['vida_jogador1'] / max(1, estado_formatado['vida_jogador2'])
                        estado_formatado['total_teclas_disponiveis'] = sum(1 for k in estado_formatado if k.startswith('teclaDisponivel_') and estado_formatado[k] == 1)
                        
                 
                        for i in range(1, 5):
                            tecla_key = f'tecla_anterior_{i}'
                            estado_formatado[tecla_key] = estado_atual.get(f'primeiraTeclaJogador1' if i == 1 else 
                                                                        f'segundaTeclaJogador1' if i == 2 else
                                                                        f'terceiraTeclaJogador1' if i == 3 else
                                                                        'quartaTeclaJogador1', 0)
                        
                      
                        userid = str(data.get('id', user_id))
                        caminho_csv = os.path.join('Dados', f"{userid}.csv")
                        
                    
                        modelador = Modelador(pd.DataFrame())  
                        teclas_previstas = modelador.prever_proximas_teclas(
                            modelo, 
                            estado_formatado,
                        )
                        
                  
                        tecla_map_reverso = {v: k for k, v in TECLA_MAP.items()}
                        teclas_alfabeticas = [tecla_map_reverso.get(t+1, '') for t in teclas_previstas]
                        
                    
                        resposta = {
                            "type": "predicao", 
                            "id": userid,
                            "teclas_previstas": teclas_alfabeticas,
                            "teclas_indices": [int(t) for t in teclas_previstas]
                        }
                        
                      
                        print(f"Previsão para {userid}: {teclas_alfabeticas}")
             
                        await websocket.send(json.dumps(resposta))
                        
                    except Exception as e:
                        print(f"Erro na inferência: {str(e)}")
                        await websocket.send(json.dumps({
                            "type": "erro", 
                            "message": f"Erro ao realizar inferência: {str(e)}"
                        }))

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
