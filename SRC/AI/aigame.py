import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
import os

# Função para carregar dados do CSV
def carregar_dados_csv(csv_path):
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

# Função para extrair sequências de teclas
def extrair_sequencias_teclas(df):
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
    return sequencias

# Função para preparar dados para a LSTM
def preparar_dados_lstm(sequencias, janela=3):
    teclas_possiveis = ['q', 'w', 'e', 't', 'r', 'a', 's']
    label_encoder = LabelEncoder()
    label_encoder.fit(teclas_possiveis)
    sequencias_codificadas = [[label_encoder.transform([t])[0] for t in seq] for seq in sequencias]
    X, y = [], []
    for i in range(len(sequencias_codificadas) - janela):
        entrada = [seq[0] for seq in sequencias_codificadas[i:i + janela]]
        saida = sequencias_codificadas[i + janela][0]
        X.append(entrada)
        y.append(saida)
    X = np.array(X, dtype=np.float32).reshape(-1, janela, 1)
    y = np.array(y, dtype=np.int32)
    return X, y, label_encoder

# Função para criar e treinar o modelo
def criar_e_treinar_modelo(X, y, num_classes, janela=3, epochs=50):
    model = Sequential([
        Embedding(input_dim=num_classes, output_dim=10, input_length=janela),
        LSTM(50, return_sequences=False),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("Sou a LSTM! Iniciando o treinamento do modelo...")
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=1)
    print("Treinamento concluído!")
    return model

# Função para prever a próxima tecla
def prever_proxima_tecla(model, ultimos_turnos, label_encoder, janela=3):
    print(f"Sou a LSTM! Analisando a sequência: {ultimos_turnos}")
    turnos_codificados = [label_encoder.transform([t[0]])[0] for t in ultimos_turnos]
    turnos_codificados = np.array(turnos_codificados, dtype=np.float32).reshape(1, janela, 1)
    predicao = model.predict(turnos_codificados)
    tecla_idx = np.argmax(predicao, axis=1)[0]
    tecla = label_encoder.inverse_transform([tecla_idx])[0]
    print(f"Com base nos jogos anteriores, previ que você provavelmente vai usar a tecla: {tecla}")
    return tecla

# Função principal para processar CSV e prever
def processar_e_prever(csv_path, janela=3, epochs=50):
    df = carregar_dados_csv(csv_path)
    if df is None or len(df) < janela + 1:
        print("Sou a LSTM! Não há dados suficientes para previsão.")
        return None, None, None
    sequencias = extrair_sequencias_teclas(df)
    print(f"Sou a LSTM! Encontrei {len(sequencias)} sequências válidas no CSV.")
    X, y, label_encoder = preparar_dados_lstm(sequencias, janela)
    num_classes = len(label_encoder.classes_)
    model = criar_e_treinar_modelo(X, y, num_classes, janela, epochs)
    ultimos_turnos = sequencias[-janela:]
    proxima_tecla = prever_proxima_tecla(model, ultimos_turnos, label_encoder, janela)
    return proxima_tecla, model, label_encoder

# Função para ser chamada pelo server.py
def prever_tecla_server(csv_path, model=None, label_encoder=None, ultimos_turnos=None, janela=3):
    if model is None or label_encoder is None:
        print("Sou a LSTM! Nenhum modelo pré-treinado encontrado, iniciando novo treinamento...")
        return processar_e_prever(csv_path, janela)
    else:
        print("Sou a LSTM! Usando modelo pré-treinado para previsão.")
        return prever_proxima_tecla(model, ultimos_turnos, label_encoder, janela), model, label_encoder

if __name__ == "__main__":
    csv_path = "Dados/Player1.csv"
    proxima_tecla, _, _ = processar_e_prever(csv_path)
    if proxima_tecla:
        print(f"Próxima tecla prevista: {proxima_tecla}")
    else:
        print("Não há dados suficientes para previsão.")