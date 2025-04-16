# lstm.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

def treinar_lstm(caminho_csv, tamanho_sequencia=2, epochs=100):
    df = pd.read_csv(caminho_csv)
    acoes = df["acao"].tolist()

    X = []
    y = []

    for i in range(len(acoes) - tamanho_sequencia):
        entrada = acoes[i:i + tamanho_sequencia]
        saida = acoes[i + tamanho_sequencia]
        X.append(entrada)
        y.append(saida)

    X = np.array(X)
    y = np.array(y)

    num_acoes = len(set(acoes))

    model = Sequential()
    model.add(Embedding(input_dim=num_acoes, output_dim=10, input_length=tamanho_sequencia))
    model.add(LSTM(50))
    model.add(Dense(num_acoes, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, epochs=epochs, verbose=1)

    return model
