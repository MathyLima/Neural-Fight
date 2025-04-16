import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Array de entrada
dados = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1]
tamanho_sequencia = 2  # Usar 2 valores anteriores para prever o próximo
epochs = 50

# Mapear valores para índices (0, 1, 2) e obter número de classes
valores_unicos = sorted(set(dados))
num_classes = len(valores_unicos)
valor_para_indice = {v: i for i, v in enumerate(valores_unicos)}  # Ex.: {1: 0, 2: 1, 3: 2}
indice_para_valor = {i: v for v, i in valor_para_indice.items()}  # Ex.: {0: 1, 1: 2, 2: 3}

# Converter dados para índices
dados_codificados = [valor_para_indice[v] for v in dados]

# Preparar dados
X = []
y = []
for i in range(len(dados_codificados) - tamanho_sequencia):
    X.append(dados_codificados[i:i + tamanho_sequencia])
    y.append(dados_codificados[i + tamanho_sequencia])

X = np.array(X, dtype=np.float32).reshape(-1, tamanho_sequencia, 1)  # Formato: (amostras, timesteps, features)
y = np.array(y, dtype=np.int32)  # Índices para sparse_categorical_crossentropy

# Criar modelo LSTM simples
model = Sequential()
model.add(LSTM(10, input_shape=(tamanho_sequencia, 1)))  # 10 unidades na LSTM
model.add(Dense(num_classes, activation='softmax'))  # Saída com probabilidade para cada classe
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Treinar
print("Treinando o modelo...")
model.fit(X, y, epochs=epochs, verbose=1)

# Prever o próximo valor
ultima_sequencia = np.array(dados_codificados[-tamanho_sequencia:], dtype=np.float32).reshape(1, tamanho_sequencia, 1)
predicao = model.predict(ultima_sequencia)
indice_predito = np.argmax(predicao, axis=1)[0]
valor_predito = indice_para_valor[indice_predito]
print(f"Próximo valor previsto: {valor_predito}")