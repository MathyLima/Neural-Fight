import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input  # certifique-se de importar
from tensorflow.keras.callbacks import EarlyStopping
# Normalizar a variável de saída (primeiraTeclaJogador1)

class Model:
    def __init__(self, df):
        self.df = df
        self.scaler = None

    def normaliza_dados(self):
        """
        Normaliza os dados de entrada, excluindo a coluna 'partida' e outras variáveis não numéricas.
        """
        # Selecionar apenas as colunas numéricas (excluindo as colunas de teclas e outras categóricas)
        colunas_numericas = self.df.select_dtypes(include=[np.number]).columns

        # Excluir a coluna 'partida' e qualquer outra coluna que não deve ser normalizada
        colunas_numericas = [col for col in colunas_numericas if col != 'partida' and col not in ['primeiraTeclaJogador1', 'segundaTeclaJogador1','terceiraTeclaJogador1','quartaTeclaJogador1']]

        # Criar um novo DataFrame apenas com as colunas numéricas
        dados_numericos = self.df[colunas_numericas]

        # Normalizar os dados
        scaler = StandardScaler()
        dados_normalizados = scaler.fit_transform(dados_numericos)

        # Substituir as colunas normalizadas no DataFrame original
        self.df[colunas_numericas] = dados_normalizados

        return self.df, scaler

    @staticmethod
    def transforma_rodada_em_vetor(rodada):
        """
        Recebe uma linha (série/pandas) e transforma em vetor numérico (array).
        """
        # Remove colunas categóricas se houver
        vetor = rodada.select_dtypes(include=[np.number]).values.astype(np.float32)
        return vetor

    def cria_sequencias_por_rodada(self, jogo):
        X, y = [], []
        # Resetar o índice para garantir continuidade
        jogo = jogo.reset_index(drop=True)

        # Criar as sequências para cada linha
        for i in range(len(jogo) - 1):  # Até a penúltima linha
            entrada = [
                jogo.iloc[i]['primeiraTeclaJogador1'],
                jogo.iloc[i]['segundaTeclaJogador1'],
                jogo.iloc[i]['terceiraTeclaJogador1'],
                jogo.iloc[i]['quartaTeclaJogador1'],
                jogo.iloc[i]['vida_jogador1'],
                jogo.iloc[i]['vida_jogador2'],
                jogo.iloc[i]['cooldown_jogador1_q'],
                jogo.iloc[i]['cooldown_jogador1_w'],
                jogo.iloc[i]['cooldown_jogador1_e'],
                jogo.iloc[i]['cooldown_jogador1_r'],
                jogo.iloc[i]['cooldown_jogador1_t'],
                jogo.iloc[i]['cooldown_jogador1_a'],
                jogo.iloc[i]['cooldown_jogador1_s'],
            ]
            X.append(entrada)

            # O próximo valor (target) é o valor da próxima linha
            proximo_input = jogo.iloc[i + 1]["primeiraTeclaJogador1"]
            y.append(proximo_input)

        return np.array(X, dtype=np.float32), np.array(y)

    def gera_todas_sequencias(self):
        """
        Cria todas as sequências sem separar por partida.
        """
        X_geral, y_geral = [], []

        # Garantir que a coluna 'partida' exista
        if 'partida' not in self.df.columns:
            raise ValueError("Coluna 'partida' não encontrada no DataFrame.")

        # Remover a divisão por partida e simplesmente iterar sobre o DataFrame
        # Resetar o índice para garantir continuidade
        jogo = self.df.reset_index(drop=True)

        print(f"Tamanho do DataFrame: {len(jogo)}")

        # Criar as sequências para cada linha
        for i in range(len(jogo) - 1):  # Até a penúltima linha
            entrada = [
                jogo.iloc[i]['primeiraTeclaJogador1'],
                jogo.iloc[i]['segundaTeclaJogador1'],
                jogo.iloc[i]['terceiraTeclaJogador1'],
                jogo.iloc[i]['quartaTeclaJogador1'],
                jogo.iloc[i]['vida_jogador1'],
                jogo.iloc[i]['vida_jogador2'],
                jogo.iloc[i]['cooldown_jogador1_q'],
                jogo.iloc[i]['cooldown_jogador1_w'],
                jogo.iloc[i]['cooldown_jogador1_e'],
                jogo.iloc[i]['cooldown_jogador1_r'],
                jogo.iloc[i]['cooldown_jogador1_t'],
                jogo.iloc[i]['cooldown_jogador1_a'],
                jogo.iloc[i]['cooldown_jogador1_s'],
            ]
            X_geral.append(entrada)

            # O próximo valor (target) é o valor da próxima linha
            proximo_input = jogo.iloc[i + 1]["primeiraTeclaJogador1"]
            y_geral.append(proximo_input)

        # Concatenar tudo em um único array
        X_final = np.array(X_geral, dtype=np.float32)
        y_final = np.array(y_geral, dtype=np.int32)

        # Ajustar a forma de X para a LSTM (num_sequencias, 1, num_features)
        X_final = X_final.reshape(X_final.shape[0], 1, X_final.shape[1])

        return X_final, y_final


# Supondo que você já tenha carregado seu DataFrame (df)
df = pd.read_csv("Dados/Player1.csv")  # Ajuste para o caminho correto do seu arquivo

# Criação do modelo com os dados carregados
modelo = Model(df)

# Normaliza os dados
df_normalizado, scaler = modelo.normaliza_dados()

# Cria as sequências para a LSTM
X, y = modelo.gera_todas_sequencias()
y-=1
y = y.astype(np.int32)
# Convertendo y para classes discretas (1 a 7)
# Verifica o formato dos dados
print("Shape de X:", X.shape)  # Espera algo como (num_sequencias, 1, num_features)
print("Shape de y:", y.shape)  # Espera algo como (num_sequencias, )
print("Valores de y:", y)
print("Tipo dos elementos:", [type(val) for val in y])
print("Maior valor em y:", np.max(y))
print("Menor valor em y:", np.min(y))
print("Tem NaN?", np.isnan(y).any())
print("Tem Inf?", np.isinf(y).any())


print("Máximo de X:", np.max(X))
print("Mínimo de X:", np.min(X))
print("Máximo de y:", np.max(y))
print("Mínimo de y:", np.min(y))


print("Valores máximos absolutos:")
print("Max em y:", np.max(y))
print("Min em y:", np.min(y))

print("Valores muito grandes em y:", np.where(y > 2**30))  # índice se houver

print("Tem 2147483648 em y?", np.any(y == 2147483648))
print("Tem 2147483648 em X?", np.any(X == 2147483648))

print(df.dtypes)
print(df.head())
# Construindo a LSTM com TensorFlow
model = Sequential()
model.add(Input(shape=(X.shape[1], X.shape[2])))  # (1, 13)
model.add(LSTM(units=20))  # tanh padrão
model.add(Dropout(0.2))
model.add(Dense(units=7, activation='softmax'))  # 7 classes

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

early_stop = EarlyStopping(
    monitor='loss',       # ou 'val_loss' se tiver validação
    patience=20,          # para de treinar se não melhorar em 20 épocas
    restore_best_weights=True
)

model.fit(X, y, epochs=1000, batch_size=16, callbacks=[early_stop])

# Avaliando o modelo (ajuste conforme necessário)
loss, accuracy = model.evaluate(X, y)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# Fazendo uma previsão com os dados de entrada
previsao = model.predict(X)

# Convertendo as previsões de volta para valores discretos
previsao_classes = np.argmax(previsao, axis=1) + 1  # As classes vão de 1 a 7

print(f"Previsões (valores discretos de 1 a 7): {previsao_classes}")

