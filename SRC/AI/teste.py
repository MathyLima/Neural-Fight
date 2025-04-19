import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

class Model:
    def __init__(self, df):
        self.df = df

    def normaliza_dados(self):
        """
        Normaliza os dados de entrada, convertendo colunas categóricas e mantendo apenas as numéricas.
        """
        if 'turnoJogador1' in self.df.columns:
            self.df['turnoJogador1'] = self.df['turnoJogador1'].map({'ataque': 0, 'defesa': 1})

        colunas_numericas = self.df.select_dtypes(include=[np.number]).columns
        self.df = self.df[colunas_numericas]
        return self.df

    # 1. Be more selective with features
    def gera_sequencias_acumuladas(self, colunaDesejada):
        X_geral, y_geral = [], []
        
        # Define relevant features that actually help predict the next key press
        relevant_features = [
            'vida_jogador1', 'vida_jogador2', 
            'primeiraTeclaJogador1', 'segundaTeclaJogador1', 
            'terceiraTeclaJogador1', 'quartaTeclaJogador1',
            'turnoJogador1',  # Keeping this as it indicates attack/defense
            'teclaDisponivel_1', 'teclaDisponivel_2', 'teclaDisponivel_3',
            'teclaDisponivel_4', 'teclaDisponivel_5', 'teclaDisponivel_6', 'teclaDisponivel_7'
        ]
        
        for i in range(1, len(self.df) - 1):
            sequencia = []
            
            # Limited context window - use last 5 turns instead of all
            for j in range(max(0, i-5), i):
                linha = self.df.iloc[j]
                sequencia.append([linha[feature] for feature in relevant_features])
            
            X_geral.append(sequencia)
            y_geral.append(self.df.iloc[i][colunaDesejada])
        
        # Padding at the end
        max_len = max(len(seq) for seq in X_geral)
        for i in range(len(X_geral)):
            # Use padding that makes more sense for your game state
            padding_values = [0] * len(relevant_features)
            while len(X_geral[i]) < max_len:
                X_geral[i].append(padding_values)  # Pad at the end with zeros
        
        return np.array(X_geral, dtype=np.float32), np.array(y_geral, dtype=np.int32)


# Carregando o DataFrame
df = pd.read_csv("Dados/Player1.csv")

# Instanciando e processando os dados
modelo = Model(df)
df_normalizado = modelo.normaliza_dados()
coluna_tecla = 'primeiraTeclaJogador1'

# Gera sequências acumuladas
X, y = modelo.gera_sequencias_acumuladas(colunaDesejada=coluna_tecla)
y-=1
y = y.astype(np.int32)

# Separando dados para treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

# Verificações de integridade
print(f"Shape de X: {X.shape}")
print(f"Shape de y: {y.shape}")
print("Classe mais comum:", pd.Series(y).value_counts().idxmax())

# Construindo a rede LSTM
model = Sequential([
    Input(shape=(X.shape[1], X.shape[2])),
    LSTM(100),  # Menos unidades
    Dense(7, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

classes = np.unique(y_train)

# Add class weights but don't make them too extreme
class_weights = compute_class_weight(
    class_weight='balanced', 
    classes=classes, 
    y=y_train
)
# Cap extreme weights
class_weights = np.clip(class_weights, 0.5, 5.0)
class_weight_dict = dict(zip(classes, class_weights))

# Use validation split to monitor overfitting
early_stop = EarlyStopping(
    monitor='val_loss',  # Change to val_loss
    patience=10,
    restore_best_weights=True
)

# Add learning rate reduction
lr_reduction = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5,
    patience=5, 
    min_lr=0.00001
)

# Proper validation split
history = model.fit(
    X_train, y_train,
    validation_split=0.2,  # Use validation split
    epochs=100,  # Start with fewer epochs
    batch_size=1,  # Smaller batch size for better learning
    class_weight=class_weight_dict,
    callbacks=[early_stop, lr_reduction],
    verbose=1
)
from sklearn.metrics import classification_report
# Detailed evaluation
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Add 1 back to return to original key values (1-7)
print("Classification report (classes 0-6):")
print(classification_report(y_test, y_pred_classes))

print("Classification report (original keys 1-7):")
print(classification_report(y_test+1, y_pred_classes+1))

# Confusion matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()