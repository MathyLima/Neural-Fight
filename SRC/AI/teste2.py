import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Multiply, Lambda
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import matplotlib.pyplot as plt

# Função para carregar e preparar os dados do CSV
def load_and_prepare_data(csv_file_path):
    """
    Carrega os dados do CSV e faz o pré-processamento básico
    """
    # Carregar dados do CSV
    df = pd.read_csv(csv_file_path)
    
    # Verificar dados
    print(f"Dados carregados: {df.shape[0]} entradas com {df.shape[1]} colunas")
    
    return df

# Função para preprocessar dados e criar sequências
def prepare_sequential_data(df, lookback=2):
    """
    Prepara os dados para o modelo sequencial de 4 steps (4 teclas por turno)
    com máscaras para teclas disponíveis
    
    Args:
        df: DataFrame com os dados do jogo
        lookback: Número de turnos anteriores a considerar
    
    Returns:
        Dados formatados para treinamento do modelo
    """
    # Colunas alvo (teclas escolhidas)
    target_columns = ['primeiraTeclaJogador1', 'segundaTeclaJogador1', 
                      'terceiraTeclaJogador1', 'quartaTeclaJogador1']
    
    # Colunas de disponibilidade das teclas
    teclas_disponiveis_columns = [f'teclaDisponivel_{i}' for i in range(1, 8)]
    
    # Colunas categóricas que precisarão de encoding
    categorical_columns = ['turnoJogador1']
    
    # Colunas que serão usadas como features para prever as teclas
    # Exclui as colunas alvo e outras não relevantes
    feature_columns = [col for col in df.columns if col not in target_columns + ['id']]
    
    # Normalizar features numéricas
    numeric_features = [f for f in feature_columns if f not in categorical_columns 
                        and df[f].dtype in [np.int64, np.float64]]
    
    scaler = StandardScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])
    
    # One-hot encoding para variáveis categóricas
    if categorical_columns:
        encoder = OneHotEncoder(sparse_output=False)
        encoded_cats = encoder.fit_transform(df[categorical_columns])
        encoded_df = pd.DataFrame(
            encoded_cats, 
            columns=[f"{col}_{i}" for col, vals in zip(categorical_columns, encoder.categories_) 
                    for i in vals]
        )
        
        # Juntar o DataFrame original com o encoding
        df_processed = pd.concat([df.drop(categorical_columns, axis=1).reset_index(drop=True), 
                                encoded_df.reset_index(drop=True)], axis=1)
    else:
        df_processed = df.copy()
    
    # Atualizar feature_columns com as novas colunas após one-hot encoding
    feature_columns = [col for col in df_processed.columns 
                       if col not in target_columns + ['id'] 
                       and col in feature_columns + list(encoded_df.columns if 'encoded_df' in locals() else [])]
    
    # Preparar sequências por turno, com 4 steps por turno (um para cada tecla)
    sequences = []
    
    for i in range(lookback, len(df)):
        # Sequência de turnos anteriores
        prev_turns = df_processed.iloc[i-lookback:i][feature_columns].values
        
        # Teclas escolhidas no turno atual
        tecla1 = df.iloc[i]['primeiraTeclaJogador1']
        tecla2 = df.iloc[i]['segundaTeclaJogador1']
        tecla3 = df.iloc[i]['terceiraTeclaJogador1']
        tecla4 = df.iloc[i]['quartaTeclaJogador1']
        
        # Máscara de teclas disponíveis para o turno atual
        mask_teclas = np.zeros(7)  # num_classes = 7
        for j in range(1, 8):
            if df.iloc[i][f'teclaDisponivel_{j}'] == 1:  # Se for realmente binário
                mask_teclas[j-1] = 1  # j-1 porque o índice das teclas começa em 0        
        # Step 1: Prever primeira tecla baseado apenas no histórico
        X_step1 = prev_turns.copy()
        y_step1 = tecla1
        mask1 = mask_teclas.copy()
        
        # Step 2: Histórico + primeira tecla escolhida
        # Criamos uma cópia do histórico e adicionamos a informação da primeira tecla
        X_step2 = np.vstack([
            prev_turns, 
            np.append([tecla1], np.zeros(len(feature_columns)-1)).reshape(1, -1)
        ])
        y_step2 = tecla2
        # Atualizar máscara para o segundo passo (pode desativar a tecla já escolhida)
        mask2 = mask_teclas.copy()
        
        # Step 3: Histórico + primeira e segunda teclas escolhidas
        X_step3 = np.vstack([
            X_step2,
            np.append([tecla2], np.zeros(len(feature_columns)-1)).reshape(1, -1)
        ])
        y_step3 = tecla3
        # Atualizar máscara para o terceiro passo
        mask3 = mask2.copy()
        
        # Step 4: Histórico + três primeiras teclas escolhidas
        X_step4 = np.vstack([
            X_step3,
            np.append([tecla3], np.zeros(len(feature_columns)-1)).reshape(1, -1)
        ])
        y_step4 = tecla4
        # Atualizar máscara para o quarto passo
        mask4 = mask3.copy()
        
        sequences.append({
            'X_step1': X_step1,
            'y_step1': y_step1,
            'mask1': mask1,
            'X_step2': X_step2,
            'y_step2': y_step2,
            'mask2': mask2,
            'X_step3': X_step3,
            'y_step3': y_step3,
            'mask3': mask3,
            'X_step4': X_step4,
            'y_step4': y_step4,
            'mask4': mask4
        })
    
    # Converter para arrays NumPy
    X1 = np.array([seq['X_step1'] for seq in sequences])
    y1 = np.array([seq['y_step1'] for seq in sequences])
    mask1 = np.array([seq['mask1'] for seq in sequences])
    
    X2 = np.array([seq['X_step2'] for seq in sequences])
    y2 = np.array([seq['y_step2'] for seq in sequences])
    mask2 = np.array([seq['mask2'] for seq in sequences])
    
    X3 = np.array([seq['X_step3'] for seq in sequences])
    y3 = np.array([seq['y_step3'] for seq in sequences])
    mask3 = np.array([seq['mask3'] for seq in sequences])
    
    X4 = np.array([seq['X_step4'] for seq in sequences])
    y4 = np.array([seq['y_step4'] for seq in sequences])
    mask4 = np.array([seq['mask4'] for seq in sequences])
    
    # Determinar o número de classes de teclas (assumindo que começam de 1)
    num_classes = max(
        df['primeiraTeclaJogador1'].max(),
        df['segundaTeclaJogador1'].max(),
        df['terceiraTeclaJogador1'].max(),
        df['quartaTeclaJogador1'].max()
    )
    
    # Converter teclas para one-hot encoding
    y1_cat = to_categorical(y1 - 1, num_classes=num_classes)  # -1 para ajustar índice 0-6
    y2_cat = to_categorical(y2 - 1, num_classes=num_classes)
    y3_cat = to_categorical(y3 - 1, num_classes=num_classes)
    y4_cat = to_categorical(y4 - 1, num_classes=num_classes)
    
    print(f"Formato dos dados de entrada para Step 1: {X1.shape}")
    print(f"Formato das máscaras para Step 1: {mask1.shape}")
    print(f"Número de classes para as teclas: {num_classes}")
    
    return {
        'X1': X1, 'y1': y1_cat, 'mask1': mask1,
        'X2': X2, 'y2': y2_cat, 'mask2': mask2,
        'X3': X3, 'y3': y3_cat, 'mask3': mask3,
        'X4': X4, 'y4': y4_cat, 'mask4': mask4,
        'num_classes': num_classes
    }

def create_masked_multi_step_model(input_shapes, n_classes):
    """
    Cria um modelo LSTM para prever a sequência de 4 teclas com máscaras para teclas disponíveis
   
    Args:
        input_shapes: Dicionário com os formatos de entrada para cada step
        n_classes: Número de classes (teclas possíveis)
    """
    # Input para cada step (sequências)
    input1 = Input(shape=(input_shapes['X1'][1], input_shapes['X1'][2]), name='input_step1')
    input2 = Input(shape=(input_shapes['X2'][1], input_shapes['X2'][2]), name='input_step2')
    input3 = Input(shape=(input_shapes['X3'][1], input_shapes['X3'][2]), name='input_step3')
    input4 = Input(shape=(input_shapes['X4'][1], input_shapes['X4'][2]), name='input_step4')
   
    # Input para as máscaras de teclas disponíveis
    mask_input1 = Input(shape=(n_classes,), name='mask_input1')
    mask_input2 = Input(shape=(n_classes,), name='mask_input2')
    mask_input3 = Input(shape=(n_classes,), name='mask_input3')
    mask_input4 = Input(shape=(n_classes,), name='mask_input4')
   
    # Camadas LSTM compartilhadas
    lstm1 = LSTM(128, return_sequences=True)
    lstm2 = LSTM(64)
   
    # Processamento para cada step
    # Step 1
    x1 = lstm1(input1)
    x1 = Dropout(0.2)(x1)
    x1 = lstm2(x1)
    x1 = Dropout(0.2)(x1)
    x1 = Dense(64, activation='relu')(x1)
    pred1 = Dense(n_classes, activation='softmax')(x1)
   
    # Aplicar máscara (desativar teclas não disponíveis)
    # Multiplicamos pela máscara e renormalizamos
    masked_pred1 = Multiply()([pred1, mask_input1])
    output1 = Lambda(
        lambda x: x / tf.keras.backend.sum(x, axis=-1, keepdims=True) + 1e-10,
        output_shape=lambda input_shape: input_shape,
        name='output1'
    )(masked_pred1)
   
    # Step 2
    x2 = lstm1(input2)
    x2 = Dropout(0.2)(x2)
    x2 = lstm2(x2)
    x2 = Dropout(0.2)(x2)
    x2 = Dense(64, activation='relu')(x2)
    pred2 = Dense(n_classes, activation='softmax')(x2)
   
    # Aplicar máscara
    masked_pred2 = Multiply()([pred2, mask_input2])
    output2 = Lambda(
        lambda x: x / tf.keras.backend.sum(x, axis=-1, keepdims=True) + 1e-10,
        output_shape=lambda input_shape: input_shape,
        name='output2'
    )(masked_pred2)
   
    # Step 3
    x3 = lstm1(input3)
    x3 = Dropout(0.2)(x3)
    x3 = lstm2(x3)
    x3 = Dropout(0.2)(x3)
    x3 = Dense(64, activation='relu')(x3)
    pred3 = Dense(n_classes, activation='softmax')(x3)
   
    # Aplicar máscara
    masked_pred3 = Multiply()([pred3, mask_input3])
    output3 = Lambda(
        lambda x: x / tf.keras.backend.sum(x, axis=-1, keepdims=True) + 1e-10,
        output_shape=lambda input_shape: input_shape,
        name='output3'
    )(masked_pred3)
   
    # Step 4
    x4 = lstm1(input4)
    x4 = Dropout(0.2)(x4)
    x4 = lstm2(x4)
    x4 = Dropout(0.2)(x4)
    x4 = Dense(64, activation='relu')(x4)
    pred4 = Dense(n_classes, activation='softmax')(x4)
   
    # Aplicar máscara
    masked_pred4 = Multiply()([pred4, mask_input4])
    output4 = Lambda(
        lambda x: x / tf.keras.backend.sum(x, axis=-1, keepdims=True) + 1e-10,
        output_shape=lambda input_shape: input_shape,
        name='output4'
    )(masked_pred4)
   
    # Criar modelo com múltiplas entradas e saídas
    model = Model(
        inputs=[input1, input2, input3, input4, mask_input1, mask_input2, mask_input3, mask_input4],
        outputs=[output1, output2, output3, output4]
    )
   
    # Compilar modelo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(clipnorm=1.0),
        loss={
            'output1': 'categorical_crossentropy',
            'output2': 'categorical_crossentropy',
            'output3': 'categorical_crossentropy',
            'output4': 'categorical_crossentropy'
        },
        metrics=['accuracy']*4
    )
   
    return model

def train_model(data, epochs=50, batch_size=32, validation_split=0.2):
    """
    Treina o modelo com os dados preparados
    
    Args:
        data: Dicionário com os dados preparados pela função prepare_sequential_data
        epochs: Número de épocas de treinamento
        batch_size: Tamanho do batch
        validation_split: Fração dos dados usada para validação
    """

    print(data)
    # Extrair dados
    X1, y1, mask1 = data['X1'], data['y1'], data['mask1']
    X2, y2, mask2 = data['X2'], data['y2'], data['mask2']
    X3, y3, mask3 = data['X3'], data['y3'], data['mask3']
    X4, y4, mask4 = data['X4'], data['y4'], data['mask4']
    num_classes = data['num_classes']
    
    # Dividir em treino e validação
    indices = np.arange(X1.shape[0])
    np.random.shuffle(indices)
    split_idx = int(X1.shape[0] * (1 - validation_split))
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    # Criar formato de entrada para o modelo
    input_shapes = {
        'X1': X1.shape,
        'X2': X2.shape,
        'X3': X3.shape,
        'X4': X4.shape
    }
    
    # Criar modelo
    model = create_masked_multi_step_model(input_shapes, num_classes)
    
    # Early stopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Treinar modelo
    history = model.fit(
        [X1[train_indices], X2[train_indices], X3[train_indices], X4[train_indices],
         mask1[train_indices], mask2[train_indices], mask3[train_indices], mask4[train_indices]],
        [y1[train_indices], y2[train_indices], y3[train_indices], y4[train_indices]],
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(
            [X1[val_indices], X2[val_indices], X3[val_indices], X4[val_indices],
             mask1[val_indices], mask2[val_indices], mask3[val_indices], mask4[val_indices]],
            [y1[val_indices], y2[val_indices], y3[val_indices], y4[val_indices]]
        ),
        callbacks=[early_stop]
    )
    
    return model, history

def plot_training_history(history):
    """
    Plota o histórico de treinamento
    """
    # Plotar loss
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plotar acurácia
    plt.subplot(1, 2, 2)
    plt.plot(history.history['lambda_accuracy'], label='Step 1 Accuracy')
    plt.plot(history.history['lambda_1_accuracy'], label='Step 2 Accuracy')
    plt.plot(history.history['lambda_2_accuracy'], label='Step 3 Accuracy')
    plt.plot(history.history['lambda_3_accuracy'], label='Step 4 Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def predict_next_moves(model, game_state, teclas_disponiveis):
    """
    Prevê as próximas 4 teclas baseado no estado atual do jogo
    
    Args:
        model: Modelo treinado
        game_state: Estado atual do jogo (mesmas features usadas no treinamento)
        teclas_disponiveis: Array indicando quais teclas (1-7) estão disponíveis
        
    Returns:
        Lista das 4 teclas previstas
    """
    # Preparar a entrada (assumindo que o formato é o mesmo usado no treinamento)
    X1 = np.expand_dims(game_state, axis=0)
    
    # Preparar máscaras
    mask1 = np.expand_dims(teclas_disponiveis, axis=0)
    
    # Para os próximos steps, precisamos adicionar as teclas previstas anteriormente
    # Inicializar com qualquer valor, será atualizado após cada previsão
    X2 = np.concatenate([X1, np.zeros((1, 1, X1.shape[2]))], axis=1)
    X3 = np.concatenate([X2, np.zeros((1, 1, X1.shape[2]))], axis=1)
    X4 = np.concatenate([X3, np.zeros((1, 1, X1.shape[2]))], axis=1)
    
    mask2 = mask1.copy()
    mask3 = mask1.copy()
    mask4 = mask1.copy()
    
    # Fazer previsão do primeiro passo
    predictions = model.predict([X1, X2, X3, X4, mask1, mask2, mask3, mask4])
    
    # Extrair resultados
    tecla1_probs = predictions[0][0]
    tecla1 = np.argmax(tecla1_probs) + 1  # +1 pois as classes são 1-7
    
    # Atualizar a entrada para o segundo passo
    X2[0, -1, 0] = tecla1
    
    # Refazer a previsão para obter os próximos passos atualizados
    predictions = model.predict([X1, X2, X3, X4, mask1, mask2, mask3, mask4])
    
    tecla2_probs = predictions[1][0]
    tecla2 = np.argmax(tecla2_probs) + 1
    
    # Atualizar a entrada para o terceiro passo
    X3[0, -1, 0] = tecla2
    
    # Refazer a previsão
    predictions = model.predict([X1, X2, X3, X4, mask1, mask2, mask3, mask4])
    
    tecla3_probs = predictions[2][0]
    tecla3 = np.argmax(tecla3_probs) + 1
    
    # Atualizar a entrada para o quarto passo
    X4[0, -1, 0] = tecla3
    
    # Refazer a previsão
    predictions = model.predict([X1, X2, X3, X4, mask1, mask2, mask3, mask4])
    
    tecla4_probs = predictions[3][0]
    tecla4 = np.argmax(tecla4_probs) + 1
    
    return [tecla1, tecla2, tecla3, tecla4]

def main():
    """
    Função principal para executar o fluxo completo
    """
    # Carregar dados
    csv_file_path = "Dados/Player1Modificado.csv"  # Substitua pelo caminho do seu arquivo
    df = load_and_prepare_data(csv_file_path)
    
    # Preparar dados para o modelo
    lookback = 3  # Considerar 3 turnos anteriores
    data = prepare_sequential_data(df, lookback=lookback)
    
    # Treinar modelo
    model, history = train_model(data, epochs=100, batch_size=32, validation_split=0.2)
    
    # Plotar histórico de treinamento
    plot_training_history(history)
    
    # Salvar modelo
    model.save("model_teclas_jogo.h5")
    print("Modelo salvo como 'model_teclas_jogo.h5'")
    
    # Demonstração de previsão (usar um exemplo dos dados de teste)
    # Aqui você precisaria extrair um estado de jogo real e suas teclas disponíveis
    
    return model

if __name__ == "__main__":
    main()