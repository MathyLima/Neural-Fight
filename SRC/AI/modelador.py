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

# Usa todos os núcleos disponíveis
tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count())
tf.config.threading.set_inter_op_parallelism_threads(os.cpu_count())
# Verificar e criar diretório para salvar modelos e resultados
if not os.path.exists('modelos'):
    os.makedirs('modelos')
if not os.path.exists('resultados'):
    os.makedirs('resultados')

# Classe de processamento com nova função para sequências completas
class Modelador:
    def __init__(self, df):
        self.df = df

    def normaliza_dados(self):
        if 'turnoJogador1' in self.df.columns:
            self.df['turnoJogador1'] = self.df['turnoJogador1'].map({'ataque': 0, 'defesa': 1})

        colunas_numericas = self.df.select_dtypes(include=[np.number]).columns
        self.df = self.df[colunas_numericas]
        return self.df

    def gera_sequencias_por_partida(self, colunas_desejadas):
        """Abordagem por partida - cada amostra é um estado da partida"""
        X_geral, y_geral = [], []
        features = [ 'vida_jogador1', 'vida_jogador2', 'turnoJogador1',
            'teclaDisponivel_1', 'teclaDisponivel_2', 'teclaDisponivel_3',
            'teclaDisponivel_4', 'teclaDisponivel_5', 'teclaDisponivel_6', 'teclaDisponivel_7',
            'delta_vida_j1', 'delta_vida_j2', 'ratio_vida', 'total_teclas_disponiveis',
            'tecla_anterior_1', 'tecla_anterior_2', 'tecla_anterior_3', 'tecla_anterior_4']
        
        # Agrupar por partida
        partidas = self.df['partida'].unique()
        
        for partida_id in partidas:
            df_partida = self.df[self.df['partida'] == partida_id].reset_index(drop=True)
            
     
            for i in range(len(df_partida) - 1):
    
                estado_atual = []
                for feature in features:
                    if feature in df_partida.columns:
                        estado_atual.append(df_partida.iloc[i][feature])
                    else:
                        estado_atual.append(0)
                
 
                proximas_teclas = df_partida.iloc[i+1][colunas_desejadas].values - 1
                
                X_geral.append([estado_atual]) 
                y_geral.append(proximas_teclas)
        
        return np.array(X_geral, dtype=np.float32), np.array(y_geral, dtype=np.int32)
    
    def gera_sequencias_com_contexto(self, colunas_desejadas, contexto=3):
        """Abordagem com contexto - considera N estados anteriores"""
        X_geral, y_geral = [], []
        features = [ 'vida_jogador1', 'vida_jogador2', 'turnoJogador1',
            'teclaDisponivel_1', 'teclaDisponivel_2', 'teclaDisponivel_3',
            'teclaDisponivel_4', 'teclaDisponivel_5', 'teclaDisponivel_6', 'teclaDisponivel_7',
            'delta_vida_j1', 'delta_vida_j2', 'ratio_vida', 'total_teclas_disponiveis',
            'tecla_anterior_1', 'tecla_anterior_2', 'tecla_anterior_3', 'tecla_anterior_4']
        
        # Agrupar por partida
        partidas = self.df['partida'].unique()
        
        for partida_id in partidas:
            df_partida = self.df[self.df['partida'] == partida_id].reset_index(drop=True)
            
            # Para cada momento da partida que tenha contexto suficiente
            for i in range(contexto, len(df_partida) - 1):
                # Obter contexto recente (últimos N estados)
                sequencia = []
                for j in range(i-contexto, i+1):
                    estado = []
                    for feature in features:
                        if feature in df_partida.columns:
                            estado.append(df_partida.iloc[j][feature])
                        else:
                            estado.append(0)
                    sequencia.append(estado)
                
                proximas_teclas = df_partida.iloc[i+1][colunas_desejadas].values - 1
                
                X_geral.append(sequencia)
                y_geral.append(proximas_teclas)
        
        return np.array(X_geral, dtype=np.float32), np.array(y_geral, dtype=np.int32)
    
    def gera_sequencias_completas(self, colunas_desejadas, max_seq_len=None):
        """Abordagem que considera todas as rodadas anteriores da partida"""
        X_geral, y_geral = [], []
        features = [ 'vida_jogador1', 'vida_jogador2', 'turnoJogador1',
            'teclaDisponivel_1', 'teclaDisponivel_2', 'teclaDisponivel_3',
            'teclaDisponivel_4', 'teclaDisponivel_5', 'teclaDisponivel_6', 'teclaDisponivel_7',
            'delta_vida_j1', 'delta_vida_j2', 'ratio_vida', 'total_teclas_disponiveis',
            'tecla_anterior_1', 'tecla_anterior_2', 'tecla_anterior_3', 'tecla_anterior_4']
        
   
        partidas = self.df['partida'].unique()
        
 
        if max_seq_len is None:
            max_seq_len = max([len(self.df[self.df['partida'] == p]) for p in partidas])
            print(f"Comprimento máximo de sequência detectado: {max_seq_len}")
        
        for partida_id in partidas:
            df_partida = self.df[self.df['partida'] == partida_id].reset_index(drop=True)

            for i in range(1, len(df_partida) - 1):
                sequencia = []
                for j in range(max(0, i - max_seq_len + 1), i + 1):
                    estado = []
                    for feature in features:
                        if feature in df_partida.columns:
                            estado.append(df_partida.iloc[j][feature])
                        else:
                            estado.append(0)
                    sequencia.append(estado)
                
             
                while len(sequencia) < max_seq_len:
               
                    sequencia.insert(0, [0.0] * len(features))
        
                proximas_teclas = df_partida.iloc[i+1][colunas_desejadas].values - 1
                
                X_geral.append(sequencia)
                y_geral.append(proximas_teclas)
        
        return np.array(X_geral, dtype=np.float32), np.array(y_geral, dtype=np.int32)

    @staticmethod
    def prever_proximas_teclas(modelo, estado_atual, caminho_csv="Dados/Player1.csv", max_seq_len=4):
        import tensorflow as tf
        import numpy as np
        
      
        """
        Faz previsão das próximas teclas a serem pressionadas com base em um estado atual.
        Preenche automaticamente o contexto histórico com dados do CSV.
        
        Args:
            modelo: Modelo carregado para fazer a previsão
            estado_atual (dict): Dicionário com o estado atual (pode conter apenas um parâmetro)
            caminho_csv (str): Caminho para o arquivo CSV com os dados históricos
            max_seq_len (int): Tamanho máximo da sequência esperado pelo modelo
            
        Returns:
            list: Lista com as 4 teclas previstas (índices 0-6)
        """
        # Carregar dados históricos
        df = pd.read_csv(caminho_csv)
        
        if 'turnoJogador1' in df.columns:
            df['turnoJogador1'] = df['turnoJogador1'].map({'ataque': 0, 'defesa': 1})
        
        colunas_numericas = df.select_dtypes(include=[np.number]).columns
        df = df[colunas_numericas]
      
        # Features que o modelo espera, na ordem correta
        features = [
            'vida_jogador1', 'vida_jogador2', 'turnoJogador1',
            'teclaDisponivel_1', 'teclaDisponivel_2', 'teclaDisponivel_3',
            'teclaDisponivel_4', 'teclaDisponivel_5', 'teclaDisponivel_6', 'teclaDisponivel_7',
            'delta_vida_j1', 'delta_vida_j2', 'ratio_vida', 'total_teclas_disponiveis',
            'tecla_anterior_1', 'tecla_anterior_2', 'tecla_anterior_3', 'tecla_anterior_4'
        ]
        
        # Pegar a última partida do CSV
        ultima_partida = df['partida'].max()
        df_ultima_partida = df[df['partida'] == ultima_partida].reset_index(drop=True)
        
      
        sequencia = []
        
        num_estados = min(max_seq_len-1, len(df_ultima_partida))
        for i in range(len(df_ultima_partida) - num_estados, len(df_ultima_partida)):
            estado = []
            for feature in features:
                if feature in df_ultima_partida.columns:
                    estado.append(df_ultima_partida.iloc[i][feature])
                else:
                    estado.append(0)
            sequencia.append(estado)
      
     
        estado_final = []
        for feature in features:
            if feature in estado_atual:
                estado_final.append(estado_atual[feature])
            elif len(sequencia) > 0: 
                ultimo_idx = len(sequencia) - 1
                feature_idx = features.index(feature) if feature in features else -1
                if feature_idx >= 0 and feature_idx < len(sequencia[ultimo_idx]):
                    estado_final.append(sequencia[ultimo_idx][feature_idx])
                else:
                    estado_final.append(0)
            else:
                estado_final.append(0)
        
        sequencia.append(estado_final)
        
        # Aplicar padding para garantir o tamanho correto
        while len(sequencia) < max_seq_len:
            sequencia.insert(0, [0.0] * len(features))
        
        # Garantir que temos exatamente max_seq_len estados
        if len(sequencia) > max_seq_len:
            sequencia = sequencia[-max_seq_len:]
        
        # Converter para array e adicionar dimensão de batch
        X = np.array([sequencia], dtype=np.float32)
        
        # Fazer previsão usando o modelo
        print('Antes do predict')
        # Verificar o tipo do modelo para debug
        print(f"Tipo do modelo: {type(modelo)}")
        
        try:
            previsoes = modelo.predict(X, verbose=0)
       
            
            # Converter saídas para índices de teclas (0-6)
            teclas_previstas = [np.argmax(previsao[0]) for previsao in previsoes]
            
            return teclas_previstas
        except Exception as e:
            print(f"Erro durante predict: {str(e)}")
            # Verificar se estamos usando um wrapper ou modelo direto
            if hasattr(modelo, 'model_'):
                print("Tentando usar modelo._model para predict")
                previsoes = modelo.model_.predict(X, verbose=0)
            else:
                raise e
                
            teclas_previstas = [np.argmax(previsao[0]) for previsao in previsoes]
            return teclas_previstas
            
    @staticmethod
    def carregar_modelo(caminho_modelo):
        """
        Carrega um modelo treinado a partir do caminho especificado (formato .h5 ou .keras).
        
        Args:
            caminho_modelo (str): Caminho para o arquivo do modelo
                
        Returns:
            model: Modelo carregado pronto para fazer previsões
        """
        import tensorflow as tf
        tf.keras.config.enable_unsafe_deserialization()
        
        if not os.path.exists(caminho_modelo):
            raise FileNotFoundError(f"Modelo não encontrado em: {caminho_modelo}")
            
        try:
            # Definir formato esperado da entrada
            input_shape = (4, 18) 
            
            # Reconstruir o modelo do zero para garantir o namespace correto de 'tf'
            modelo = build_model_masked(input_shape)
            
            # Carregar apenas os pesos
            modelo.load_weights(caminho_modelo)
            print(f"Modelo carregado com sucesso de: {caminho_modelo}")
            return modelo
            
        except Exception as e:
            print(f"Erro detalhado ao carregar o modelo: {str(e)}")
            
            # Tente com custom_objects se o carregamento padrão falhar
            try:
                # Definir funções Lambda customizadas explicitamente
                import tensorflow as tf
                
                def extract_last_timestep(x):
                    return x[:, -1, 3:10]
                    
                def apply_mask(inputs):
                    return inputs[0] + tf.math.log(tf.clip_by_value(inputs[1], 1e-9, 1.0)) * 1e9
                    
                custom_objects = {
                    'extract_last_timestep': extract_last_timestep,
                    'apply_mask': apply_mask
                }
                
                # Tentar carregar com objetos customizados
                from tensorflow.keras.models import load_model
                modelo = load_model(caminho_modelo, custom_objects=custom_objects)
                print(f"Modelo carregado com sucesso usando custom_objects")
                return modelo
                
            except Exception as e2:
                print(f"Erro adicional: {str(e2)}")
                raise Exception(f"Não foi possível carregar o modelo: {str(e)}")
            

def build_model_masked(input_shape, units=64, dropout_rate=0.2, lr=0.001):
    """Modelo com mascaramento de teclas indisponíveis"""
    import tensorflow as tf  # Import inside the function
    
    input_layer = Input(shape=input_shape)
    
    # Adicionar camada de Masking para lidar com padding
    masked_input = Masking(mask_value=0.0)(input_layer)
    
    # Processamento LSTM
    lstm_output = LSTM(units)(masked_input)
    lstm_output = Dropout(dropout_rate)(lstm_output)
    
    last_timestep = tf.keras.layers.Lambda(
        lambda x: x[:, -1, 3:10],  
        output_shape=(7,)  
    )(input_layer)
    
    # Saídas para cada posição de tecla
    outputs = []
    for i in range(4):
        logits = Dense(7, name=f'logits_{i+1}')(lstm_output)
        
        def create_masking_lambda(tensorflow):
            return tf.keras.layers.Lambda(
                lambda inputs: inputs[0] + tensorflow.math.log(tensorflow.clip_by_value(inputs[1], 1e-9, 1.0)) * 1e9,
                output_shape=(7,)
            )
        
        masked_logits = create_masking_lambda(tf)([logits, last_timestep])
        
   
        outputs.append(Activation('softmax', name=f'tecla{i+1}')(masked_logits))
    
    model = Model(inputs=input_layer, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'] * 4
    )
    return model


class MaskedMultiOutputKerasWrapper(BaseEstimator, ClassifierMixin):
    """Wrapper para modelo com mascaramento"""
    def __init__(self, input_shape, epochs=50, batch_size=16, verbose=0, units=64, dropout_rate=0.2, lr=0.001):
        self.input_shape = input_shape
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.units = units
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.model_ = None

    def fit(self, X, y):
        self.model_ = build_model_masked(
            self.input_shape, 
            units=self.units, 
            dropout_rate=self.dropout_rate, 
            lr=self.lr
        )
        
        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-5)
        ]
        
        self.model_.fit(
            X, [y[:, i] for i in range(4)],
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            callbacks=callbacks
        )
        return self

    def predict(self, X):
        y_pred = self.model_.predict(X, verbose=0)
        return np.stack([np.argmax(p, axis=1) for p in y_pred], axis=1)

    def score(self, X, y):
        y_pred = self.predict(X)
        accs = [accuracy_score(y[:, i], y_pred[:, i]) for i in range(4)]
        return np.mean(accs)

# Função de avaliação com métricas customizadas
def evaluate_model_with_masks(model, X_test, y_test):
    import tensorflow as tf
    """Avalia modelo com métricas de disponibilidade"""

    y_preds = model.predict(X_test)
    

    availability_masks = X_test[:, X_test.shape[1]-1, 3:10]
    
    # Calcular métricas para cada posição
    results = {}
    for i in range(4):
        standard_acc = accuracy_score(y_test[:, i], y_preds[:, i])
   
        correct = 0
        total = 0
        for j in range(len(y_test)):
            if y_preds[j, i] == y_test[j, i]:
                correct += 1
            total += 1
        masked_acc = correct / total if total > 0 else 0
        
  
        avg_options = np.mean(np.sum(availability_masks, axis=1))
        

        random_baseline = np.mean([1.0/np.sum(mask) if np.sum(mask) > 0 else 0 
                                  for mask in availability_masks])
        
        results[f'tecla_{i+1}'] = {
            'acuracia_padrao': standard_acc,
            'acuracia_ajustada': masked_acc,
            'media_opcoes_disponiveis': avg_options,
            'baseline_aleatorio': random_baseline,
            'melhoria_sobre_aleatorio': masked_acc / random_baseline if random_baseline > 0 else 0
        }
    
    # Médias gerais
    results['media'] = {
        'acuracia_padrao': np.mean([results[f'tecla_{i+1}']['acuracia_padrao'] for i in range(4)]),
        'acuracia_ajustada': np.mean([results[f'tecla_{i+1}']['acuracia_ajustada'] for i in range(4)]),
        'melhoria_sobre_aleatorio': np.mean([results[f'tecla_{i+1}']['melhoria_sobre_aleatorio'] for i in range(4)])
    }
    
    return results

def converter_para_serializavel(obj):
    if isinstance(obj, dict):
        return {k: converter_para_serializavel(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [converter_para_serializavel(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()  
    else:
        return obj


def treinar_modelo_sequencias_completas(modelo, colunas_teclas, batch_size, max_seq_len=50):
    modelo_id = f"seq_completas_batch{batch_size}"
    print(f"\n\n===== TREINANDO MODELO: SEQUÊNCIAS COMPLETAS, MAX_LEN {max_seq_len}, BATCH SIZE {batch_size} =====")
    
    # Gerar os dados
    X, y = modelo.gera_sequencias_completas(colunas_teclas, max_seq_len=max_seq_len)
    print(f"Shape dos dados: X={X.shape}, y={y.shape}")
    
    # Split treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Parâmetros fixos
    input_shape = (X.shape[1], X.shape[2])
    units = 64
    dropout_rate = 0.2
    lr = 0.001
    
    # Criar modelo
    clf = MaskedMultiOutputKerasWrapper(
        input_shape=input_shape,
        epochs=300,
        batch_size=batch_size,
        verbose=1,
        units=units,
        dropout_rate=dropout_rate,
        lr=lr
    )
    
    # Treinar
    try:
        clf.fit(X_train, y_train)
        
     
        results = evaluate_model_with_masks(clf, X_test, y_test)
        
        print(f"\nResultados para sequências completas, batch_size={batch_size}:")
        print(f"Acurácia média: {results['media']['acuracia_padrao']:.4f}")
        print(f"Melhoria sobre aleatório: {results['media']['melhoria_sobre_aleatorio']:.4f}x")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"modelos/{modelo_id}_{timestamp}.keras"
        
        if hasattr(clf, 'model_'):
            clf.model_.save(filename)
            print(f"Modelo salvo como: {filename}")
            results_serializaveis = converter_para_serializavel({
                "modelo_id": modelo_id,
                "max_seq_len": max_seq_len,
                "batch_size": batch_size,
                "resultados": results
            })

            results_filename = f"resultados/{modelo_id}_{timestamp}_results.json"
            with open(results_filename, 'w') as f:
                json.dump(results_serializaveis, f, indent=4)
           
            print(f"Resultados salvos como: {results_filename}")
        
        return {
            "modelo_id": modelo_id,
            "max_seq_len": max_seq_len,
            "batch_size": batch_size,
            "acuracia": results['media']['acuracia_padrao'],
            "melhoria_aleatorio": results['media']['melhoria_sobre_aleatorio']
        }
        
    except Exception as e:
        print(f"Erro ao treinar {modelo_id}: {str(e)}")
        return {
            "modelo_id": modelo_id,
            "max_seq_len": max_seq_len,
            "batch_size": batch_size,
            "acuracia": None,
            "melhoria_aleatorio": None,
            "erro": str(e)
        }


# Função para treinar modelo com contexto específico e batch size específico
def treinar_modelo_contexto_batch(modelo, colunas_teclas, contexto, batch_size):
    modelo_id = f"contexto{contexto}_batch{batch_size}"
    print(f"\n\n===== TREINANDO MODELO: CONTEXTO {contexto}, BATCH SIZE {batch_size} =====")
    
    X, y = modelo.gera_sequencias_com_contexto(colunas_teclas, contexto=contexto)
    print(f"Shape dos dados: X={X.shape}, y={y.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    input_shape = (X.shape[1], X.shape[2])
    units = 64
    dropout_rate = 0.2
    lr = 0.001
    
    # Criar modelo
    clf = MaskedMultiOutputKerasWrapper(
        input_shape=input_shape,
        epochs=300,
        batch_size=batch_size,
        verbose=1,
        units=units,
        dropout_rate=dropout_rate,
        lr=lr
    )
    
    try:
        clf.fit(X_train, y_train)
        
    
        results = evaluate_model_with_masks(clf, X_test, y_test)

        print(f"\nResultados para contexto={contexto}, batch_size={batch_size}:")
        print(f"Acurácia média: {results['media']['acuracia_padrao']:.4f}")
        print(f"Melhoria sobre aleatório: {results['media']['melhoria_sobre_aleatorio']:.4f}x")
        

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"modelos/{modelo_id}_{timestamp}.keras"
        
        if hasattr(clf, 'model_'):
            clf.model_.save(filename)
            print(f"Modelo salvo como: {filename}")
            results_serializaveis = converter_para_serializavel({
            "modelo_id": modelo_id,
            "contexto": contexto,
            "batch_size": batch_size,
            "resultados": results
            })

            results_filename = f"resultados/{modelo_id}_{timestamp}_results.json"
            with open(results_filename, 'w') as f:
                json.dump(results_serializaveis, f, indent=4)
           
            print(f"Resultados salvos como: {results_filename}")
        
        return {
            "modelo_id": modelo_id,
            "contexto": contexto,
            "batch_size": batch_size,
            "acuracia": results['media']['acuracia_padrao'],
            "melhoria_aleatorio": results['media']['melhoria_sobre_aleatorio']
        }
        
    except Exception as e:
        print(f"Erro ao treinar {modelo_id}: {str(e)}")
        return {
            "modelo_id": modelo_id,
            "contexto": contexto,
            "batch_size": batch_size,
            "acuracia": None,
            "melhoria_aleatorio": None,
            "erro": str(e)
        }

# Função para plotar resultados dos experimentos
def plotar_resultados(resultados):

    df_resultados = pd.DataFrame(resultados)

    df_resultados = df_resultados.dropna(subset=['acuracia'])
    
    tem_seq_completa = any('seq_completas' in r.get('modelo_id', '') for r in resultados)
    
    if tem_seq_completa:
        df_contexto = df_resultados[df_resultados['modelo_id'].str.contains('contexto')].copy()
        df_seq_completa = df_resultados[df_resultados['modelo_id'].str.contains('seq_completas')].copy()
        
        fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    else:
        df_contexto = df_resultados.copy()
   
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    

    if not df_contexto.empty:
        pivot1 = df_contexto.pivot(index='contexto', columns='batch_size', values='acuracia')
        if not pivot1.empty:
            sns.heatmap(pivot1, annot=True, fmt=".4f", cmap="YlGnBu", ax=axes[0])
            axes[0].set_title('Acurácia por Contexto e Batch Size')
            axes[0].set_xlabel('Batch Size')
            axes[0].set_ylabel('Tamanho do Contexto')
        
        pivot2 = df_contexto.pivot(index='contexto', columns='batch_size', values='melhoria_aleatorio')
        if not pivot2.empty:
            sns.heatmap(pivot2, annot=True, fmt=".4f", cmap="YlGnBu", ax=axes[1])
            axes[1].set_title('Melhoria sobre Aleatório por Contexto e Batch Size')
            axes[1].set_xlabel('Batch Size')
            axes[1].set_ylabel('Tamanho do Contexto')
    
    if tem_seq_completa and not df_seq_completa.empty:
        sns.barplot(x='batch_size', y='acuracia', data=df_seq_completa, ax=axes[2])
        axes[2].set_title('Acurácia para Sequências Completas por Batch Size')
        axes[2].set_xlabel('Batch Size')
        axes[2].set_ylabel('Acurácia')
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"resultados/comparacao_modelos_{timestamp}.png")
    print(f"Gráfico salvo como: resultados/comparacao_modelos_{timestamp}.png")
    
    plt.close()


def carregar_modelo(caminho_modelo):
        """
        Carrega um modelo treinado reconstruindo a arquitetura com contexto=3
        e carregando apenas os pesos.
        """
        try:
            input_shape = (4, 18)
 
            modelo = build_model_masked(input_shape)
            
            modelo.load_weights(caminho_modelo)
            print(f"Pesos do modelo carregados com sucesso de: {caminho_modelo}")
            return modelo
            
        except Exception as e:
            print(f"Erro detalhado ao carregar o modelo: {str(e)}")
            
            try:
                modelo = build_model_masked(input_shape)
                modelo.load_weights(caminho_modelo, by_name=True)
                print(f"Pesos carregados com sucesso usando by_name=True")
                return modelo
            except Exception as e2:
                print(f"Erro adicional: {str(e2)}")
                raise Exception(    f"Não foi possível carregar o modelo: {str(e)}")
