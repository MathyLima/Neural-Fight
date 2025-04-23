export class ServerCommunicator {
  constructor(config) {
    this.config = config;
    this.socket = null;
    this.porta = config.porta;
    this.isConnected = false;
    this.playerId = null;
    this.input = config.input;
    this.isPredicting = false;

    try {
      this.connect();
    } catch (error) {
      console.error("Erro ao conectar ao servidor:", error);
    }
  }

  connect() {
    this.socket = new WebSocket(`ws://${this.config.host}:${this.porta}`);
    this.socket.addEventListener("open", () => {
      console.log("Conectado ao servidor!");
      this.isConnected = true;
    });

    this.socket.addEventListener("message", (event) => {
      console.log("Recebido:", event.data);
      const data = JSON.parse(event.data);

      if (data.type === "playerId") {
        this.playerId = data.playerId;
        console.log(`ID do jogador: ${this.playerId}`);
      } else if (data.type === "gameState") {
        console.log(data.state);
      } else if (data.type === "predicao") {
        console.log("Teclas previstas:", data.teclas_previstas);

        if (this.pendingPredictionResolve) {
          this.pendingPredictionResolve(data.teclas_previstas);
          this.pendingPredictionResolve = null;
          this.isPredicting = false;
        }
      } else if (data.type === "erro") {
        console.error("Erro do servidor:", data.message);

        if (this.pendingPredictionResolve) {
          this.pendingPredictionReject(data.message);
          this.pendingPredictionResolve = null;
          this.pendingPredictionReject = null;
          this.isPredicting = false;
        }
      }
    });

    this.socket.addEventListener("close", () => {
      console.log("Conexão fechada!");
      this.isConnected = false;
    });

    this.socket.addEventListener("error", (error) => {
      console.error("Erro na conexão WebSocket:", error);
      if (this.pendingPredictionResolve && this.pendingPredictionReject) {
        this.pendingPredictionReject("Erro na conexão WebSocket");
        this.pendingPredictionResolve = null;
        this.pendingPredictionReject = null;
        this.isPredicting = false;
      }
    });
  }

  enviarMensagem(mensagem) {
    if (this.isConnected) {
      this.socket.send(JSON.stringify(mensagem));
    } else {
      console.log("Não conectado ao servidor!");
    }
  }

  enviarEstadoDoJogo(estadoDoJogo) {
    console.log(estadoDoJogo.turnoAtual);
    if (this.isConnected) {
      const mensagem = {
        id: estadoDoJogo.id,
        state: estadoDoJogo,
        caminho: "salvar",
      };
      this.socket.send(JSON.stringify(mensagem));
    } else {
      console.log("Não conectado ao servidor!");
    }
  }

  carregarModelo(modelo) {
    if (this.isConnected) {
      const mensagem = {
        modelo: modelo,
        caminho: "carregar",
      };
      this.socket.send(JSON.stringify(mensagem));
    } else {
      console.log("Não conectado ao servidor!");
    }
  }

  predicao(contexto, estadoJogo) {
    const modelos = {
      300: "contexto3_batch32_20250421_221318",
      1000: "contexto3_batch32_20250421_221318",
      10000: "contexto3_batch32_20250421_221318",
    };

    if (this.isPredicting) {
      return Promise.reject("Uma predição já está em andamento");
    }

    return new Promise((resolve, reject) => {
      if (this.isConnected) {
        this.isPredicting = true;
        this.pendingPredictionResolve = resolve;
        this.pendingPredictionReject = reject;

        const formattedEstadoJogo = this.formatGameState(estadoJogo);

        const mensagem = {
          caminho: "inferir",
          state: formattedEstadoJogo,
          modelo: modelos[contexto] || modelos[1000],
          id: estadoJogo.id,
        };

        try {
          this.socket.send(JSON.stringify(mensagem));
        } catch (error) {
          this.isPredicting = false;
          this.pendingPredictionResolve = null;
          this.pendingPredictionReject = null;
          reject(`Erro ao enviar mensagem: ${error.message}`);
        }

        setTimeout(() => {
          if (this.isPredicting) {
            this.isPredicting = false;
            if (this.pendingPredictionReject) {
              this.pendingPredictionReject("Timeout na predição");
              this.pendingPredictionResolve = null;
              this.pendingPredictionReject = null;
            }
          }
        }, 10000);
      } else {
        reject("Não conectado ao servidor!");
      }
    });
  }

  formatGameState(estadoJogo) {
    const formattedState = JSON.parse(JSON.stringify(estadoJogo));

    if (
      formattedState.teclasPressionadasJogador1 &&
      !Array.isArray(formattedState.teclasPressionadasJogador1)
    ) {
      formattedState.teclasPressionadasJogador1 = [
        formattedState.teclasPressionadasJogador1,
      ];
    }

    if (!formattedState.turnoAtual) formattedState.turnoAtual = 1;
    if (!formattedState.id) formattedState.id = this.playerId || "player";

    return formattedState;
  }
}
