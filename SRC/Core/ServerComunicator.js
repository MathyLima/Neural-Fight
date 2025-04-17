export class ServerCommunicator{
    constructor(config){
        this.config = config;
        this.socket = null;
        this.porta = config.porta
        this.isConnected = false;
        this.playerId = null;
        this.input = config.input;

       

        try{
            this.connect();
        }catch(error){ 
            console.error('Erro ao conectar ao servidor:', error);
        }
        
    }

    connect(){
        this.socket = new WebSocket(`ws://${this.config.host}:${this.porta}`);
        this.socket.addEventListener('open', () => {
            console.log('Conectado ao servidor!');
            this.isConnected = true;
        });
        
        this.socket.addEventListener('message', (event) => {
            console.log('Recebido:', event.data);
            const data = JSON.parse(event.data);
            if(data.type === 'playerId'){
                this.playerId = data.playerId;
                console.log(`ID do jogador: ${this.playerId}`);
            }else if(data.type === 'gameState'){
                // Atualizar o estado do jogo com os dados recebidos
                console.log(data.state);
            }
        });
        this.socket.addEventListener('close', () => {
            console.log('Conexão fechada!');
            this.isConnected = false;
        });
    }

    enviarMensagem(mensagem){
        if(this.isConnected){
            this.socket.send(JSON.stringify(mensagem));
        }else{
            console.log('Não conectado ao servidor!');
        }
    }


    enviarEstadoDoJogo(estadoDoJogo) {
        console.log(estadoDoJogo)
        if (this.isConnected) {
            const mensagem = {
                type: 'gameState',
                id: estadoDoJogo.id,
                state: estadoDoJogo
            };
            this.socket.send(JSON.stringify(mensagem));
        } else {
            console.log('Não conectado ao servidor!');
        }
    }
}