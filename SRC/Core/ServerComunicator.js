export class ServerCommunicator{
    constructor(config){
        this.config = config;
        this.socket = null;
        this.porta = config.porta
        this.isConnected = false;
        this.playerId = null;
        this.input = config.input;

        this.input.onEvent(['a'], (isPressed) => {
            this.enviarMensagem('a');
            }
        );
        this.input.onEvent(['d'], (isPressed) =>{
            this.enviarMensagem('d');
              
            }
        );

        this.input.onEvent(['j'],(isPressed)=>{
            this.enviarMensagem('ataque1');
        })
        this.input.onEvent(['k'],(isPressed)=>{
            this.enviarMensagem('ataque2');
        })
        this.input.onEvent(['l'],(isPressed)=>{
            this.enviarMensagem('ataque3');
        })

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
}