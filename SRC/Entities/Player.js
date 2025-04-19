import { Fighter } from "./Fighter.js";

export class Player extends Fighter {
    constructor(id, config) {
        super(config);
        console.log(id)
        this.id = id;
        this.input = config.input;
        this.lastKey = null;
        this.isMoving = false; // Nova flag para controlar a movimentação
        this.keysPressed = {};
        this.inputTimeout = 500; // Tempo máximo para cada input (1 segundo)
        this.lastInputTime = {}; // Armazena o último tempo de input para cada tecla
        this.inputDelay = 200; // Delay entre as entradas (500ms)
        this.lastInput = 0; // Marca o tempo do último input
    };

    storePressedKey(key,inputList) {
        // Impede duplicatas, exceto se a tecla for 'a'
        this.pressedKeys.push(key);
        const actualKey = this.pressedKeys.length - 1;
    


       

    
        Array.from(document.querySelectorAll('.inputSpace')).forEach((input, index) => {
            if (index === actualKey) {
                input.querySelector('.inputRect').style.backgroundColor = 'white';
                input.querySelector('.inputRect').querySelector('h1').innerText = key;
            }
        });
    }
    
    inputTime() {
        const inputList = this.roundKeys;
        //if(!this.inputing || !this.isCentered)return;
        this.inputSession = true
        // Remove listeners anteriores
        if (this._inputCallback && this._inputKeys) {
            this.input.removeEvent(this._inputKeys, this._inputCallback);
        }
    
        const pressedKeysSet = new Set(); // Armazena teclas já pressionadas
    
        const callback = (key, isPressed) => {
            const currentTime = Date.now();
    
            if (currentTime - this.lastInput < this.inputDelay) return;
            this.lastInput = currentTime;
    
            if (isPressed) {
                
    
                // Armazena o tempo e marca a tecla como pressionada
                this.lastInputTime[key] = currentTime;
                this.keysPressed[key] = true;
    
                // Tenta armazenar a tecla (pode ser rejeitada internamente)
                const beforeCount = this.pressedKeys.length;
                this.storePressedKey(key, inputList);
                const afterCount = this.pressedKeys.length;
    
                // Se a tecla foi realmente adicionada, siga com o fluxo
                if (afterCount > beforeCount) {
                    pressedKeysSet.add(key);
    
                    
                    console.log(key)
                    // Se atingiu o número máximo de inputs, remove todos os listeners
                    if (afterCount >= this.numberInputs) {
                        inputList.forEach(k => {
                            this.input.removeEvent(k, callback);
                        });

                        this.inputing = false;
                        
                        return;
                    }
                }
            } else {
                this.keysPressed[key] = false;
            }
        };
    
        // Armazena callback e keys atuais para possível limpeza posterior
        this._inputCallback = callback;
        this._inputKeys = inputList;
    
        this.input.onEvent(inputList, callback);
    
        // Listener extra de segurança que remove tudo em caso de clique externo
        this._cancelClickListener = () => {
            this.input.removeEvent(inputList, callback);
            document.removeEventListener("click", this._cancelClickListener);
        };
        document.addEventListener("click", this._cancelClickListener);
    }
}
