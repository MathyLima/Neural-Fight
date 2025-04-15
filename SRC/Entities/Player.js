import { Fighter } from "./Fighter.js";

export class Player extends Fighter {
    constructor(name, config) {
        super(config);
        this.name = name;
        this.input = config.input;
        this.lastKey = null;
        this.isMoving = false; // Nova flag para controlar a movimentação
        this.keysPressed = {};
        this.inputTimeout = 1000; // Tempo máximo para cada input (1 segundo)
        this.lastInputTime = {}; // Armazena o último tempo de input para cada tecla
        this.inputDelay = 500; // Delay entre as entradas (500ms)
        this.lastInput = 0; // Marca o tempo do último input
    };

    // Método para armazenar as teclas pressionadas
    storePressedKey(key) {

        this.pressedKeys.push(key); // Adiciona a tecla ao array de teclas pressionadas
        console.log(this.pressedKeys,this.numberInputs)
    }

    inputTime() {
        let inputCount = 0;
    
        this.input.onEvent(['j', 'k', 'l', 'i', 'o', 'p'], (key, isPressed) => {
            const currentTime = Date.now();
    
            // Protege contra múltiplos triggers rápidos
            if (currentTime - this.lastInput < this.inputDelay) return;
    
            this.lastInput = currentTime;
    
            const isAtaque = this.turno === 'ataque';
            const isDefesa = this.turno === 'defesa';
    
            if (isPressed) {
    
    
                if (inputCount >= this.numberInputs) {
                    this.inputing = false;
                    inputCount = 0
                    return;
                }
    
                    this.lastInputTime[key] = currentTime;
                    this.keysPressed[key] = true;
    
                    if (isAtaque && ['j', 'k', 'l'].includes(key) && !this.isAttacking) {
                        if (this.isBlocking) this.stopBlock();
                        this.storePressedKey(key);
                        inputCount++;
                    }
    
                    if (isDefesa && ['i', 'o', 'p'].includes(key) && !this.isBlocking) {
                        this.storePressedKey(key);
                        inputCount++;
                    }
                
            } else {
                this.keysPressed[key] = false;
            }
        });
    }
}
