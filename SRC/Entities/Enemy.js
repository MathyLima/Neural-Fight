import {Fighter} from './Fighter.js';
import {Input} from '../Core/Input.js';


export class Enemy extends Fighter {
    constructor(name,config) {
        super(config);
        this.name = name;
        this.input = new Input();

        this.lastAttackTime = 0;
        this.attackCooldown = 1000; // em milissegundos (1 segundo, por exemplo)
    }

    updateAI() {
        throw new Error('implemente')
    }


    inputTime(){

                // Supondo que 'this.roundKeys' seja um array de teclas possíveis.
        const inputList = this.roundKeys;

        // Inicia a sessão de entrada
        this.inputSession = true;

        // Escolhe 4 teclas aleatórias (com repetição) a partir de inputList
        this.pressedKeys = [];
        for (let i = 0; i < 4; i++) {
            const randomKey = inputList[Math.floor(Math.random() * inputList.length)];
            this.pressedKeys.push(randomKey);
            
        }
        
        // Termina a sessão de entrada
        this.inputSession = false;

        console.log(this.pressedKeys); // Exibe as teclas escolhidas
                
    }
    
    
       
    }

    


