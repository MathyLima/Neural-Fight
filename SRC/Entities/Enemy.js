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


    async inputTime(fazerPredicao) {
        // Supondo que 'this.roundKeys' seja um array de teclas possíveis
        const inputList = this.roundKeys;
        
        // Inicia a sessão de entrada
        this.inputSession = true;
        
        // Inicializa pressedKeys como um array vazio
        this.pressedKeys = [];
        
        // Verifica se fazerPredicao é uma função
        if (typeof fazerPredicao === 'function') {
            try {
                // Chama a função assíncrona fazerPredicao para obter as teclas
                const predicao = await fazerPredicao();
                
                // Verifica se a predição retornou teclas válidas
                if (predicao && Array.isArray(predicao.keys) && predicao.keys.length > 0) {
                    this.pressedKeys = predicao.keys;
                } else {
                    // Se não houver predição válida, gera teclas aleatórias
                    for (let i = 0; i < 4; i++) {
                        const randomKey = inputList[Math.floor(Math.random() * inputList.length)];
                        this.pressedKeys.push(randomKey);
                    }
                }
            } catch (error) {
                // Em caso de erro na predição, usa teclas aleatórias
                console.error("Erro ao fazer predição:", error);
                for (let i = 0; i < 4; i++) {
                    const randomKey = inputList[Math.floor(Math.random() * inputList.length)];
                    this.pressedKeys.push(randomKey);
                }
            }
        } else {
            // Se fazerPredicao não for uma função, gera teclas aleatórias
            for (let i = 0; i < 4; i++) {
                const randomKey = inputList[Math.floor(Math.random() * inputList.length)];
                this.pressedKeys.push(randomKey);
            }
        }
        
        // Termina a sessão de entrada
        this.inputSession = false;
        
        console.log(this.pressedKeys); // Exibe as teclas escolhidas
    }
    
    
       
    }

    


