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

        const inputList = this.roundKeys
        this.inputSession = true
        this.pressedKeys = inputList;
        this.pressedKeys = ['a','a','a','a'];
        this.inputSession = false
        
    }
    
    
       
    }

    


