import {Fighter} from './Fighter.js';
import {Input} from '../Core/Input.js';

export class Player extends Fighter {
    constructor(name,config) {
        super(config);
        this.name = name;
        this.input = new Input();
        //this.input.onEvent('ArrowUp', () => this.moveUp());
       // this.input.onEvent('ArrowDown', () => this.moveDown());
        this.input.onEvent(['a'], () => this.moveLeft());
        this.input.onEvent(['d'], () => this.moveRight());
    }

    update(deslocamento) {
        this.deslocamento.x = deslocamento.x;
        this.deslocamento.y = deslocamento.y;
    }

    
}

