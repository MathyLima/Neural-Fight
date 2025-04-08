import {Fighter} from './Fighter.js';
import {Input} from '../Core/Input.js';
import { MovementHandler } from '../Movement/MovementHandler.js';
import { CollisionHandler } from '../Movement/CollisionHandler.js';

export class Enemy extends Fighter {
    constructor(name,config) {
        super(config);
        this.name = name;
        this.input = new Input();
        //this.input.onEvent('ArrowUp', () => this.moveUp());
       // this.input.onEvent('ArrowDown', () => this.moveDown());
        this.input.onEvent(['a'], (isPressed) => {
           // if(!isPressed) this.animationState = 'idle';
            //else this.moveLeft()
            }
        );
        this.input.onEvent(['d'], (isPressed) =>{
            //if(!isPressed) this.animationState = 'idle';
            //else this.moveRight()
            }
        );

        this.colissionHandler = new CollisionHandler(this);
        this.movement = new MovementHandler(this,config.speed,this.colissionHandler);
    }

    update(deslocamento) {
        this.deslocamento.x = deslocamento.x;
        this.deslocamento.y = deslocamento.y;
    }

    
}

