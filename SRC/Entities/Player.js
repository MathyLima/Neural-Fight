import {Fighter} from './Fighter.js';
import {Input} from '../Core/Input.js';


export class Player extends Fighter {
    constructor(name,config) {
        super(config);
        this.name = name;
        this.input = config.input
        //this.input.onEvent('ArrowUp', () => this.moveUp());
       // this.input.onEvent('ArrowDown', () => this.moveDown());
        this.input.onEvent(['a'], (isPressed) => {
            if(!isPressed) this.animationState = 'idle';
            else this.moveLeft()}
        );
        this.input.onEvent(['d'], (isPressed) =>{
            if(!isPressed) this.animationState = 'idle';
            else this.moveRight()}
        );


        this.input.onEvent(['j'],(isPressed)=>{
            this.attack1();
        })

        this.input.onEvent(['k'],(isPressed)=>{
            this.attack2();
        })
        this.input.onEvent(['l'],(isPressed)=>{
            this.attack3();
        })

        
    }

    update(deslocamento) {
        this.deslocamento.x = deslocamento.x;
        this.deslocamento.y = deslocamento.y;
    }

    
}

