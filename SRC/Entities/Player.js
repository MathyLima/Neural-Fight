import { Fighter } from "./Fighter.js";
export class Player extends Fighter {
    constructor(name, config) {
        super(config);
        this.name = name;
        this.input = config.input;
        this.lastKey = null;
        this.isMoving = false; // Nova flag para controlar a movimentação


        this.keysPressed = {};


       

        this.input.onEvent(['a', 'd', 'w','j','k','l','i','o','p'], (key,isPressed) => {
            
            if(isPressed){
                this.keysPressed[key] = true;
                if(key === 'a' || key === 'd'){
                    this.lastKey = key;
                }
                
                // Horizontal
                if (this.keysPressed['a'] && this.lastKey === 'a') {
                    console.log('estou poressionando a')
                    this.moveLeft();
                } else if (this.keysPressed['d'] && this.lastKey === 'd') {
                    this.moveRight();
                } 
            
                // Vertical
                if (this.keysPressed['w']) {
                    this.moveUp();
                    console.log(this.keysPressed)
                } 

                if( this.keysPressed['j'] && !this.isAttacking){
                    if(this.isBlocking) this.stopBlock();
                    this.attack1();
                }

                if( this.keysPressed['k'] && !this.isAttacking){
                    if(this.isBlocking) this.stopBlock();
                    this.attack2();
                }
                if( this.keysPressed['l'] && !this.isAttacking){
                    if(this.isBlocking) this.stopBlock();
                    this.attack3();
                }

                if( this.keysPressed['i'] && !this.isBlocking){
                    this.block(1);
                }
                if( this.keysPressed['o'] && !this.isBlocking){
                    this.block(2);
                }
                if( this.keysPressed['p'] && !this.isBlocking){
                    this.block(3);
                }

            }else{
                this.keysPressed[key] = false;
                if (!this.keysPressed['a'] && !this.keysPressed['d']) {
                    this.movement.stopHorizontal();
                }
                else if (!this.keysPressed['w']) {
                    this.movement.stopVertical();
                }

            }
        });

       
       

        // Bloqueio 1
        this.input.onEvent(['i'], (isPressed) => {
            if (isPressed && !this.isActioning) {
                this.isActioning = true;
                this.block(1);
            } else if (!isPressed) {
                this.isActioning = false;
                this.stopBlock();
            }
        });

        // Bloqueio 2
        this.input.onEvent(['o'], (isPressed) => {
            if (isPressed && !this.isActioning) {
                this.isActioning = true;
                this.block(2);
            } else if (!isPressed) {
                this.isActioning = false;
                this.stopBlock();
            }
        });

        // Bloqueio 3
        this.input.onEvent(['p'], (isPressed) => {
            if (isPressed && !this.isActioning) {
                this.isActioning = true;
                this.block(3);
            } else if (!isPressed) {
                this.isActioning = false;
                this.stopBlock();
            }
        });
    }
}