import { MovementHandler } from '../Movement/MovementHandler.js';
import { CollisionHandler } from '../Movement/CollisionHandler.js';

export class Fighter {
    constructor(x, y, width, height, sprite, health = 100, speed = 5, map) {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
        this.sprite = sprite;
        this.health = health;
        this.speed = speed;
        this.velocity = { x: 0, y: 0 };
        this.isAttacking = false;
        this.isBlocking = false;
        this.isJumping = false;
        this.animationState = 'idle';

        // Instâncias para controle de movimento e colisões
        this.colissionHandler = new CollisionHandler(this,map);
        this.movement = new MovementHandler(this,speed,this.colissionHandler);
    }

    // Métodos para movimentação
    moveLeft() {
        this.movement.moveLeft();
    }

    moveRight() {
        this.movement.moveRight();
    }

    stop() {
        this.movement.stop();
    }

    jump() {
        this.movement.jump();
    }

    // Métodos de ataque e defesa
    attack() {
        if (!this.isAttacking) {
            this.isAttacking = true;
            this.animationState = 'attacking';
        }
    }

    block() {
        this.isBlocking = true;
        this.animationState = 'blocking';
    }

    stopBlock() {
        this.isBlocking = false;
        this.animationState = 'idle';
    }

    // Atualizar o estado do lutador
    update() {
        if (this.constructor === Fighter) {
            throw new Error('Método "update" deve ser implementado nas subclasses!');
        }
    }

    // Renderizar o lutador (exemplo com contexto de canvas)
    render(context) {
        // Lógica para desenhar o lutador (pode ser ajustada conforme a animação)
        context.drawImage(this.sprite, this.x, this.y, this.width, this.height);
    }

    // Receber dano
    takeDamage(amount) {
        this.health -= amount;
        if (this.health <= 0) {
            this.health = 0;
            this.die();
        }
    }

    // Morrer
    die() {
        if (this.constructor === Fighter) {
            throw new Error('Método "die" deve ser implementado nas subclasses!');
        }
    }
}
