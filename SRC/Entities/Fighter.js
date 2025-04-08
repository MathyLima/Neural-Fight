import { MovementHandler } from '../Movement/MovementHandler.js';
import { CollisionHandler } from '../Movement/CollisionHandler.js';

export class Fighter {
    constructor(config) {
        this.x = config.x;
        this.y = config.y; 
        this.width = config.width;
        this.height = config.height;
        this.sprite = config.sprite;
        this.health = config.health;
        this.speed = config.speed;
        this.staggerFrame = config.staggerFrame || 50;
        this.velocity = { x: 0, y: 0 };
        this.sprite_Frame = 0;
        this.isAttacking = false;
        this.isBlocking = false;
        this.isJumping = false;
        this.animationState = 'idle';
        this.sprite_map = config.sprite_map;
        // Instâncias para controle de movimento e colisões
        this.colissionHandler = new CollisionHandler(this);
        this.movement = new MovementHandler(this,config.speed,this.colissionHandler);
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

    render(context) {
        const spriteState = this.sprite_map[this.animationState];
        
        
        if (!spriteState) {
            console.warn(`Estado de animação "${this.animationState}" não existe no sprite_map.`);
            return;
        }
    
        console.log(`Renderizando estado "${this.animationState}" com imagem:`, spriteState.image);
    
        // Calcular o quadro a ser exibido com base no número de frames e o frame atual
        let position = Math.floor(this.sprite_Frame / this.staggerFrame) % spriteState.frames;
        console.log(`Posição do frame: ${position}`);
        console.log(`Quantidade de frames: ${spriteState.frames}`);
        
        // Calcular a posição do quadro na sprite sheet
        let frameX = spriteState.frameWidth * position;
        let frameY = 0; // Se houver mais de uma linha de animação, altere isso conforme necessário
    
        // Renderizar a sprite no local correto (não depende de movimentação)
        context.drawImage(
            spriteState.image,
            frameX,
            frameY,
            spriteState.frameWidth,
            spriteState.frameHeight,
            this.x,
            this.y,
            this.width,
            this.height
        );
    
        // Desenha um quadrado ao redor do lutador para fins de depuração
        context.strokeStyle = 'red'; // Cor do quadrado
        context.lineWidth = 2; // Espessura da linha
        context.strokeRect(this.x, this.y, this.width, this.height);
    
        // Incrementa o contador de frames da animação
        this.sprite_Frame++;
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
