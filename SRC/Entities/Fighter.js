import { MovementHandler } from '../Movement/MovementHandler.js';
import { CollisionHandler } from '../Movement/CollisionHandler.js';
import { HealthBar } from '../UI/HealthBar.js';
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

        this.healthBar = new HealthBar({
            x: this.x,
            y: 10, // Posiciona a barra acima do lutador
            width: this.width,
            height: 30,
            maxHealth: this.maxHealth,
            currentHealth: this.health,
        });
        // Instâncias para controle de movimento e colisões
        // vou precisar colocar uma flag para saber se está em uma ação, seja correndo, atacando ou defendendo
        this.isActioning = false; // Flag para saber se o lutador está em uma ação (movendo, atacando, defendendo)
        this.colissionHandler = new CollisionHandler({xMin:0,xMax:config.map.width,yMin:0,yMax:config.map.height}, null); // null para o outro jogador, pois ainda não foi definido
        this.movement = new MovementHandler(this,config.speed,this.colissionHandler);
        this.enemy = null; // Referência ao inimigo, se necessário
    }

    setEnemy(enemy) {
        this.enemy = enemy;
        this.colissionHandler.otherPlayer = enemy; // Atualiza o manipulador de colisões com o inimigo
    }

    // Métodos para movimentação
    moveLeft() {
        this.animationState = 'walk_left';
        this.movement.moveLeft();
    }

    attack1(){
        this.animationState = 'attack1';


        this.isAttacking = true;
        
        //this.movement.attack1();
    }

    attack2(){
        this.animationState = 'attack2';
        this.isAttacking = true;
        
        
    }

    attack3(){
        this.animationState = 'attack3';
        this.isAttacking = true;
        
        
    }

    moveRight() {
        this.animationState = 'walk';
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
    
        // Calcular o quadro a ser exibido com base no número de frames e o frame atual
        let position = Math.floor(this.sprite_Frame / this.staggerFrame) % spriteState.frames;
        
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

        this.healthBar.render(context);

        
        // Desenha um quadrado ao redor do lutador para fins de depuração
        //context.strokeStyle = 'red'; // Cor do quadrado
        //context.lineWidth = 2; // Espessura da linha
        //context.strokeRect(this.x, this.y, this.width, this.height);
        
        // Incrementa o contador de frames da animação
        if(this.isAttacking) {
            if(position === this.sprite_map[this.animationState].frames - 1) {
                this.isAttacking = false;
                this.animationState = 'idle';
                this.sprite_Frame = 0; // Reinicia o contador de frames após o ataque
            }
        }
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
