import { Game } from "../Core/Game.js";
export class MovementHandler {
    constructor(entity, collisionHandler) {
        this.entity = entity; // Referência ao jogador
        this.collisionHandler = collisionHandler;
        this.speed = {x:5,y:20}; // Velocidade de movimento
    }


    getspeedX() {
        return this.speed.x;
    }
    getspeedY() {
        return this.speed.y;
    }
    // Movimento para a esquerda
    moveLeft() {
        
        this.speed.x = -Math.abs(this.speed.x); // Inverte a velocidade para mover para a esquerda
    
        this.entity.speed.x = this.speed.x; // Define a velocidade negativa para mover para a esquerda
    }

    // Movimento para a direita
    moveRight() {
        this.speed.x = Math.abs(this.speed.x); // Mantém a velocidade positiva para mover para a direita
        this.entity.speed.x = this.speed.x; // Define a velocidade positiva para mover para a direita
       
    }

    // Movimento para cima
    moveUp() {
       
       this.speed.y = -Math.abs(this.speed.y); // Inverte a velocidade para mover para cima
       this.entity.speed.y = this.speed.y; // Define a velocidade negativa para mover para cima
    }

    // Movimento para baixo
    moveDown() {
        
    }

    // Para movimento horizontal
    stopHorizontal() {
        this.entity.speed.x = 0; // Para o movimento horizontal
        this.entity.ismovingRight = false; // Reseta a flag de movimento para a direita
        this.entity.ismovingLeft = false; // Reseta a flag de movimento para a esquerda
        this.entity.changeAnimationState('idle'); // Muda o estado de animação para "parado"
    }

    // Para movimento vertical
    stopVertical() {
        this.entity.speed.y = 0; // Para o movimento vertical
    }


    moveToCenter() {
        const centerX = this.collisionHandler.mapBounds.xMax / 2;
    
        const entity = this.entity;
    
        // Calcula a posição central da attackBox com base na posição atual do entity
        const attackBoxCenterX = entity.x + entity.attackBox.offset.x + (entity.attackBox.width / 2);
    
        const delta = centerX - attackBoxCenterX;
    
        if (Math.abs(delta) > Math.abs(this.entity.speed.x)) {
            if (delta > 0) {
                this.moveRight();
                this.entity.moveRight()
                entity.ismovingRight = true;
                entity.ismovingLeft = false;
            } else {
                this.moveLeft();
                this.entity.moveLeft();
                entity.ismovingLeft = true;
                entity.ismovingRight = false;
            }
        } else {
            this.entity.isCentered = true
            this.stopHorizontal(); // Centralizou, então para
            const targetX = centerX - this.entity.attackBox.offset.x - (this.entity.attackBox.width / 2);
            this.entity.x = targetX;
            const gameInstance = Game.getInstance();

            gameInstance.determinaInputsMostrados();
            gameInstance.abrirTelaInput();
            
        }
    }




    moveToInitialPosition() {
        const currentX = this.entity.x;
        const targetX = this.entity.initialX;
        const delta = targetX - currentX;
        this.entity.movingToInitialPosition = true
        if (Math.abs(delta) > Math.abs(this.entity.speed.x)) {
            if (delta > 0) {
                this.moveRight();
                this.entity.ismovingRight = true;
                this.entity.ismovingLeft = false;
            } else {
                this.moveLeft();
                this.entity.ismovingLeft = true;
                this.entity.ismovingRight = false;
            }
        } else {
            this.stopHorizontal(); // Chegou na posição
            this.entity.x = targetX; // Corrige qualquer pequeno erro de arredondamento
            this.entity.movingToInitialPosition = false;
            const gameInstance = Game.getInstance();
            this.entity.takeDamageFinal();
            
            const proximaRodadaTela = document.getElementById('proximaRodada')
            if(!gameInstance.gameEnded){
                proximaRodadaTela.style.display = 'flex';

            }

            proximaRodadaTela.querySelector('#proximaRodadaBotao').onclick = ()=>{

                proximaRodadaTela.style.display = 'none';
                this.entity.isCentered = false;
                this.entity.enemy.isCentered = false;
                gameInstance.addRound();
            }

        }
    }
    
    
}
