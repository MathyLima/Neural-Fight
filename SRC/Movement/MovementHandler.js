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
        console.log(this.entity.speed.x); // Para depuração
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
    
}
