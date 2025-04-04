export class MovementHandler {
    constructor(entity, speed = 5, collisionHandler) {
        this.entity = entity; // Referência ao jogador
        this.speed = speed;
        this.velocity = { x: 0, y: 0 };
        this.collisionHandler = collisionHandler; // Verificação de colisões
    }

    // Movimento para a esquerda
    moveLeft() {
        if (!this.collisionHandler.isCollidingWithMap(this.entity) && !this.collisionHandler.isCollidingWithPlayer(this.entity)) {
            this.velocity.x = -this.speed;
        } else {
            this.velocity.x = 0;
        }
    }

    // Movimento para a direita
    moveRight() {
        if (!this.collisionHandler.isCollidingWithMap(this.entity) && !this.collisionHandler.isCollidingWithPlayer(this.entity)) {
            this.velocity.x = this.speed;
        } else {
            this.velocity.x = 0;
        }
    }

    // Movimento para cima
    moveUp() {
        if (!this.collisionHandler.isCollidingWithMap(this.entity) && !this.collisionHandler.isCollidingWithPlayer(this.entity)) {
            this.velocity.y = -this.speed;
        } else {
            this.velocity.y = 0;
        }
    }

    // Movimento para baixo
    moveDown() {
        if (!this.collisionHandler.isCollidingWithMap(this.entity) && !this.collisionHandler.isCollidingWithPlayer(this.entity)) {
            this.velocity.y = this.speed;
        } else {
            this.velocity.y = 0;
        }
    }

    // Para movimento horizontal
    stopHorizontal() {
        this.velocity.x = 0;
    }

    // Para movimento vertical
    stopVertical() {
        this.velocity.y = 0;
    }

    // Atualiza a posição do jogador
    update() {
        this.entity.x += this.velocity.x;
        this.entity.y += this.velocity.y;
    }
}
