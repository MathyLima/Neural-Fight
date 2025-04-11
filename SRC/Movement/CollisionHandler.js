export class CollisionHandler {
    constructor(mapBounds, otherPlayer) {
        this.mapBounds = mapBounds; // Limites do mapa (ex: {xMin, xMax, yMin, yMax})
        this.otherPlayer = otherPlayer; // Referência ao outro jogador (para colisão jogador x jogador)
    }

    // Verifica se o jogador colide com os limites do mapa
    isCollidingWithMap(player, deslocamento) {
        const { x, y, width, height } = player;
        if (x + deslocamento < this.mapBounds.xMin || x + width + deslocamento > this.mapBounds.xMax) {
            return true; // Colidiu com o limite X
        }
        

        return false;
    }

    // Verifica colisão apenas na segunda metade do otherPlayer
    checkCollision(playerX, playerY, playerWidth, playerHeight,
        otherPlayerX, otherPlayerY, otherPlayerWidth, otherPlayerHeight) {

        const secondHalfX = otherPlayerX + (otherPlayerWidth / 2); // começo da segunda metade

        return (
            playerX < otherPlayerX + otherPlayerWidth &&    // ainda dentro do limite direito
            playerX + playerWidth > secondHalfX &&          // entrou na segunda metade
            playerY < otherPlayerY + otherPlayerHeight &&
            playerY + playerHeight > otherPlayerY
        );
    }


    checkCollisionAttack(entity, target) {
        // Calcula a metade do alvo (target)
        const targetMidX = target.x + target.width / 2;
    
        // Ajusta as posições da entidade adicionando ou subtraindo 5
        const entityLeft = entity.x - 5; // Lado esquerdo da entidade com deslocamento para a esquerda
        const entityRight = entity.x + entity.width + 5; // Lado direito da entidade com deslocamento para a direita
    
        // Verifica se a entidade está sobrepondo a metade do alvo
        return (
            entityRight >= targetMidX && // O lado direito da entidade está à direita da metade do alvo
            entityLeft <= targetMidX // O lado esquerdo da entidade está à esquerda da metade do alvo
        );
    }
    isCollidingWithPlayer(player, deslocamento) {
        const { x: playerX, y: playerY, width: playerWidth, height: playerHeight } = player;
        const { x: otherPlayerX, y: otherPlayerY, width: otherPlayerWidth, height: otherPlayerHeight } = this.otherPlayer;
        
        return this.checkCollision(
            playerX + deslocamento, playerY, playerWidth, playerHeight,
            otherPlayerX, otherPlayerY, otherPlayerWidth, otherPlayerHeight
        );
    }
}
