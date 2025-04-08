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
        if (y < this.mapBounds.yMin || y + height > this.mapBounds.yMax) {
            return true; // Colidiu com o limite Y
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

    // Verifica se o jogador colide com o outro jogador
    isCollidingWithPlayer(player, deslocamento) {
        const { x: playerX, y: playerY, width: playerWidth, height: playerHeight } = player;
        const { x: otherPlayerX, y: otherPlayerY, width: otherPlayerWidth, height: otherPlayerHeight } = this.otherPlayer;

        return this.checkCollision(
            playerX + deslocamento, playerY, playerWidth, playerHeight,
            otherPlayerX, otherPlayerY, otherPlayerWidth, otherPlayerHeight
        );
    }
}
