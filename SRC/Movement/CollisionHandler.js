export class CollisionHandler {
    constructor(mapBounds, otherPlayer) {
        this.mapBounds = mapBounds; // Limites do mapa (ex: {xMin, xMax, yMin, yMax})
        this.otherPlayer = otherPlayer; // Referência ao outro jogador (para colisão jogador x jogador)
    }

    // Verifica se o jogador colide com os limites do mapa
    isCollidingWithMap(player) {
        const { x, y, width, height } = player;

        // Verifica se o jogador está dentro dos limites horizontais e verticais do mapa
        if (x < this.mapBounds.xMin || x + width > this.mapBounds.xMax) {
            return true; // Colidiu com o limite X
        }
        if (y < this.mapBounds.yMin || y + height > this.mapBounds.yMax) {
            return true; // Colidiu com o limite Y
        }

        return false;
    }

    // Verifica se o jogador colide com o outro jogador
    isCollidingWithPlayer(player) {
        const { x: playerX, y: playerY, width: playerWidth, height: playerHeight } = player;
        const { x: otherPlayerX, y: otherPlayerY, width: otherPlayerWidth, height: otherPlayerHeight } = this.otherPlayer;

        // Verifica se há sobreposição entre os jogadores
        return !(playerX + playerWidth < otherPlayerX || 
                 playerX > otherPlayerX + otherPlayerWidth || 
                 playerY + playerHeight < otherPlayerY || 
                 playerY > otherPlayerY + otherPlayerHeight);
    }
}
