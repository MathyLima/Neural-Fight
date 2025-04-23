export class CollisionHandler {
  constructor(mapBounds, otherPlayer) {
    this.mapBounds = mapBounds;
    this.otherPlayer = otherPlayer;
  }

  isCollidingWithMap(player, deslocamento) {
    const { x, y, width, height } = player;
    if (
      x + deslocamento < this.mapBounds.xMin ||
      x + width + deslocamento > this.mapBounds.xMax
    ) {
      return true;
    }

    return false;
  }

  checkCollision(
    playerX,
    playerY,
    playerWidth,
    playerHeight,
    otherPlayerX,
    otherPlayerY,
    otherPlayerWidth,
    otherPlayerHeight
  ) {
    const secondHalfX = otherPlayerX + otherPlayerWidth / 2;

    return (
      playerX < otherPlayerX + otherPlayerWidth &&
      playerX + playerWidth > secondHalfX &&
      playerY < otherPlayerY + otherPlayerHeight &&
      playerY + playerHeight > otherPlayerY
    );
  }

  checkCollisionAttack(entity, target) {
    const targetMidX = target.x + target.width / 2;
    const entityLeft = entity.x - 5;
    const entityRight = entity.x + entity.width + 5;

    return entityRight >= targetMidX && entityLeft <= targetMidX;
  }
  isCollidingWithPlayer(player, deslocamento) {
    const {
      x: playerX,
      y: playerY,
      width: playerWidth,
      height: playerHeight,
    } = player;
    const {
      x: otherPlayerX,
      y: otherPlayerY,
      width: otherPlayerWidth,
      height: otherPlayerHeight,
    } = this.otherPlayer;

    return this.checkCollision(
      playerX + deslocamento,
      playerY,
      playerWidth,
      playerHeight,
      otherPlayerX,
      otherPlayerY,
      otherPlayerWidth,
      otherPlayerHeight
    );
  }
}
