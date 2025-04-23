import { Game } from "../Core/Game.js";
export class MovementHandler {
  constructor(entity, collisionHandler) {
    this.entity = entity;
    this.collisionHandler = collisionHandler;
    this.speed = { x: 5, y: 20 };
  }

  getspeedX() {
    return this.speed.x;
  }
  getspeedY() {
    return this.speed.y;
  }
  moveLeft() {
    this.speed.x = -Math.abs(this.speed.x);

    this.entity.speed.x = this.speed.x;
  }

  moveRight() {
    this.speed.x = Math.abs(this.speed.x);
    this.entity.speed.x = this.speed.x;
  }

  moveUp() {
    this.speed.y = -Math.abs(this.speed.y);
    this.entity.speed.y = this.speed.y;
  }
  moveDown() {}

  stopHorizontal() {
    this.entity.speed.x = 0;
    this.entity.ismovingRight = false;
    this.entity.ismovingLeft = false;
    this.entity.changeAnimationState("idle");
  }

  stopVertical() {
    this.entity.speed.y = 0;
  }

  moveToCenter() {
    const centerX = this.collisionHandler.mapBounds.xMax / 2;

    const entity = this.entity;
    const attackBoxCenterX =
      entity.x + entity.attackBox.offset.x + entity.attackBox.width / 2;

    const delta = centerX - attackBoxCenterX;

    if (Math.abs(delta) > Math.abs(this.entity.speed.x)) {
      if (delta > 0) {
        this.moveRight();
        this.entity.moveRight();
        entity.ismovingRight = true;
        entity.ismovingLeft = false;
      } else {
        this.moveLeft();
        this.entity.moveLeft();
        entity.ismovingLeft = true;
        entity.ismovingRight = false;
      }
    } else {
      this.entity.isCentered = true;
      this.stopHorizontal();
      const targetX =
        centerX -
        this.entity.attackBox.offset.x -
        this.entity.attackBox.width / 2;
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
    this.entity.movingToInitialPosition = true;
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
      this.stopHorizontal();
      this.entity.x = targetX;
      this.entity.movingToInitialPosition = false;
      const gameInstance = Game.getInstance();
      this.entity.takeDamageFinal();

      const proximaRodadaTela = document.getElementById("proximaRodada");
      if (!gameInstance.gameEnded) {
        proximaRodadaTela.style.display = "flex";
      }

      proximaRodadaTela.querySelector("#proximaRodadaBotao").onclick = () => {
        proximaRodadaTela.style.display = "none";
        this.entity.isCentered = false;
        this.entity.enemy.isCentered = false;
        gameInstance.addRound();
      };
    }
  }
}
