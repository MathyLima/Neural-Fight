import { MovementHandler } from "../Movement/MovementHandler.js";
import { CollisionHandler } from "../Movement/CollisionHandler.js";
import { HealthBar } from "../UI/HealthBar.js";
import { Game } from "../Core/Game.js";
export class Fighter {
  constructor(config) {
    this.x = config.x;
    this.y = config.y;
    this.initialX = config.x;
    this.initialY = config.y;
    this.width = config.width;
    this.height = config.height;
    this.sprite = config.sprite;
    this.health = config.health;
    this.speed = config.speed;
    this.staggerFrame = config.staggerFrame || 20;
    this.velocity = { x: 0, y: 0 };
    this.sprite_Frame = 0;
    this.isAttacking = false;
    this.isBlocking = false;
    this.isJumping = false;
    this.animationState = "idle";
    this.sprite_map = config.sprite_map;
    this.actualFrame = 0;
    this.damageReduction = 0;
    this.roundKeys;

    this.poisonDuration = 0;
    this.enfraquecerDuration = 0;
    this.sangramentoDuration = 0;

    this.currentEffect = [];

    this.attackBox = {
      x: config.attackBox.x,
      y: config.attackBox.y,
      width: config.attackBox.width,
      height: config.attackBox.height,
      offset: config.attackBox.offset,
    };

    this.healthBar = new HealthBar({
      x: this.x,
      y: 10,
      width: 400,
      height: 30,
      maxHealth: config.health,
      currentHealth: config.health,
    });

    this.isActioning = false;
    this.colissionHandler = new CollisionHandler(
      { xMin: 0, xMax: config.map.width, yMin: 0, yMax: config.map.height },
      null
    );
    this.movement = new MovementHandler(this, this.colissionHandler);
    this.enemy = null;
    this.gravity = config.gravity || 0.5;
    this.ismovingLeft = false;
    this.ismovingRight = false;
    this.takingDamage = false;
    this.turno = config.turno;
    this.isCentered = false;
    this.inputing = false;
    this.numberInputs;
    this.pressedKeys = [];
    this.isAnimating = false;
    this.onAnimationEnd = null;
    this.movingToInitialPosition = false;
    this.attacks = config.inputMap;
    this.inputSession = false;
  }

  inputTime() {
    return;
  }

  inputTimeAutomatico() {
    const inputList = this.roundKeys;

    let pressedKeys = [];

    // ===== ANÁLISE DO ESTADO DO JOGO =====
    const vidaAtual = this.health;
    const vidaMaxima = 100;
    const percentualVida = vidaAtual / vidaMaxima;

    const temEfeito = this.currentEffect.length > 0;
    const tipoEfeito = temEfeito ? this.currentEffect : null;

    const modoAtaque = this.turno === "ataque";
    const modoDefesa = this.turno === "defesa";

    const vidaInimigo = this.opponentHealth;
    const percentualVidaInimigo = vidaInimigo / vidaMaxima;

    const temE = inputList.includes("e");
    const temR = inputList.includes("r");
    const temQ = inputList.includes("q");
    const temT = inputList.includes("t");
    const temW = inputList.includes("w");
    const temY = inputList.includes("y");
    const temU = inputList.includes("u");

    // ===== HEURÍSTICAS DE SELEÇÃO =====

    // Array de prioridades com peso para cada tecla
    // [tecla, prioridade_base, está_disponível]
    const prioridades = [
      ["e", 0, temE], // Cura - prioridade base 0
      ["r", 0, temR], // Remove efeito - prioridade base 0
      ["q", 0, temQ], // Dano - prioridade base 0
      ["t", 0, temT], // Efeito de dano ao longo do tempo - prioridade base 0
      ["w", 0, temW], // Defesa - prioridade base 0
      ["y", 0, temY], // Ataque especial - prioridade base 0
      ["u", 0, temU], // Defesa especial - prioridade base 0
    ];

    prioridades.forEach((tecla) => {
      const [letra, _, disponivel] = tecla;

      if (!disponivel) return;

      // Lógica para tecla 'e' (cura)
      if (letra === "e") {
        // Prioriza cura quando vida está baixa (exponencialmente mais importante conforme vida diminui)
        if (percentualVida < 0.9) {
          tecla[1] += 10 * Math.pow(1.5, (1 - percentualVida) * 10);
        }
      }

      // Lógica para tecla 'r' (remove efeito)
      else if (letra === "r") {
        // Alta prioridade se tem efeito negativo
        if (temEfeito) {
          tecla[1] += 15;
        } else if (temEfeito) {
          tecla[1] += 5;
        }
      }

      // Lógica para tecla 'q' (dano direto)
      else if (letra === "q") {
        // Prioriza dano quando em modo ataque
        if (modoAtaque) {
          tecla[1] += 12;
        } else {
          tecla[1] += 6;
        }
      }

      // Lógica para tecla 't' (dano ao longo do tempo)
      else if (letra === "t") {
        // Bom para início de combate
        if (modoAtaque && percentualVidaInimigo > 0.7) {
          tecla[1] += 14;
        } else if (modoAtaque) {
          tecla[1] += 8;
        }
      }

      // Lógica para tecla 'w' (defesa)
      else if (letra === "w") {
        // Prioriza defesa quando em modo defesa
        if (modoDefesa) {
          tecla[1] += 12;
        } else {
          tecla[1] += 4;
        }
      }

      // Lógica para tecla 'y' (ataque especial)
      else if (letra === "y") {
        if (modoAtaque && percentualVidaInimigo < 0.4) {
          tecla[1] += 15; // Golpe de misericórdia
        } else if (modoAtaque) {
          tecla[1] += 10;
        }
      }

      // Lógica para tecla 'u' (defesa especial)
      else if (letra === "u") {
        if (modoDefesa && percentualVida < 0.3) {
          tecla[1] += 15; // Defesa desesperada
        } else if (modoDefesa) {
          tecla[1] += 10;
        }
      }

      tecla[1] += Math.random() * 2;
    });

    const prioridadesFiltradas = prioridades
      .filter((tecla) => tecla[2])
      .sort((a, b) => b[1] - a[1]);

    const teclasEscolhidas = prioridadesFiltradas.slice(0, 4);

    if (teclasEscolhidas.length < 4) {
      const teclasRestantes = inputList.filter(
        (tecla) => !teclasEscolhidas.some((t) => t[0] === tecla)
      );

      while (teclasEscolhidas.length < 4 && teclasRestantes.length > 0) {
        const randomIndex = Math.floor(Math.random() * teclasRestantes.length);
        const teclaAleatoria = teclasRestantes.splice(randomIndex, 1)[0];
        teclasEscolhidas.push([teclaAleatoria, 0, true]);
      }
    }

    pressedKeys = teclasEscolhidas.map((tecla) => tecla[0]);

    if (pressedKeys.length < 4) {
      console.warn("Não há teclas suficientes disponíveis");
      while (pressedKeys.length < 4) {
        pressedKeys.push(pressedKeys[0]);
      }
    }

    console.log("Teclas geradas com heurísticas:", pressedKeys);
    this.pressedKeys = pressedKeys;
  }

  setEnemy(enemy) {
    this.enemy = enemy;
    this.colissionHandler.otherPlayer = enemy;
  }
  setRoundKeys(keysMap) {
    console.log(keysMap);
    this.roundKeys = keysMap;
  }

  moveLeft() {
    this.changeAnimationState("walk_left");
    this.ismovingLeft = true;
    this.ismovingRight = false;
    this.movement.moveLeft();
  }

  getStatusEffects() {
    const efeitos = ["veneno", "sangramento", "enfraquecer"];
    const resumo = [];

    for (let efeito of efeitos) {
      const quantidade = this.contarEfeito(this.currentEffect, efeito);
      let duracao = 0;

      switch (efeito) {
        case "veneno":
          duracao = this.poisonDuration;
          break;
        case "sangramento":
          duracao = this.sangramentoDuration;
          break;
        case "enfraquecer":
          duracao = this.enfraquecerDuration;
          break;
      }

      if (quantidade > 0) {
        resumo.push({
          tipo: efeito,
          quantidade,
          duracao,
        });
      }
    }

    return resumo;
  }

  moveUp() {
    this.changeAnimationState("jump");
    this.movement.moveUp();
  }

  moveRight() {
    this.animationState = "walk";
    this.ismovingRight = true;
    this.ismovingLeft = false;
    this.movement.moveRight();
  }

  update(context) {
    this.render(context);
    if (this.movingToInitialPosition) {
      this.movement.moveToInitialPosition();
    } else {
      if (!this.isCentered) {
        this.movement.moveToCenter();
      }
    }
    if (
      !this.colissionHandler.isCollidingWithMap(this, this.movement.getspeedX())
    ) {
      this.x += this.speed.x;
    } else {
      this.speed.x = 0;
    }

    if (
      this.isAttacking &&
      this.actualFrame === this.sprite_map[this.animationState].frames - 1
    ) {
      this.isAttacking = false;
      this.changeAnimationState("idle");
      if (typeof this.onAnimationEnd === "function") {
        this.onAnimationEnd();
        this.onAnimationEnd = null;
      }
    }

    this.sprite_Frame++;
  }

  envenenar() {
    this.currentEffect.push("veneno");
    this.poisonDuration = 4;
  }

  enfraquecer() {
    if (this.damageReduction < 50) {
      this.damageReduction += 10;
    }
    this.enfraquecerDuration = 3;
    if (!this.currentEffect.includes("enfraquecer")) {
      this.currentEffect.push("enfraquecer");
    }
  }

  sangramento() {
    this.currentEffect.push("sangramento");

    this.sangramentoDuration = 3;
  }

  curarEfeitos() {
    this.currentEffect = [];
    this.damageReduction = 0;
    this.sangramentoDuration = 0;
    this.poisonDuration = 0;
  }

  curar() {
    if (this.health + 10 >= 100) {
      this.health = 100;
      return;
    }

    this.health += 10;
  }

  block(type) {
    this.isBlocking = true;
    this.blockType = type;
    this.changeAnimationState("defend");
  }

  stopBlock() {
    this.isBlocking = false;
    this.blockType = null;
    this.changeAnimationState("idle");
  }

  takeDamage() {
    this.takingDamage = true;
    this.changeAnimationState("hurt");
  }

  stopTakeDamage(danoBase) {
    this.takingDamage = false;

    let modificadorEnfraquecido = this.enemy.damageReduction;

    let danoFinal = danoBase;

    if (modificadorEnfraquecido > 0) {
      danoFinal = danoBase * ((100 - modificadorEnfraquecido) / 100);
    }

    this.atualizarVida(danoFinal);
    this.changeAnimationState("idle");
  }

  atualizarVida(danoFinal) {
    this.health -= danoFinal;

    this.healthBar.update(this.health);

    if (this.health <= 0) {
      this.health = 0;
      document.getElementById("finalizaJogo").style.display = "flex";
      this.die();
      const gameInstance = Game.getInstance();
      gameInstance.gameEnded = true;

      location.reload();
    }
  }

  contarEfeito(lista, efeito) {
    return lista.filter((e) => e === efeito).length;
  }

  decrementMinZero(value) {
    return Math.max(0, value - 1);
  }

  takeDamageFinal() {
    let dano = 0;

    const qtdVeneno = this.contarEfeito(this.currentEffect, "veneno");
    const qtdSangramento = this.contarEfeito(this.currentEffect, "sangramento");
    if (this.poisonDuration > 0) {
      const danoVeneno = 2 * qtdVeneno;
      dano += danoVeneno;
    }

    if (this.sangramentoDuration > 0) {
      const danoSangramento = 3 * qtdSangramento;
      dano += danoSangramento;
    }

    this.sangramentoDuration = this.decrementMinZero(this.sangramentoDuration);
    this.poisonDuration = this.decrementMinZero(this.poisonDuration);

    if (this.poisonDuration === 0) {
      this.removerEfeito("veneno");
    }

    if (this.sangramentoDuration === 0) {
      this.removerEfeito("sangramento");
    }

    this.atualizarVida(dano);
  }

  removerEfeito(tipo) {
    const index = this.currentEffect.indexOf(tipo);
    if (index !== -1) {
      this.currentEffect.splice(index, 1);
    }
  }

  comecarCooldownTecla(tecla) {
    this.attacks[tecla].currentCooldown = this.attacks[tecla].cooldown;

    console.log(this.attacks[tecla]);
  }

  render(context) {
    const initialSpriteCut = { x: 0, y: 0 };
    const spriteWidth = this.sprite_map[this.animationState].frameWidth;
    const spriteHeight = this.sprite_map[this.animationState].frameHeight;
    this.attackBox.y = this.y + this.attackBox.offset.y;
    this.attackBox.x = this.x + this.attackBox.offset.x;

    let frameAnterior = this.actualFrame;
    this.actualFrame =
      Math.floor(this.sprite_Frame / this.staggerFrame) %
      this.sprite_map[this.animationState].frames;
    if (
      (this.isBlocking || this.takingDamage) &&
      this.actualFrame === this.sprite_map[this.animationState].frames - 1
    ) {
      this.actualFrame = frameAnterior;
      if (typeof this.onAnimationEnd === "function") {
        this.onAnimationEnd();
        this.onAnimationEnd = null;
      }
    }

    context.drawImage(
      this.sprite_map[this.animationState].image,
      spriteWidth * this.actualFrame,
      initialSpriteCut.y,
      spriteWidth,
      spriteHeight,
      this.x,
      this.y,
      this.width,
      this.height
    );

    context.strokeStyle = "white";
    context.lineWidth = 2;
    context.strokeRect(
      this.attackBox.x,
      this.attackBox.y,
      this.attackBox.width,
      this.attackBox.height
    );

    this.healthBar.render(context);
  }

  changeAnimationState(newState) {
    if (this.animationState === "die") return;
    const currentIsAttack = ["attack1", "attack2", "attack3"].includes(
      this.animationState
    );

    if (
      this.isAttacking &&
      currentIsAttack &&
      this.actualFrame < this.sprite_map[this.animationState].frames - 1
    ) {
      return;
    }

    if (this.animationState === newState) return;

    this.animationState = newState;
    this.sprite_Frame = 0;
    return;
  }

  die() {
    // Ainda não implementado.
  }
}
