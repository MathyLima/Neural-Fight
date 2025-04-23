export class Game {
  static #instance = null;
  constructor(config) {
    this.round = config.round;
    this.fighters = config.fighters;
    this.game_state = config.game_state;
    this.renderer = config.renderer;
    this.canvas = config.canvas;
    this.context = config.context;
    this.map = config.map;
    this.audio = new Audio("../../Assets/Audios/Background.mp3");
    this.animationFrameId = null;
    this.animate = this.animate.bind(this);
    this.numberInputs;
    this.inputMap = config.inputMap;

    this.gameEnded = false;
    this.availableKeys = [];

    this.attackType = [];
    this.defenseType = [];
    this.server = config.server;
    this.turnAttacks = [];
    if (!Game.#instance) {
      Game.#instance = this;
    }
  }

  static getInstance() {
    return Game.#instance;
  }

  decreaseCooldowns() {
    this.fighters.forEach((fighter) => {
      if (fighter.turno === "ataque") {
        Object.keys(fighter.attacks).forEach((key) => {
          const attack = fighter.attacks[key];
          attack.currentCooldown = Math.max(attack.currentCooldown - 1, 0);
        });
      }
    });
  }

  addRound() {
    this.round += 1;

    this.decreaseCooldowns();
  }
  startGame() {
    this.animate();
  }

  drawImage(image, x, y, width, height) {
    return function (context) {
      context.drawImage(image, x, y, width, height);
    };
  }

  drawBackgroundWithMessage() {
    this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.renderer.draw(
      this.drawImage(this.map.background, 0, 0, this.map.width, this.map.height)
    );

    this.context.fillStyle = "white";
    this.context.font = "30px Arial";
    this.context.textAlign = "center";
    this.context.fillText(
      "Clique no jogo para iniciar",
      this.canvas.width / 2,
      this.canvas.height / 2
    );
  }

  inicioJogo() {
    this.drawBackgroundWithMessage();
    this.canvas.onclick = () => {
      this.game_state.gameStarted = true;
      this.audio.volume = 0.1;
      this.audio.loop = true;
      this.audio.play().catch((error) => {
        console.error("Erro ao reproduzir a música:", error);
      });
    };
  }

  animate() {
    if (!this.game_state.gameStarted) {
      this.inicioJogo();
    } else {
      this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
      this.renderer.draw(
        this.drawImage(
          this.map.background,
          0,
          0,
          this.map.width,
          this.map.height
        )
      );
      const text = `${this.round}`;
      this.context.font = "bold 40px Arial";
      this.context.textAlign = "center";
      this.context.textBaseline = "top";

      const x = this.canvas.width / 2;
      const y = 20;

      const metrics = this.context.measureText(text);
      const padding = 20;
      const rectWidth = metrics.width + padding * 2;
      const rectHeight = 40;
      const radius = 10;

      function roundRect(ctx, x, y, width, height, radius) {
        ctx.beginPath();
        ctx.moveTo(x + radius, y);
        ctx.lineTo(x + width - radius, y);
        ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
        ctx.lineTo(x + width, y + height - radius);
        ctx.quadraticCurveTo(
          x + width,
          y + height,
          x + width - radius,
          y + height
        );
        ctx.lineTo(x + radius, y + height);
        ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
        ctx.lineTo(x, y + radius);
        ctx.quadraticCurveTo(x, y, x + radius, y);
        ctx.closePath();
        ctx.fill();
      }

      this.context.fillStyle = "black";
      roundRect(
        this.context,
        x - rectWidth / 2,
        y,
        rectWidth,
        rectHeight,
        radius
      );

      this.context.fillStyle = "white";
      this.context.fillText(text, x, y + (rectHeight - 36) / 2);
      this.fighters.forEach((fighter) => {
        this.renderer.draw((ctx) => {
          fighter.update(ctx);
        });
      });

      const allCentered = this.fighters.every((f) => f.isCentered);

      if (allCentered) {
        if (!this.numberInputs) this.numberInputs = 4;
        this.determinaInputsMostrados();
        this.fighters.forEach((fighter) => {
          if (!fighter.inputing) {
            fighter.numberInputs = this.numberInputs;
            fighter.inputing = true;
            fighter.inputSession = false;
          }
        });

        const allPlayersReady = this.fighters.every(
          (f) => f.pressedKeys.length >= this.numberInputs
        );
        if (
          (allPlayersReady ||
            this.attackType.length >= this.numberInputs ||
            this.defenseType.length >= this.numberInputs) &&
          !this.executandoRodada
        ) {
          setTimeout(() => {
            document.getElementById("Input").style.display = "none";
          }, 200);

          if (this.fighters[0]._cancelClickListener) {
            document.removeEventListener(
              "click",
              this.fighters[0]._cancelClickListener
            );
            this.fighters[0]._cancelClickListener;
          }

          this.executandoRodada = true;

          this.fighters.forEach((fighter) => {
            const keys = fighter.pressedKeys.slice(0, this.numberInputs);

            if (this.attackType.length === 0 || this.defenseType.length === 0) {
              keys.forEach((key) => {
                if (fighter.turno === "ataque") {
                  this.attackType.push(key);
                }

                if (fighter.turno === "defesa") {
                  this.defenseType.push(key);
                }
              });
            }
          });

          const cooldownsPlayer1 = {};
          Object.entries(this.fighters[0].attacks).forEach(([key, attack]) => {
            cooldownsPlayer1[key] = attack.currentCooldown;
          });

          const cooldownsPlayer2 = {};
          Object.entries(this.fighters[1].attacks).forEach(([key, attack]) => {
            cooldownsPlayer2[key] = attack.currentCooldown;
          });
          const teclasPressionadas = this.attackType;
          const frozenAttack = [...this.attackType];
          console.log("Teclas antes do envio:", teclasPressionadas);
          const dadosJogo = this.gerarEstadoDoJogo({
            turnoJogador1: this.fighters[0].turno,
            numeroInputs: this.numberInputs,
            id: this.fighters[0].id,
            vidaJogador1: this.fighters[0].health,
            vidaJogador2: this.fighters[1].health,
            turnoAtual: this.round,
            teclasPressionadasJogador1: Array.isArray(teclasPressionadas)
              ? [...teclasPressionadas]
              : teclasPressionadas,
            efeitosJogador1: JSON.parse(
              JSON.stringify(this.fighters[0].getStatusEffects())
            ),
            efeitosJogador2: JSON.parse(
              JSON.stringify(this.fighters[1].getStatusEffects())
            ),
            teclasDisponiveis: JSON.parse(JSON.stringify(this.availableKeys)),
          });

          this.correctDefense().then(async (result) => {
            for (let i = 0; i < this.numberInputs; i++) {
              for (const fighter of this.fighters) {
                if (fighter.turno === "ataque") {
                  const attackType1 = ["a", "q", "e"];
                  const attackType2 = ["t", "s", "e", "w"];
                  const attackType3 = ["r"];
                  const attack = this.attackType[0];
                  let attackType;
                  if (attackType1.includes(attack)) {
                    attackType = "attack1";
                  } else if (attackType2.includes(attack)) {
                    attackType = "attack2";
                  } else if (attackType3.includes(attack)) {
                    attackType = "attack3";
                  } else {
                    attackType = "unknown";
                  }

                  fighter.isAttacking = true;
                  fighter.changeAnimationState(attackType);

                  await this.waitForAnimationEnd(fighter);
                  this.attackType.shift();
                  fighter.pressedKeys.shift();
                  fighter.changeAnimationState("idle");
                  fighter.isAttacking = false;
                }

                if (fighter.turno === "defesa") {
                  const defense = this.defenseType.shift();
                  fighter.pressedKeys.shift();
                  if (result[i]) {
                    fighter.block(defense);
                    const sound = new Audio();
                    sound.src = "../../Assets/Audios/swordBlock.mp3";
                    sound.volume = 0.3;
                    sound.play().catch((error) => {
                      console.error(
                        "Erro ao tentar reproduzir o áudio:",
                        error
                      );
                    });
                    await this.waitForAnimationEnd(fighter);
                    fighter.stopBlock();
                  } else {
                    let danoBase;
                    const primeiroAtaque = this.turnAttacks.shift();

                    switch (primeiroAtaque) {
                      case "q":
                        danoBase = 5;
                        fighter.envenenar();
                        break;
                      case "w":
                        danoBase = 3;
                        fighter.enfraquecer();
                        break;
                      case "e":
                        danoBase = 1;
                        fighter.enemy.curar();
                        break;
                      case "r":
                        danoBase = 0.5;
                        fighter.enemy.curarEfeitos();
                        break;
                      case "t":
                        danoBase = 6;
                        fighter.sangramento();
                        break;
                      case "a":
                        danoBase = 7;
                        break;
                      case "s":
                        danoBase = 10;
                        break;
                    }

                    fighter.takeDamage();
                    const sound = new Audio();
                    sound.src = "../../Assets/Audios/swordHurt.mp3";
                    sound.volume = 0.3;
                    sound.play().catch((error) => {
                      console.error(
                        "Erro ao tentar reproduzir o áudio:",
                        error
                      );
                    });
                    await this.waitForAnimationEnd(fighter);
                    fighter.stopTakeDamage(danoBase);
                  }
                }
              }

              await new Promise((r) => setTimeout(r, 200));
            }

            this.fighters.forEach((f) => {
              if (f.turno === "ataque") {
                f.enfraquecerDuration = f.decrementMinZero(
                  f.enfraquecerDuration
                );
                if (f.enfraquecerDuration === 0) f.damageReduction = 0;
                frozenAttack.forEach((key) => {
                  f.comecarCooldownTecla(key);
                });
              }
              f.turno = f.turno === "ataque" ? "defesa" : "ataque";
              f.inputing = false;
              f.movement.moveToInitialPosition();
              f.keysPressed = {};
            });
            this.numberInputs = null;

            this.executandoRodada = false;
            this.server.enviarEstadoDoJogo(dadosJogo);
          });
        }
      }
    }
    this.animationFrameId = requestAnimationFrame(this.animate);
  }

  waitForAnimationEnd(fighter) {
    return new Promise((resolve) => {
      fighter.onAnimationEnd = resolve;
    });
  }

  abrirTelaInput() {
    this.fighters.forEach((fighter) => {
      fighter.keysPressed = {};
      fighter.pressedKeys = [];
    });
    Array.from(document.querySelectorAll(".inputSpace")).forEach((input) => {
      const rect = input.querySelector(".inputRect");
      rect.style.backgroundColor = ""; // ou uma cor padrão tipo 'transparent'
      const h1 = rect.querySelector("h1");
      if (h1) h1.innerText = "";
    });

    document.getElementById(
      "turnoDescription"
    ).innerText = `TURNO:${this.round}`;

    // Para ataque, somente mostrar teclas que não estão em cooldown
    this.availableKeys = this.getAvailableAttackKeys();

    document.getElementById(
      "teclasDisponiveis"
    ).innerText = `TECLAS DISPONÍVEIS: ${this.availableKeys}`;

    document.getElementById("Input").style.display = "flex";

    //inicia o input listener de cada jogador
    this.fighters.forEach(async (fighter, index) => {
      if (index === 0) {
        fighter.inputTime();
      } else {
        this.fazerPredicao(300).then((previsao) => {
          fighter.pressedKeys = previsao;
          fighter.inputing = false;

          console.log(fighter.inputKeys);
        });
      }
    });
  }

  getAvailableAttackKeys() {
    const availableKeys = [];
    const attackingFighter = this.fighters.find(
      (fighter) => fighter.turno === "ataque"
    );
    if (attackingFighter) {
      const attackTypeMap = {
        q: "Envenenar",
        w: "Enfraquecer",
        e: "Cura",
        r: "Cura de Efeitos",
        t: "Sangramento",
        a: "Ataque Básico",
        s: "Ataque Pesado",
      };
      // Percorrer todas as teclas no inputMap
      Object.keys(attackingFighter.attacks).forEach((key) => {
        const attack = attackingFighter.attacks[key];
        if (attack.currentCooldown === 0) {
          // Se o cooldown for 0, o ataque pode ser feito
          availableKeys.push(`${key} (${attackTypeMap[key]})`); // Adiciona a tecla com o tipo de ataque
        }
      });
    }
    this.fighters.forEach((f) =>
      f.setRoundKeys([...availableKeys.map((k) => k.split(" ")[0])])
    );
    return availableKeys.join(","); // Retorna as teclas disponíveis separadas por vírgula
  }

  determinaInputsMostrados() {
    const numeroInputs = this.numberInputs;
    Array.from(document.querySelectorAll(".inputSpace")).forEach(
      (input, index) => {
        if (index > numeroInputs - 1) {
          input.style.display = "none";
        } else {
          input.style.display = "flex";
        }
      }
    );
  }

  gerarEstadoDoJogo({
    turnoJogador1,
    id,
    vidaJogador1,
    vidaJogador2,
    teclasPressionadasJogador1,
    turnoAtual,
    efeitosJogador1,
    efeitosJogador2,
    teclasDisponiveis,
  }) {
    console.log(turnoAtual);
    const estado = {
      turnoJogador1: turnoJogador1,
      id: id,
      numeroInputs: this.numberInputs,
      vida: {
        jogador1: vidaJogador1,
        jogador2: vidaJogador2,
      },
      teclasPressionadasJogador1: teclasPressionadasJogador1,
      turnoAtual: turnoAtual,
      efeitos: {
        jogador1: this.formatarEfeitos(efeitosJogador1, this.fighters[0]),
        jogador2: this.formatarEfeitos(efeitosJogador2, this.fighters[1]),
      },

      teclasDisponiveis: teclasDisponiveis,
    };
    console.log(estado.teclasPressionadasJogador1);
    return estado;
  }

  contarEfeito(lista, efeito) {
    console.log(lista);
    return lista.filter((e) => e === efeito).length;
  }

  formatarEfeitos(efeitosJogador1, jogador) {
    const efeitos = ["veneno", "sangramento", "enfraquecer"];
    const resultado = [];

    for (let tipo of efeitos) {
      const quantidade = this.contarEfeito(efeitosJogador1, tipo);
      let duracao = 0;

      switch (tipo) {
        case "veneno":
          duracao = jogador.poisonDuration;
          break;
        case "sangramento":
          duracao = jogador.sangramentoDuration;
          break;
        case "enfraquecer":
          duracao = jogador.enfraquecerDuration;
          break;
      }

      if (quantidade > 0) {
        resultado.push({ tipo, quantidade, duracao });
      }
    }

    return resultado;
  }
  correctDefense() {
    return new Promise((resolve) => {
      const answers = [];
      // Define quais tipos de defesa bloqueiam quais ataques

      this.attackType.forEach((attack, index) => {
        this.turnAttacks.push(attack);
        answers.push(attack === this.defenseType[index]);
      });

      resolve(answers);
    });
  }

  stopAnimation() {
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
  }

  resumeAnimation() {
    if (!this.animationFrameId) {
      this.animationFrameId = requestAnimationFrame(this.animate);
    }
  }

  gerarDadosPredicao(availableKeys) {
    const defenderFighter = this.fighters.find(
      (fighter) => fighter.turno === "defesa"
    );
    const attackerFighter = this.fighters.find(
      (fighter) => fighter.turno === "ataque"
    );

    if (!defenderFighter || !attackerFighter) {
      console.error("Não foi possível encontrar os lutadores para predição");
      return null;
    }

    // Obter efeitos de status atuais
    const attackerEffects = JSON.parse(
      JSON.stringify(attackerFighter.getStatusEffects())
    );
    const defenderEffects = JSON.parse(
      JSON.stringify(defenderFighter.getStatusEffects())
    );

    // Formar os dados para predição
    const dadosPredicao = {
      turnoJogador1: "ataque",
      id: defenderFighter.id,
      numeroInputs: this.numberInputs || 4,
      vida: {
        jogador1: this.fighters[0].health,
        jogador2: this.fighters[1].health,
      },
      teclasPressionadasJogador1: Array.isArray(this.attackType)
        ? [...this.attackType]
        : [this.attackType],
      turnoAtual: this.round,
      efeitos: {
        jogador1: this.formatarEfeitos(attackerEffects, this.fighters[0]),
        jogador2: this.formatarEfeitos(defenderEffects, this.fighters[1]),
      },
      teclasDisponiveis: availableKeys,
    };

    return dadosPredicao;
  }

  async fazerPredicao(contexto) {
    try {
      // Verificar se o servidor está conectado
      if (!this.server || !this.server.isConnected) {
        console.error("Servidor não está conectado");
        return null;
      }

      // Gerar dados para predição
      const dadosPredicao = this.gerarDadosPredicao(
        JSON.stringify(this.availableKeys)
      );

      if (!dadosPredicao) {
        console.error("Não foi possível gerar dados para predição");
        return null;
      }

      console.log("Enviando dados para predição:", dadosPredicao);

      const teclasPrevistas = await this.server.predicao(
        contexto,
        dadosPredicao
      );

      console.log("Teclas previstas recebidas:", teclasPrevistas);
      console.log(teclasPrevistas);
      return teclasPrevistas;
    } catch (error) {
      console.error("Erro ao fazer predição:", error);
      return null;
    }
  }
}
