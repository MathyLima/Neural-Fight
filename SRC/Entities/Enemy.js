import { Fighter } from "./Fighter.js";
import { Input } from "../Core/Input.js";

export class Enemy extends Fighter {
  constructor(name, config) {
    super(config);
    this.name = name;
    this.input = new Input();

    this.lastAttackTime = 0;
    this.attackCooldown = 1000;
  }

  updateAI() {
    throw new Error("implemente");
  }

  async inputTime(fazerPredicao) {
    const inputList = this.roundKeys;

    this.inputSession = true;

    this.pressedKeys = [];

    if (typeof fazerPredicao === "function") {
      try {
        const predicao = await fazerPredicao();

        if (
          predicao &&
          Array.isArray(predicao.keys) &&
          predicao.keys.length > 0
        ) {
          this.pressedKeys = predicao.keys;
        } else {
          for (let i = 0; i < 4; i++) {
            const randomKey =
              inputList[Math.floor(Math.random() * inputList.length)];
            this.pressedKeys.push(randomKey);
          }
        }
      } catch (error) {
        console.error("Erro ao fazer predição:", error);
        for (let i = 0; i < 4; i++) {
          const randomKey =
            inputList[Math.floor(Math.random() * inputList.length)];
          this.pressedKeys.push(randomKey);
        }
      }
    } else {
      for (let i = 0; i < 4; i++) {
        const randomKey =
          inputList[Math.floor(Math.random() * inputList.length)];
        this.pressedKeys.push(randomKey);
      }
    }

    this.inputSession = false;

    console.log(this.pressedKeys);
  }
}
