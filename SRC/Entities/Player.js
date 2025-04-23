import { Fighter } from "./Fighter.js";

export class Player extends Fighter {
  constructor(id, config) {
    super(config);
    console.log(id);
    this.id = id;
    this.input = config.input;
    this.lastKey = null;
    this.isMoving = false;
    this.keysPressed = {};
    this.inputTimeout = 500;
    this.lastInputTime = {};
    this.inputDelay = 200;
    this.lastInput = 0;
  }

  storePressedKey(key, inputList) {
    this.pressedKeys.push(key);
    const actualKey = this.pressedKeys.length - 1;

    Array.from(document.querySelectorAll(".inputSpace")).forEach(
      (input, index) => {
        if (index === actualKey) {
          input.querySelector(".inputRect").style.backgroundColor = "white";
          input.querySelector(".inputRect").querySelector("h1").innerText = key;
        }
      }
    );
  }

  inputTime() {
    const inputList = this.roundKeys;
    this.inputSession = true;
    if (this._inputCallback && this._inputKeys) {
      this.input.removeEvent(this._inputKeys, this._inputCallback);
    }

    const pressedKeysSet = new Set();

    const callback = (key, isPressed) => {
      const currentTime = Date.now();

      if (currentTime - this.lastInput < this.inputDelay) return;
      this.lastInput = currentTime;

      if (isPressed) {
        this.lastInputTime[key] = currentTime;
        this.keysPressed[key] = true;

        const beforeCount = this.pressedKeys.length;
        this.storePressedKey(key, inputList);
        const afterCount = this.pressedKeys.length;

        if (afterCount > beforeCount) {
          pressedKeysSet.add(key);

          console.log(key);
          if (afterCount >= this.numberInputs) {
            inputList.forEach((k) => {
              this.input.removeEvent(k, callback);
            });

            this.inputing = false;

            return;
          }
        }
      } else {
        this.keysPressed[key] = false;
      }
    };

    this._inputCallback = callback;
    this._inputKeys = inputList;

    this.input.onEvent(inputList, callback);

    this._cancelClickListener = () => {
      this.input.removeEvent(inputList, callback);
      document.removeEventListener("click", this._cancelClickListener);
    };
    document.addEventListener("click", this._cancelClickListener);
  }
}
