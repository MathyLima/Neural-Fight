import { Observer } from "../Notifier/Observer.js";

export class Input {
    constructor() {
        this.observer = new Observer();

        window.addEventListener("keydown", (e) => this.handleKey(e, true));
        window.addEventListener("keyup", (e) => this.handleKey(e, false));
    }

    handleKey(event, isPressed) {
        const key = event.key.toLowerCase();
        this.observer.notify(key, isPressed);
    }

    // Adiciona listeners para qualquer tecla definida pelo usuário
    onEvent(keys, callback) {
        if (Array.isArray(keys)) {
            keys.forEach((key) => this.observer.subscribe(key.toLowerCase(), callback));
        } else {
            this.observer.subscribe(keys.toLowerCase(), callback);
        }
    }

    // Remove listeners de teclas definidas pelo usuário
    removeEvent(keys, callback) {
        if (Array.isArray(keys)) {
            keys.forEach((key) => this.observer.unsubscribe(key.toLowerCase(), callback));
        } else {
            this.observer.unsubscribe(keys.toLowerCase(), callback);
        }
    }
}
