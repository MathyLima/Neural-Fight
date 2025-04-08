export class Game {
    constructor(config){
        this.round = config.round; // Número da rodada
        this.fighters = config.fighters; // Lista de lutadores
        this.game_state = config.game_state; // Estado do jogo
        this.renderer = config.renderer; // Renderizador
        this.canvas = config.canvas; // Canvas do jogo
        this.context = config.context; // Contexto do canvas
        this.map = config.map; // Referência ao mapa
        this.players = config.players; // Lista de jogadores
        this.start();
    }

      // Inicia o jogo
    start() {
        console.log("Jogo iniciado!");
        this.game_state.isPaused = false;
        this.gameLoop();
    }

    // Pausa o jogo
    pause() {
        console.log("Jogo pausado!");
        this.game_state.isPaused = true;
    }

    // Atualiza o estado do jogo
    update() {
        if (this.game_state.isPaused || this.game_state.isGameOver) return;

        // Atualiza todos os lutadores
        this.fighters.forEach((fighter) => fighter.update());
    }

    drawImage(image, x, y, width, height) {
        return function (context) {
            context.drawImage(image, x, y, width, height);
        };
    }

    // Renderiza o jogo
    render() {
        // Limpa o canvas
        this.context.clearRect(0, 0, this.map.width, this.map.height);
        this.renderer.draw(
            this.drawImage(this.map.background, 0, 0, this.map.width, this.map.height)
        )

        // Renderiza todos os lutadores
        this.fighters.forEach((fighter) => this.renderer.draw(()=>{fighter.render(this.context)}));
    }

    // Loop principal do jogo
    gameLoop() {
        if (!this.game_state.isPaused && !this.game_state.isGameOver) {
            this.render();
            requestAnimationFrame(() => this.gameLoop());
        }
    }

    // Finaliza o jogo
    end() {
        console.log("Jogo finalizado!");
        this.game_state.isGameOver = true;
    }

    // Reinicia o jogo
    restart() {
        console.log("Jogo reiniciado!");
        this.game_state.isGameOver = false;
        this.game_state.currentRound = 1;
        this.fighters.forEach((fighter) => {
            fighter.health = 100; // Reinicia a saúde dos lutadores
            fighter.animationState = 'idle'; // Reinicia o estado de animação
        });
        this.start();
    }



}