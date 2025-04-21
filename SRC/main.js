import { initialConfig,loadAllImages} from "./config.js";
import { Renderer} from "./Core/Renderer.js";
import { Enemy } from "./Entities/Enemy.js";
import { Player } from "./Entities/Player.js";
import { Input } from "./Core/Input.js";
import { ServerCommunicator } from "./Core/ServerComunicator.js";
import { Game } from "./Core/Game.js";
loadAllImages().then(() => {
        
    const canvas = document.getElementById('gameContainer');
    const renderer = new Renderer(canvas);
    const context = renderer.getContext();
    const config_Player1 = {
        x: initialConfig.fighters.player1.x,
        y: initialConfig.fighters.player1.y,
        width: 300,
        height: 550,
        turno:'ataque',
        sprite_map: initialConfig.fighters.player1.sprite_map,
        health: initialConfig.game_state.player1Health,
        speed: {x: 0, y: 0},    
        map: initialConfig.map,
        staggerFrame: 2,
        input: new Input(),
        attackBox: {
            x: initialConfig.fighters.player1.x - 50,
            y: initialConfig.fighters.player1.y,
            width: 50,
            height: 100,
            offset: { x: 200, y: 330 },
        },
        inputMap: JSON.parse(JSON.stringify(initialConfig.inputMap))
    }

    const config_Player2 = {
        x: initialConfig.fighters.enemy1.x,
        y: initialConfig.fighters.enemy1.y,
        width: 300,
        height: 550,
        
        turno:'defesa',

        sprite_map: initialConfig.fighters.enemy1.sprite_map,
        health: initialConfig.game_state.player1Health,
        
        speed: {x: 0, y: 0},    
        
        map: initialConfig.map,
        
        staggerFrame: 2,

        attackBox: {
            x: initialConfig.fighters.enemy1.x - 50,
            y: initialConfig.fighters.enemy1.y,
            width: 50,
            height: 100,
            offset: { x: 0, y: 330 },
        },
        inputMap: JSON.parse(JSON.stringify(initialConfig.inputMap))

    }

    const config_Server = {
        host: initialConfig.server.host,
        porta: initialConfig.server.porta,
        input: config_Player1.input,
    }

    const server = new ServerCommunicator(config_Server);

    const map = initialConfig.map;
    const player1 = new Player('Player1', config_Player1);

    const player2 = new Enemy('Enemy1', config_Player2);
    player1.setEnemy(player2);
    player2.setEnemy(player1);
    

    const config_game={
        fighters:[player1,player2],
        round:0,
        game_state:initialConfig.game_state,
        renderer:renderer,
        canvas:canvas,
        context:context,
        map:map,
        server:server
    }

    const game = new Game(config_game);
    game.startGame();
}).catch((error) => {
    console.error('Error loading images:', error);
});