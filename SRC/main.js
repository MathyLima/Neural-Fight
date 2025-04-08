import { initialConfig,loadAllImages} from "./config.js";
import { Renderer} from "./Core/Renderer.js";
import { Enemy } from "./Entities/Enemy.js";
import { Player } from "./Entities/Player.js";
import { Input } from "./Core/Input.js";
import { ServerCommunicator } from "./Core/ServerComunicator.js";
loadAllImages().then(() => {
        
    const canvas = document.getElementById('gameContainer');
    const renderer = new Renderer(canvas);
    const context = renderer.getContext();
    const config_Player1 = {
        x: initialConfig.fighters.player1.x,
        y: initialConfig.fighters.player1.y,
        width: 300,
        height: 350,
        sprite_map: initialConfig.fighters.player1.sprite_map,
        health: initialConfig.game_state.player1Health,
        speed: 5,
        map: initialConfig.map,
        staggerFrame: 30,
        input: new Input(),
    }

    const config_Player2 = {
        x: initialConfig.fighters.enemy1.x,
        y: initialConfig.fighters.enemy1.y,
        width: 300,
        height: 350,
        sprite_map: initialConfig.fighters.enemy1.sprite_map,
        health: initialConfig.game_state.player1Health,
        speed: 5,    
        map: initialConfig.map,
        staggerFrame: 30,
    }

    const config_Server = {
        host: initialConfig.server.host,
        porta: initialConfig.server.porta,
        input: config_Player1.input,
    }

    const map = initialConfig.map;
    const player1 = new Player('Player1', config_Player1);

    const player2 = new Enemy('Enemy1', config_Player2);
    player1.setEnemy(player2);
    player2.setEnemy(player1);
    
    const server = new ServerCommunicator(config_Server);

    function drawImage(image, x, y, width, height) {
        return function (context) {
            context.drawImage(image, x, y, width, height);
        };
    }


    function animate() {
        context.clearRect(0, 0, canvas.width, canvas.height);
        renderer.draw(drawImage(map.background, 0, 0, map.width, map.height));
        renderer.draw((ctx) => {
            player1.render(ctx);
        });

        renderer.draw((ctx) => {
            player2.render(ctx);
        });
        
        requestAnimationFrame(animate);
    }

    animate()

}).catch((error) => {
    console.error('Error loading images:', error);
});