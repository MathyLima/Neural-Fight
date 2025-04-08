import { initialConfig,loadAllImages} from "./config.js";
import { Renderer} from "./Core/Renderer.js";
import { Player } from "./Entities/Player.js";

loadAllImages().then(() => {
        
    const canvas = document.getElementById('gameContainer');
    const renderer = new Renderer(canvas);
    const context = renderer.getContext();
    const config_Player1 = {
        x: initialConfig.fighters.player1.x,
        y: initialConfig.fighters.player1.y,
        width: 100,
        height: 100,
        sprite_map: initialConfig.fighters.player1.sprite_map,
        health: initialConfig.game_state.player1Health,
        speed: 5,
    }

    const player1 = new Player('Player1', config_Player1);



    const map = initialConfig.map;


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
        
        requestAnimationFrame(animate);
    }

    animate()

}).catch((error) => {
    console.error('Error loading images:', error);
});