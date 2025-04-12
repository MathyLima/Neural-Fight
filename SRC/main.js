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
        width: 200,
        height: 350,
        sprite_map: initialConfig.fighters.player1.sprite_map,
        health: initialConfig.game_state.player1Health,
        speed: {x: 0, y: 0},    
        map: initialConfig.map,
        staggerFrame: 20,
        input: new Input(),
        attackBox: {
            x: initialConfig.fighters.player1.x,
            y: initialConfig.fighters.player1.y,
            width: 50,
            height: 100,
            offset: { x: 140, y: 150 },
        },
    }

    const config_Player2 = {
        x: initialConfig.fighters.enemy1.x,
        y: initialConfig.fighters.enemy1.y,
        width: 200,
        height: 350,
        sprite_map: initialConfig.fighters.enemy1.sprite_map,
        health: initialConfig.game_state.player1Health,
        speed: {x: 0, y: 0},    
        map: initialConfig.map,
        staggerFrame: 20,

        attackBox: {
            x: initialConfig.fighters.enemy1.x,
            y: initialConfig.fighters.enemy1.y,
            width: 50,
            height: 100,
            offset: { x: 0, y: 200 },
        },

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

    let isGameStarted = false; // Variável para controlar o estado do jogo
    const backgroundMusic = new Audio('../Assets/Audios/Background.mp3'); // Substitua pelo caminho da sua música

    function drawBackgroundWithMessage() {
        // Renderiza o fundo
        context.clearRect(0, 0, canvas.width, canvas.height);
        renderer.draw(drawImage(map.background, 0, 0, map.width, map.height));
    
        // Renderiza a mensagem
        context.fillStyle = 'white'; // Cor do texto
        context.font = '30px Arial'; // Fonte do texto
        context.textAlign = 'center'; // Alinhamento do texto
        context.fillText('Clique no jogo para iniciar', canvas.width / 2, canvas.height / 2);
    }
    
    // Adiciona um evento de clique para iniciar o jogo
    canvas.addEventListener('click', () => {
        if (!isGameStarted) {
            isGameStarted = true; // Atualiza o estado para iniciar o jogo
            backgroundMusic.volume = 0.1; // Ajusta o volume da música
            backgroundMusic.loop = true; // Define a música para tocar em loop
            backgroundMusic.play().catch((error) => {
                console.error('Erro ao reproduzir a música:', error);
            });
        }
    });


    let frameCounter = 0; // Contador de frames

    
    function animate() {
        if (!isGameStarted) {
            drawBackgroundWithMessage(); // Renderiza apenas o fundo e a mensagem
        } else {
            context.clearRect(0, 0, canvas.width, canvas.height);
            renderer.draw(drawImage(map.background, 0, 0, map.width, map.height));
            renderer.draw((ctx) => {
                player1.update(ctx);
            });
            

            renderer.draw((ctx) => {
                player2.update(ctx);
            })

            player2.updateAI(); // Atualiza a IA do inimigo
            frameCounter++; // Incrementa o contador de frames

        }
    
        requestAnimationFrame(animate);
    }
    
    animate();

}).catch((error) => {
    console.error('Error loading images:', error);
});