function loadImage(src) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.src = src;
        img.onload = () => {
            resolve(img);
        }
        img.onerror = (err) => reject(err);
    });
}

const playerSpriteMap = {
    idle: {image:loadImage('../Assets/Sprites/Player/Knight_1/Idle.png'),
           frames:4,
           frameWidth: 67,
           frameHeight: 86,
    },
    attack1: {
           image:loadImage('../Assets/Sprites/Player/Knight_1/Attack 1.png'),
           frames: 5,
           frameWidth: 86,
           frameHeight: 86,
        },
    attack2: {
            image:loadImage('../Assets/Sprites/Player/Knight_1/Attack 2.png'),
            frames: 4,
            frameWidth: 107.5,
            frameHeight: 86,
        },
    attack3: {
            image:loadImage('../Assets/Sprites/Player/Knight_1/Attack 3.png'),
            frames: 4,
            frameWidth: 100,
            frameHeight: 86,
        },
    defend:{
            image:loadImage('../Assets/Sprites/Player/Knight_1/Defend.png'),
            frames: 5,
            frameWidth: 80,
            frameHeight: 86,

    },
    jump:{
            image:loadImage('../Assets/Sprites/Player/Knight_1/Jump.png'),
            frames: 6,
            frameWidth: 80,
            frameHeight: 86,
    },
    walk:{
            image:loadImage('../Assets/Sprites/Player/Knight_1/Run.png'),
            frames: 7,
            frameWidth: 70,
            frameHeight: 86,
    },
   
}


const EnemySpriteMap = {
    idle: {image:loadImage('../Assets/Sprites/Player/Knight_2/Idle.png'),
           frames:4,
           frameWidth: 67,
           frameHeight: 86,
    },
    attack1: {
            image:loadImage('../Assets/Sprites/Player/Knight_2/Attack 1.png'),
           frames: 5,
           frameWidth: 86,
           frameHeight: 86,
        },
    attack2: {
            image:loadImage('../Assets/Sprites/Player/Knight_2/Attack 2.png'),
            frames: 4,
            frameWidth: 107.5,
            frameHeight: 86,
        },
    attack3: {
            image:loadImage('../Assets/Sprites/Player/Knight_2/Attack 3.png'),
            frames: 4,
            frameWidth: 100,
            frameHeight: 86,
        },
    defend:{
            image:loadImage('../Assets/Sprites/Player/Knight_2/Defend.png'),
            frames: 5,
            frameWidth: 80,
            frameHeight: 86,

    },
    jump:{
            image:loadImage('../Assets/Sprites/Player/Knight_2/Jump.png'),
            frames: 6,
            frameWidth: 80,
            frameHeight: 86,
    },
    walk:{
            image:loadImage('../Assets/Sprites/Player/Knight_2/Walk.png'),
            frames: 8,
            frameWidth: 72.5,
            frameHeight: 86,
    },
   
}





export const initialConfig = Object.freeze({
    fighters:{
        player1:{
            x: document.getElementById('gameContainer').offsetWidth * 0.2,
            y: document.getElementById('gameContainer').offsetHeight * 0.8,
            sprite_map: playerSpriteMap,
        },
        enemy1:{
            x: document.getElementById('gameContainer').offsetWidth - 0.8,
            y: document.getElementById('gameContainer').offsetHeight * 0.8,
            sprite_map: EnemySpriteMap,
        }
    },
    map: {
        width: document.getElementById('gameContainer').offsetWidth,
        height: document.getElementById('gameContainer').offsetHeight,
        background: (() => {
            const img = new Image();
            img.src = '../Assets/Tiles/City1/Bright/City1.png';
            return img;
        })(),
    },

    game_config:{
        gameSpeed: 60,
        gravity: 9.8,
        jumpHeight: 15,
        attackPower: 10,
        attackSpeed: 5,
        blockPower: 5,
        blockSpeed: 5,
    },
    match_config:{
        timeLimit: 300,
        roundLimit: 3,
        scoreLimit: 10,
    },
    game_state:{
        isPaused: false,
        isGameOver: false,
        isRoundOver: false,
        currentRound: 1,
        player1Score: 0,
        player2Score: 0,
        gameOver: false,
        roundOver: false,
        player1Health: 100,
        player2Health: 100,
    },
})


export function loadAllImages() {
    const playerPromises = Object.entries(playerSpriteMap).map(async ([state, data]) => {
        const image = await data.image; // Resolve a Promise
        playerSpriteMap[state].image = image; // Substitui a Promise pela imagem carregada
    });

    const enemyPromises = Object.entries(EnemySpriteMap).map(async ([state, data]) => {
        const image = await data.image; // Resolve a Promise
        EnemySpriteMap[state].image = image; // Substitui a Promise pela imagem carregada
    });

    return Promise.all([...playerPromises, ...enemyPromises]);
}