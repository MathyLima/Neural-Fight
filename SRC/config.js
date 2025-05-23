function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.src = src;
    img.onload = () => {
      resolve(img);
    };
    img.onerror = (err) => reject(err);
  });
}

const playerSpriteMap = {
  idle: {
    image: loadImage("../Assets/Sprites/Shinobi/Idle.png"),
    frames: 6,
    frameWidth: 128,
    frameHeight: 128,
    staggerFrame: 200,
  },
  attack1: {
    image: loadImage("../Assets/Sprites/Shinobi/Attack_1.png"),
    frames: 5,
    frameWidth: 128,
    frameHeight: 128,
    staggerFrame: 130,
  },
  attack2: {
    image: loadImage("../Assets/Sprites/Shinobi/Attack_2.png"),
    frames: 3,
    frameWidth: 128,
    frameHeight: 128,
    staggerFrame: 170,
  },
  attack3: {
    image: loadImage("../Assets/Sprites/Shinobi/Attack_3.png"),
    frames: 4,
    frameWidth: 128,
    frameHeight: 128,
    staggerFrame: 200,
  },
  defend: {
    image: loadImage("../Assets/Sprites/Shinobi/Shield.png"),
    frames: 4,
    frameWidth: 128,
    frameHeight: 128,
    staggerFrame: 40,
  },
  jump: {
    image: loadImage("../Assets/Sprites/Shinobi/Jump.png"),
    frames: 12,
    frameWidth: 128,
    frameHeight: 128,
    staggerFrame: 30,
  },
  walk: {
    image: loadImage("../Assets/Sprites/Shinobi/Run.png"),
    frames: 8,
    frameWidth: 128,
    frameHeight: 128,
    staggerFrame: 100,
  },

  walk_left: {
    image: loadImage("../Assets/Sprites/Shinobi/Run_left.png"),
    frames: 8,
    frameWidth: 128,
    frameHeight: 128,
    staggerFrame: 100,
  },
  hurt: {
    image: loadImage("../Assets/Sprites/Shinobi/Hurt.png"),
    frames: 2,
    frameWidth: 128,
    frameHeight: 128,
    staggerFrame: 300,
  },
  die: {
    image: loadImage("../Assets/Sprites/Shinobi/Dead.png"),
    frames: 4,
    frameWidth: 128,
    frameHeight: 128,
    staggerFrame: 300,
  },
};

const EnemySpriteMap = {
  idle: {
    image: loadImage("../Assets/Sprites/Samurai/Idle_left.png"),
    frames: 6,
    frameWidth: 128,
    frameHeight: 128,
    staggerFrame: 200,
  },
  attack1: {
    image: loadImage("../Assets/Sprites/Samurai/Attack_1_left.png"),
    frames: 6,
    frameWidth: 128,
    frameHeight: 128,
    staggerFrame: 30,
  },
  attack2: {
    image: loadImage("../Assets/Sprites/Samurai/Attack_2_left.png"),
    frames: 4,
    frameWidth: 128,
    frameHeight: 128,
  },
  attack3: {
    image: loadImage("../Assets/Sprites/Samurai/Attack_3_left.png"),
    frames: 3,
    frameWidth: 128,
    frameHeight: 128,
  },
  defend: {
    image: loadImage("../Assets/Sprites/Samurai/Shield_left.png"),
    frames: 2,
    frameWidth: 128,
    frameHeight: 128,
  },
  jump: {
    image: loadImage("../Assets/Sprites/Samurai/Jump_left.png"),
    frames: 6,
    frameWidth: 128,
    frameHeight: 128,
  },
  walk: {
    image: loadImage("../Assets/Sprites/Samurai/Run.png"),
    frames: 8,
    frameWidth: 128,
    frameHeight: 128,
  },
  walk_left: {
    image: loadImage("../Assets/Sprites/Samurai/Run_left.png"),
    frames: 8,
    frameWidth: 128,
    frameHeight: 128,
  },
  hurt: {
    image: loadImage("../Assets/Sprites/Samurai/Hurt_left.png"),
    frames: 2,
    frameWidth: 128,
    frameHeight: 128,
    staggerFrame: 300,
  },
  die: {
    image: loadImage("../Assets/Sprites/Samurai/Dead.png"),
    frames: 3,
    frameWidth: 128,
    frameHeight: 128,
    staggerFrame: 300,
  },
};

export const initialConfig = Object.freeze({
  fighters: {
    player1: {
      x: document.getElementById("gameContainer").offsetWidth * 0.1,
      y: document.getElementById("gameContainer").offsetHeight * 0.1,
      sprite_map: playerSpriteMap,
    },
    enemy1: {
      x: document.getElementById("gameContainer").offsetWidth * 0.7,
      y: document.getElementById("gameContainer").offsetHeight * 0.1,
      sprite_map: EnemySpriteMap,
    },
  },
  map: {
    width: document.getElementById("gameContainer").offsetWidth,
    height: document.getElementById("gameContainer").offsetHeight,
    background: (() => {
      const img = new Image();
      img.src = "../Assets/Tiles/spring/6.png";
      return img;
    })(),
  },

  game_config: {
    gameSpeed: 60,
    gravity: 9.8,
    jumpHeight: 15,
    attackPower: 10,
    attackSpeed: 5,
    blockPower: 5,
    blockSpeed: 5,
  },
  match_config: {
    timeLimit: 300,
    roundLimit: 3,
    scoreLimit: 10,
  },
  game_state: {
    isPaused: false,
    isGameOver: false,
    isRoundOver: false,
    currentRound: 1,
    player1Score: 0,
    player2Score: 0,
    gameOver: false,
    roundOver: false,
    gameStarted: false,
    player1Health: 100,
    player2Health: 100,
  },
  server: {
    host: "localhost",
    porta: 8080,
  },
  inputMap: {
    q: {
      name: "Envenenar",
      cooldown: 2,
      currentCooldown: 0,
      duration: 4,
    },
    w: {
      name: "Enfraquecer",
      cooldown: 1,
      currentCooldown: 0,
      duration: 3,
    },
    e: {
      name: "Cura",
      cooldown: 5,
      currentCooldown: 0,
    },
    t: {
      name: "Sangramento",
      cooldown: 2,
      currentCooldown: 0,
      duration: 4,
    },
    r: {
      name: "Cura de efeitos",
      cooldown: 7,
      currentCooldown: 0,
      duration: 0,
    },
    a: {
      name: "Ataque Básico",
      cooldown: 0,
      currentCooldown: 0,
    },
    s: {
      name: "Ataque Pesado",
      cooldown: 2,
      currentCooldown: 0,
    },
  },
});

export function loadAllImages() {
  const playerPromises = Object.entries(playerSpriteMap).map(
    async ([state, data]) => {
      const image = await data.image;
      playerSpriteMap[state].image = image;
    }
  );

  const enemyPromises = Object.entries(EnemySpriteMap).map(
    async ([state, data]) => {
      const image = await data.image;
      EnemySpriteMap[state].image = image;
    }
  );

  return Promise.all([...playerPromises, ...enemyPromises]);
}
