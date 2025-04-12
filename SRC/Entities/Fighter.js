import { MovementHandler } from '../Movement/MovementHandler.js';
import { CollisionHandler } from '../Movement/CollisionHandler.js';
import { HealthBar } from '../UI/HealthBar.js';
export class Fighter {
    constructor(config) {
        this.x = config.x;
        this.y = config.y; 
        this.initialX = config.x; // Armazena a posição inicial
        this.initialY = config.y; // Armazena a posição inicial
        this.width = config.width;
        this.height = config.height;
        this.sprite = config.sprite;
        this.health = config.health;
        this.speed = config.speed;
        this.staggerFrame = config.staggerFrame || 20;
        this.velocity = { x: 0, y: 0 };
        this.sprite_Frame = 0;
        this.isAttacking = false;
        this.isBlocking = false;
        this.isJumping = false;
        this.animationState = 'idle';
        this.sprite_map = config.sprite_map;
        this.actualFrame = 0;

        this.attackBox = {
            x:config.attackBox.x,
            y:config.attackBox.y,
            width:config.attackBox.width,
            height:config.attackBox.height,
            offset:config.attackBox.offset,
        };


        this.healthBar = new HealthBar({
            x: this.x,
            y: 10, // Posiciona a barra acima do lutador
            width: 400,
            height: 30,
            maxHealth: config.health,
            currentHealth: config.health,
        });
        // Instâncias para controle de movimento e colisões
        // vou precisar colocar uma flag para saber se está em uma ação, seja correndo, atacando ou defendendo
        this.isActioning = false; // Flag para saber se o lutador está em uma ação (movendo, atacando, defendendo)
        this.colissionHandler = new CollisionHandler({xMin:0,xMax:config.map.width,yMin:0,yMax:config.map.height}, null); // null para o outro jogador, pois ainda não foi definido
        this.movement = new MovementHandler(this,this.colissionHandler);
        this.enemy = null; // Referência ao inimigo, se necessário
        this.gravity = config.gravity || 0.5; // Gravidade padrão
        this.ismovingLeft = false; // Flag para saber se o lutador está se movendo para a esquerda
        this.ismovingRight = false; // Flag para saber se o lutador está se movendo para a direita
        this.takingDamage = false; // Flag para saber se o lutador está levando dano

    }

    setEnemy(enemy) {
        this.enemy = enemy;
        this.colissionHandler.otherPlayer = enemy; // Atualiza o manipulador de colisões com o inimigo
    }

    // Métodos para movimentação
    moveLeft() {
        this.changeAnimationState('walk_left');
        this.ismovingLeft = true; // Define a flag de movimento para a esquerda
        this.ismovingRight = false; // Reseta a flag de movimento para a direita
        this.movement.moveLeft();
    }

    moveUp(){
        this.changeAnimationState('jump');
        this.movement.moveUp();
    }

    attack1(){
        this.changeAnimationState('attack1');

        this.isAttacking = true;
        
        //this.movement.attack1();
    }

    attack2(){
        this.changeAnimationState('attack2');
        this.isAttacking = true;
        
        
    }

    attack3(){
        this.changeAnimationState('attack3');
        this.isAttacking = true;
        
        
    }

    moveRight() {
        this.animationState = 'walk';
        this.ismovingRight = true; // Define a flag de movimento para a direita
        this.ismovingLeft = false; // Reseta a flag de movimento para a esquerda
        this.movement.moveRight();
    }

    stop() {
        this.movement.stop();
    }

    jump() {
        this.movement.jump();
    }

    
    // Atualizar o estado do lutador
    update(context) {
        this.render(context);
        

       

        if(!this.colissionHandler.isCollidingWithMap(this, this.movement.getspeedX())){
            this.x += this.speed.x; // Atualiza a posição do lutador com base na velocidade
        }
        else{
            this.speed.x = 0; // Para o movimento se colidir com o mapa
        }
        if(this.y + this.height + this.speed.y >= this.initialY + this.height){
            

            this.speed.y = 0; // Para o movimento se colidir com o mapa
            
        }
        else{
            this.speed.y += this.gravity;
        }
        this.y += this.speed.y; // Atualiza a posição do lutador com base na velocidade



        if(this.actualFrame === this.sprite_map[this.animationState].frames - 1) {
            if(this.isAttacking){
                if(this.attackBox.x + this.attackBox.width >= this.enemy.x &&
                    this.attackBox.x <= this.enemy.x + this.enemy.width &&
                    this.y <= this.enemy.y + this.enemy.height &&
                    this.y + this.height >= this.enemy.y
                    && this.isAttacking
                ){
                    //aqui o ataque pegou no inimigo

                    this.enemy.takeHit(this.calculateDamage(this.getAttackType()),this.getAttackType()); // Calcula o dano e aplica ao inimigo


                }

                    this.isAttacking = false; // Reseta o ataque após a colisão
                    if(!this.ismovingLeft && !this.ismovingRight){
                        this.changeAnimationState('idle'); // Muda o estado de animação para "idle"
                    }
                    else if(this.ismovingLeft){
                        this.changeAnimationState('walk_left'); // Muda o estado de animação para "walk"
                    }
                    else if(this.ismovingRight){
                        this.changeAnimationState('walk'); // Muda o estado de animação para "walk"
                    }

            }
            if(this.takingDamage){
                this.takingDamage = false; // Reseta a flag de dano após a animação
                this.changeAnimationState('idle'); // Muda o estado de animação para "idle"

            }

        }



        this.sprite_Frame++;

    }

    calculateDamage(type) {
        // Define o dano com base no tipo de ataque
        const damageValues = {
            1: 10, // Dano do ataque 1
            2: 20, // Dano do ataque 2
            3: 30, // Dano do ataque 3
        };
        return damageValues[type] || 0;
    }

    takeHit(amount,attackType) {
        console.log('tomei dano')
        
       if(this.isBlocking && this.correctDefense(attackType)) {
        return
       }
       console.log('aqui')
        this.takingDamage = true; // Marca que o lutador está levando dano
        this.changeAnimationState('hurt'); 
        
        this.isActioning = true; // Marca que uma ação está em andamento
        this.isAttacking = false; // Para o ataque se o lutador for atingido
        this.health -= amount;
        this.healthBar.update(this.health); // Atualiza a barra de saúde do inimigo
        if (this.health <= 0) {
            this.health = 0;
            document.getElementById('finalizaJogo').style.display = 'flex';
            this.die();
        }
    }



    block(type) {
        this.isBlocking = true;
        this.blockType = type; // Define o tipo de defesa
        this.animationState = `defend`;
    }

    stopBlock() {
        this.isBlocking = false;
        this.blockType = null; // Reseta o tipo de defesa
        this.animationState = 'idle';
    }

    getAttackType() {
        if (this.animationState === 'attack1') return 1;
        if (this.animationState === 'attack2') return 2;
        if (this.animationState === 'attack3') return 3;
        return 0; // Nenhum ataque
    }


    correctDefense(attackType) {
        // Define quais tipos de defesa bloqueiam quais ataques
        const defenseMap = {
            1: 2, // Defesa 1 bloqueia ataque 1
            2: 1, // Defesa 2 bloqueia ataque 2
            3: 3, // Defesa 3 bloqueia ataque 3
        };
        return defenseMap[attackType] === this.blockType;
    }
    
    render(context) {
        const initialSpriteCut = {x:0,y:0};
        const spriteWidth = this.sprite_map[this.animationState].frameWidth; // Largura do sprite
        const spriteHeight = this.sprite_map[this.animationState].frameHeight; // Altura do sprite
        this.attackBox.y = this.y + this.attackBox.offset.y; // Atualiza a posição Y da caixa de ataque com base na posição do lutador
        this.attackBox.x = this.x + this.attackBox.offset.x; // Atualiza a posição X da caixa de ataque com base na posição do lutador
        //agora vamos fazer a animacao da sprite
        //temos que calcular a posicao
        this.actualFrame = Math.floor(this.sprite_Frame/this.staggerFrame)%this.sprite_map[this.animationState].frames; // Posição X do corte do sprite
        context.drawImage(
            this.sprite_map[this.animationState].image, 
            spriteWidth * this.actualFrame, // Posição X do corte do sprite
            initialSpriteCut.y, 
            spriteWidth, 
            spriteHeight,
            this.x,
            this.y,
            this.width,
            this.height); // Desenha o lutador na tela
       
            context.strokeStyle = "white"; // Cor da borda
            context.lineWidth = 2;         // Espessura da borda (opcional)
            context.strokeRect(this.attackBox.x, this.attackBox.y, this.attackBox.width, this.attackBox.height);
        
            this.healthBar.render(context); // Renderiza a barra de saúde
        
    }


    changeAnimationState(newState) {
        console.log(newState)
        if( (this.isAttacking) // Se o lutador estiver atacando ou levando dano, não muda o estado de animação
            &&this.actualFrame < this.sprite_map[this.animationState].frames - 1) return; 

        if (this.takingDamage && this.animationState === 'hurt') {
            if (this.actualFrame < this.sprite_map[this.animationState].frames - 1) return; // Se o lutador estiver levando dano, não muda o estado de animação
           
        }
        if(this.animationState === newState) return; // Se o estado de animação for o mesmo, não faz nada
        this.animationState = newState;
        this.sprite_Frame = 0; // Reinicia o contador de frames da animação
        
    }
    
    

    // Morrer
    die() {
        if (this.constructor === Fighter) {
            throw new Error('Método "die" deve ser implementado nas subclasses!');
        }
    }
}
