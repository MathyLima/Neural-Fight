import { MovementHandler } from '../Movement/MovementHandler.js';
import { CollisionHandler } from '../Movement/CollisionHandler.js';
import { HealthBar } from '../UI/HealthBar.js';
import { Game } from '../Core/Game.js';
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
        this.damageReduction = 0;
        this.roundKeys;

        this.poisonDuration = 0;
        this.enfraquecerDuration = 0;
        this.sangramentoDuration = 0;
        
        this.currentEffect = [];


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
        this.turno = config.turno;
        this.isCentered = false;
        this.inputing = false;
        this.numberInputs;
        this.pressedKeys = []; // Armazenará as teclas pressionadas permitidas
        this.isAnimating = false;
        this.onAnimationEnd = null;
        this.movingToInitialPosition = false;
        this.attacks = config.inputMap;
        this.inputSession = false;
    }


    inputTime(){
        return
    }
    setEnemy(enemy) {
        this.enemy = enemy;
        this.colissionHandler.otherPlayer = enemy; // Atualiza o manipulador de colisões com o inimigo
    }
    setRoundKeys(keysMap){
        console.log(keysMap)
        this.roundKeys =keysMap
        
    }

    // Métodos para movimentação
    moveLeft() {
        this.changeAnimationState('walk_left');
        this.ismovingLeft = true; // Define a flag de movimento para a esquerda
        this.ismovingRight = false; // Reseta a flag de movimento para a direita
        this.movement.moveLeft();
    }

    getStatusEffects() {
        const efeitos = ['veneno', 'sangramento', 'enfraquecer'];
        const resumo = [];
    
        for (let efeito of efeitos) {
            const quantidade = this.contarEfeito(this.currentEffect, efeito);
            let duracao = 0;
    
            switch (efeito) {
                case 'veneno':
                    duracao = this.poisonDuration;
                    break;
                case 'sangramento':
                    duracao = this.sangramentoDuration;
                    break;
                case 'enfraquecer':
                    duracao = this.enfraquecerDuration;
                    break;
            }
    
            if (quantidade > 0) {
                resumo.push({
                    tipo: efeito,
                    quantidade,
                    duracao
                });
            }
        }
    
        return resumo;
    }

    moveUp(){
        this.changeAnimationState('jump');
        this.movement.moveUp();
    }

  

    moveRight() {
        this.animationState = 'walk';
        this.ismovingRight = true; // Define a flag de movimento para a direita
        this.ismovingLeft = false; // Reseta a flag de movimento para a esquerda
        this.movement.moveRight();
    }

    update(context){
        this.render(context);
        if(this.movingToInitialPosition){
            this.movement.moveToInitialPosition()
        }else{
            
            if(!this.isCentered){
                this.movement.moveToCenter();
            }
           
        }
        if(!this.colissionHandler.isCollidingWithMap(this, this.movement.getspeedX())){
            this.x += this.speed.x; // Atualiza a posição do lutador com base na velocidade
        }
        else{
            this.speed.x = 0; // Para o movimento se colidir com o mapa
        }
       


        if (this.isAttacking && this.actualFrame === this.sprite_map[this.animationState].frames - 1) {
            this.isAttacking = false;
            this.changeAnimationState('idle');
            if (typeof this.onAnimationEnd === 'function') {
                this.onAnimationEnd(); // Chama um callback se tiver
                this.onAnimationEnd = null;
            }
        }
        

        this.sprite_Frame++;
    }


    envenenar(){
        this.currentEffect.push('veneno');
        this.poisonDuration = 4;
    }


    enfraquecer(){
        if(this.damageReduction< 50){
            this.damageReduction+=10;
        }
        this.enfraquecerDuration = 3;
        if(!this.currentEffect.includes('enfraquecer')){
            this.currentEffect.push('enfraquecer');
        }
    }


    sangramento(){
            this.currentEffect.push('sangramento');
            
            this.sangramentoDuration = 3;
    }

    curarEfeitos(){
        this.currentEffect = [];
        this.damageReduction = 0;
        this.sangramentoDuration = 0;
        this.poisonDuration = 0;
    }


    curar(){
        if(this.health+10 >= 100){
            this.health = 100;
            return
        }

        this.health+=10;
    }


    block(type) {
        this.isBlocking = true;
        this.blockType = type; // Define o tipo de defesa
        this.changeAnimationState('defend');
    }

    stopBlock() {
        this.isBlocking = false;
        this.blockType = null; // Reseta o tipo de defesa
        this.changeAnimationState('idle')
    }

    takeDamage(){
        
        this.takingDamage = true;
        this.changeAnimationState('hurt');
        
        
    }
    
        stopTakeDamage(danoBase) {
            this.takingDamage = false;
        
            let modificadorEnfraquecido = this.enemy.damageReduction;
        
            let danoFinal = danoBase;
        
            if (modificadorEnfraquecido > 0) {
                danoFinal = danoBase * ((100 - modificadorEnfraquecido) / 100);
            }
        
           
        
        
            this.atualizarVida(danoFinal)
            this.changeAnimationState('idle')
    }


    atualizarVida(danoFinal){
        this.health -= danoFinal;
    
        this.healthBar.update(this.health); // Atualiza a barra de saúde do inimigo
    

        if (this.health <= 0) {
            this.health = 0;
            document.getElementById('finalizaJogo').style.display = 'flex';
            this.die();
            const gameInstance = Game.getInstance();
            gameInstance.gameEnded = true;
        }
        
    }

    contarEfeito(lista, efeito) {
        return lista.filter(e => e === efeito).length;
    }
    
    decrementMinZero(value) {
        return Math.max(0, value - 1);
    }

    takeDamageFinal() {
        let dano = 0;
    
        const qtdVeneno = this.contarEfeito(this.currentEffect, 'veneno');
        const qtdSangramento = this.contarEfeito(this.currentEffect, 'sangramento');
        if (this.poisonDuration > 0) {
            const danoVeneno = 2 * qtdVeneno;
            dano += danoVeneno;
        }
    
        if (this.sangramentoDuration > 0) {
            const danoSangramento = 3 * qtdSangramento;
            dano += danoSangramento;
        }
    
        this.sangramentoDuration = this.decrementMinZero(this.sangramentoDuration);
        this.poisonDuration = this.decrementMinZero(this.poisonDuration);

        if (this.poisonDuration === 0) {
            this.removerEfeito('veneno');
        }
        
        if (this.sangramentoDuration === 0) {
            this.removerEfeito('sangramento');
        }
    
        this.atualizarVida(dano);
    }


    removerEfeito(tipo) {
        const index = this.currentEffect.indexOf(tipo);
        if (index !== -1) {
            this.currentEffect.splice(index, 1);
        }
    }

    comecarCooldownTecla(tecla){
        this.attacks[tecla].currentCooldown = this.attacks[tecla].cooldown; 

        console.log(this.attacks[tecla])
    }

    
    render(context) {
        const initialSpriteCut = {x:0,y:0};
        const spriteWidth = this.sprite_map[this.animationState].frameWidth; // Largura do sprite
        const spriteHeight = this.sprite_map[this.animationState].frameHeight; // Altura do sprite
        this.attackBox.y = this.y + this.attackBox.offset.y; // Atualiza a posição Y da caixa de ataque com base na posição do lutador
        this.attackBox.x = this.x + this.attackBox.offset.x; // Atualiza a posição X da caixa de ataque com base na posição do lutador
        //agora vamos fazer a animacao da sprite
        //temos que calcular a posicao
        let frameAnterior = this.actualFrame
        this.actualFrame = Math.floor(this.sprite_Frame/this.staggerFrame)%this.sprite_map[this.animationState].frames; // Posição X do corte do sprite
        if((this.isBlocking  || this.takingDamage)&& this.actualFrame === this.sprite_map[this.animationState].frames - 1){
            
            this.actualFrame = frameAnterior
            if (typeof this.onAnimationEnd === 'function') {
                this.onAnimationEnd(); // Chama um callback se tiver
                this.onAnimationEnd = null;
            }
        }

        
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
        if(this.animationState === 'die') return;
        const currentIsAttack = ['attack1', 'attack2', 'attack3'].includes(this.animationState);
    
        if (this.isAttacking && currentIsAttack && this.actualFrame < this.sprite_map[this.animationState].frames - 1) {
            // Está no meio de uma animação de ataque, e quer trocar pra outro ataque → bloqueia
            return ;
        }
    
        if (this.animationState === newState) return; // já tá no estado desejado
    
        this.animationState = newState;
        this.sprite_Frame = 0;
        return ;
    }
    
    

    // Morrer
    die() {
        //this.changeAnimationState('die')
    }
}
