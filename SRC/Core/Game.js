export class Game {
    constructor(config){
        this.round = config.round; // Número da rodada
        this.fighters = config.fighters; // Lista de lutadores
        this.game_state = config.game_state; // Estado do jogo
        this.renderer = config.renderer; // Renderizador
        this.canvas = config.canvas; // Canvas do jogo
        this.context = config.context; // Contexto do canvas
        this.map = config.map; // Referência ao mapa
        this.audio = new Audio('../../Assets/Audios/Background.mp3'); // Substitua pelo caminho da sua música
        this.animationFrameId = null;
        this.animate = this.animate.bind(this);
        this.numberInputs;


        this.attackType = [];
        this.defenseType = [];
      
    }

     //vamos fazer o jogo funcionar por turnos de ataque e defesa
     //no inicio do jogo vamos fazer os jogadores irem para o centro
     startGame(){
        this.animate();
     }
     
     drawImage(image, x, y, width, height) {
        return function (context) {
            context.drawImage(image, x, y, width, height);
        };
    }

     drawBackgroundWithMessage() {
        // Renderiza o fundo
        this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.renderer.draw(this.drawImage(this.map.background, 0, 0, this.map.width, this.map.height));
    
        // Renderiza a mensagem
         this.context.fillStyle = 'white'; // Cor do texto
         this.context.font = '30px Arial'; // Fonte do texto
         this.context.textAlign = 'center'; // Alinhamento do texto
         this.context.fillText('Clique no jogo para iniciar', this.canvas.width / 2, this.canvas.height / 2);
    }




     

     inicioJogo(){
        this.drawBackgroundWithMessage();
        this.canvas.onclick = ()=>{
            this.game_state.gameStarted = true; // Atualiza o estado para iniciar o jogo
            this.audio.volume = 0.1; // Ajusta o volume da música
            this.audio.loop = true; // Define a música para tocar em loop
            this.audio.play().catch((error) => {
                console.error('Erro ao reproduzir a música:', error);
            });
        }
     }


     animate(){
        if(!this.game_state.gameStarted){
            this.inicioJogo();
        }else{
            this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
            this.renderer.draw(this.drawImage(this.map.background, 0, 0, this.map.width, this.map.height));
            
            this.fighters.forEach(fighter=>{
                this.renderer.draw((ctx)=>{
                    fighter.update(ctx);
                })
            })
            
            

            const allCentered = this.fighters.every(f => f.isCentered);

            if (allCentered) {
                const numberInteractions = 3;
                if (!this.numberInputs) this.numberInputs = numberInteractions;
            
                this.fighters.forEach(fighter => {
                    if (!fighter.inputing) {
                        fighter.numberInputs = this.numberInputs;
                        fighter.inputing = true;
                    }
                });
            
                const allPlayersReady = this.fighters.every(f => f.pressedKeys.length >= this.numberInputs);
                if ((allPlayersReady || this.attackType.length >= this.numberInputs || this.defenseType.length >= this.numberInputs) && !this.executandoRodada) {
                    this.executandoRodada = true;
                
                    this.fighters.forEach(fighter => {
                        const keys = fighter.pressedKeys.slice(0, this.numberInputs);
                
                        if (this.attackType.length === 0 && this.defenseType.length === 0) {
                            keys.forEach(key => {
                                if (fighter.turno === 'ataque') {
                                    if (key === 'j') this.attackType.push(1);
                                    else if (key === 'k') this.attackType.push(2);
                                    else if (key === 'l') this.attackType.push(3);
                                }
                
                                if (fighter.turno === 'defesa') {
                                    if (key === 'i') this.defenseType.push(1);
                                    else if (key === 'o') this.defenseType.push(2);
                                    else if (key === 'p') this.defenseType.push(3);
                                }
                            });

                            console.log(this.attackType)
                        }
                    });
                
                    this.correctDefense().then(async result => {
                        for (let i = 0; i < this.numberInputs; i++) {
                            for (const fighter of this.fighters) {
                                if (fighter.turno === 'ataque') {
                                    
                                    const attack = this.attackType[0]
                                    let attackType = 'attack1';
                                    if (attack === 2) attackType = 'attack2';
                                    else if (attack === 3) attackType = 'attack3';
                    
                                    fighter.isAttacking = true;
                                    fighter.changeAnimationState(attackType);
                                    console.log(fighter.animationState,attackType);
                                    
                                    await this.waitForAnimationEnd(fighter);
                                    this.attackType.shift();
                                    fighter.pressedKeys.shift();
                                    fighter.changeAnimationState('idle');
                                    fighter.isAttacking = false;

                                }
                    
                                if (fighter.turno === 'defesa') {
                                    const defense = this.defenseType.shift();
                                    fighter.pressedKeys.shift();
                    
                                    // espera o inimigo atacar
                                    if (fighter.enemy.isAttacking) {
                                        fighter.block(defense);
                                        await new Promise(r => setTimeout(r, 150)); // pequena pausa para o bloqueio
                                        fighter.stopBlock();
                                    }
                                }
                            }
                    
                            // pequena pausa entre cada input (se quiser)
                            await new Promise(r => setTimeout(r, 200));
                        }
                    
                        // Após os 3 inputs:
                        this.fighters.forEach(f => {
                            f.turno = f.turno === 'ataque' ? 'defesa' : 'ataque';
                            f.inputing = false;
                        });
                        this.numberInputs = null;
                        this.executandoRodada = false;
                    });
                }
            }

        }
        this.animationFrameId = requestAnimationFrame(this.animate);
        }

        waitForAnimationEnd(fighter) {
            return new Promise(resolve => {
                fighter.onAnimationEnd = resolve;
            });
        }
    correctDefense() {
        return new Promise((resolve)=>{
            
            const answers = []
            // Define quais tipos de defesa bloqueiam quais ataques
            const defenseMap = {
                1: 1, // Defesa 1 bloqueia ataque 1
                2: 2, // Defesa 2 bloqueia ataque 2
                3: 3, // Defesa 3 bloqueia ataque 3
            };
            this.attackType.forEach((attack,index)=>{
                answers.push(defenseMap[attack] === this.defenseType[index])    
                
            })
    
           
            resolve(answers)

        })
    }


     stopAnimation() {
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
    }



    resumeAnimation() {
        if (!this.animationFrameId) {
            this.animationFrameId = requestAnimationFrame(this.animate);
            console.log('Animação retomada');
        }
    }


}
