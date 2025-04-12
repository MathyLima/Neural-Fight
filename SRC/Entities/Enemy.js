import {Fighter} from './Fighter.js';
import {Input} from '../Core/Input.js';
import { MovementHandler } from '../Movement/MovementHandler.js';
import { CollisionHandler } from '../Movement/CollisionHandler.js';

export class Enemy extends Fighter {
    constructor(name,config) {
        super(config);
        this.name = name;
        this.input = new Input();
        this.colissionHandler = new CollisionHandler(this);
        this.movement = new MovementHandler(this,config.speed,this.colissionHandler);
        
        this.lastAttackTime = 0;
        this.attackCooldown = 1000; // em milissegundos (1 segundo, por exemplo)
    }

    updateAI() {
        const dx = this.attackBox.x - this.enemy.attackBox.x;
        const dy = this.attackBox.y - this.enemy.attackBox.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
    
        const horizontalDistance = Math.abs(this.attackBox.x - this.enemy.attackBox.x);
        const attackRange = this.attackBox.width; // Ou um valor fixo
        const isEnemyToRight = this.attackBox.x < this.enemy.attackBox.x;
    
        const now = Date.now();
    
        // 1. Está longe → aproxima
        if (distance > this.attackBox.width + this.enemy.attackBox.width) {
            if (this.attackBox.x > this.enemy.attackBox.x) {
                this.changeAnimationState('walk_left');
                this.movement.moveLeft();
            } else {
                this.changeAnimationState('walk');
                this.movement.moveRight();
            }
        }
    
        // 2. Está perto o suficiente → agir
        else {
            this.movement.stopHorizontal();
    
            // 2.1. Inimigo está no alcance de ataque?
            if (horizontalDistance <= attackRange) {
    
                // Verifica se o inimigo está atacando
                if (this.enemy.isAttacking) {
                    this.block(2); // Defende por 2 frames ou tempo custom
                    this.changeAnimationState('defend');
                }
    
                // Se não está atacando, IA pode atacar
                else if (!this.isAttacking && !this.takingDamage) {
                    this.attack1();
                    this.lastAttackTime = now;
                }
    
            } else {
                // Dentro da zona de combate, mas fora do alcance real
                this.changeAnimationState('idle');
            }
        }
    }
    
    
       
    }

    


