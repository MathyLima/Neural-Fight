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
        
        
    }

    
    
       
    }

    


