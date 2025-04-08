export class Renderer{
    static #instance
    constructor(canvas){
        if(Renderer.#instance){
            return Renderer.#instance
        }
        this.canvas = canvas;
        this.context = canvas.getContext('2d')
        Renderer.#instance = this
        this.canvas.width = this.canvas.offsetWidth
        this.canvas.height = this.canvas.offsetHeight
    }

    getInstance(){
        return Renderer.#instance
    };
    getCanvas(){
        return this.canvas
    };
    getContext(){
        return this.context
    };
    draw(callback){
        callback(this.context)
    };
}