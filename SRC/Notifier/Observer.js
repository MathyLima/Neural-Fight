export class Observer{
    constructor(){
        this.subcribers = {};
    }

    subscribe(event,callback){
        if(!this.subcribers[event]){
            this.subcribers[event]= [];
        }
        this.subcribers[event].push(callback);
    }

    unsubscribe(event,callback){
        if(this.subcribers[event]){
            this.subcribers[event] = this.subcribers[event].filter(cb=>cb!==callback);
        }
    }

    notify(event, data) {
        if (this.subcribers[event]) {
            this.subcribers[event].forEach((callback) => {
                callback(event,data);
            });
        }
    }
}