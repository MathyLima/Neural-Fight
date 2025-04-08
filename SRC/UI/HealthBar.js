export class HealthBar {
    constructor(config) {
        this.x = config.x; // Posição X da barra
        this.y = config.y; // Posição Y da barra
        this.width = config.width || 100; // Largura total da barra
        this.height = config.height || 10; // Altura da barra
        this.maxHealth = config.maxHealth || 100; // Saúde máxima
        this.currentHealth = config.currentHealth || 100; // Saúde atual
        this.borderColor = config.borderColor || 'black'; // Cor da borda
        this.backgroundColor = config.backgroundColor || 'red'; // Cor do fundo
        this.foregroundColor = config.foregroundColor || 'green'; // Cor da barra de saúde
    }

    // Atualiza a saúde atual
    update(health) {
        this.currentHealth = Math.max(0, Math.min(health, this.maxHealth)); // Garante que a saúde esteja entre 0 e maxHealth
    }

    // Renderiza a barra de saúde no canvas
    render(context) {
        // Desenha o fundo da barra
        context.fillStyle = this.backgroundColor;
        context.fillRect(this.x, this.y, this.width, this.height);

        // Calcula a largura proporcional da barra de saúde
        const healthWidth = (this.currentHealth / this.maxHealth) * this.width;

        // Desenha a barra de saúde
        context.fillStyle = this.foregroundColor;
        context.fillRect(this.x, this.y, healthWidth, this.height);

        // Desenha a borda da barra
        context.strokeStyle = this.borderColor;
        context.lineWidth = 2;
        context.strokeRect(this.x, this.y, this.width, this.height);
    }
}