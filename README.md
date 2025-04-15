# Neural-Fight


# 🥋 Jogo de Luta por Turnos (JavaScript)

Este é um jogo de luta baseado em turnos implementado em JavaScript com uso de sprites animados e entrada por teclado. O jogo permite dois lutadores se enfrentarem alternando entre ataques e defesas com base em comandos do jogador. O jogo é feito para 1 jogador, o lutador adversário é operado por uma LSTM.

---

## 🎮 Como Funciona

### Tela Inicial
- O jogo inicia com uma tela de fundo e a mensagem:  
  **"Clique no jogo para iniciar"**
- Ao clicar, a música de fundo começa a tocar em loop e o jogo é iniciado.

---

## 🚀 Início do Jogo
- Os dois lutadores caminham automaticamente até o centro da arena.
- Uma vez centralizados, o jogo entra no modo **entrada de comandos**.

---

## 🎯 Sistema de Turnos

- O jogo funciona com base em turnos alternados:
  - Um jogador é **atacante**
  - O outro é **defensor**
- Cada jogador insere **uma quantidade determinada de inputs sequenciais** por turno, essa quantidade pode variar entre 2 a 4.

### Comandos de Ataque
| Tecla | Tipo de Ataque |
|-------|----------------|
| `J`   | Ataque 1       |
| `K`   | Ataque 2       |
| `L`   | Ataque 3       |

### Comandos de Defesa
| Tecla | Tipo de Defesa |
|-------|----------------|
| `I`   | Defesa 1       |
| `O`   | Defesa 2       |
| `P`   | Defesa 3       |

---

## ⚔️ Execução da Rodada

1. O jogador insere seus respectivos comandos.
2. A IA tenta prever quais são esses comandos e fazer uma contra jogada.
3. O sistema compara os ataques e defesas:
   - Defesa correta bloqueia o ataque correspondente.
   - O dano máximo por rodada é 10%, esse valor é diminuido com o tanto de defesas realizadas com sucesso.
4. Para cada input:
   - O atacante realiza uma animação (`attack1`, `attack2`, `attack3`).
   - O defensor tenta bloquear se a defesa coincidir.
5. Cada ação é executada sequencialmente com **espera da animação finalizar**.
6. Jogadores voltam para a posição inicial e aguarda o início da próxima jogada.
---

## 🔁 Revezamento de Turnos

- Após os 3 comandos, os papéis se invertem:
  - O atacante vira defensor
  - O defensor vira atacante
- O processo de input e ataque/defesa se repete.

---

## 🖼️ Renderização

- O jogo usa um sistema de `Renderer` para desenhar:
  - Fundo do mapa
  - Sprites dos personagens
  - Animações de ataque, defesa e idle

---

## 🔊 Áudio

- A música de fundo é carregada e reproduzida após o clique inicial.
- O volume é ajustado para 10%.
- A música toca em loop durante o jogo.

---

## ✅ Estados Internos

- `gameStarted`: controla se o jogo já começou
- `executandoRodada`: impede múltiplas execuções simultâneas
- `isCentered`: indica se o personagem chegou ao centro
- `pressedKeys`: armazena os inputs do jogador
- `turno`: "ataque" ou "defesa"

---

## 📦 Estrutura Básica
- O jogo carrega suas configurações através do arquivo config.js, lá são descritas as configurações iniciais e de sprites.
```js
game = new Game({
  round: 0,
  fighters: [player1, player2],
  game_state: { gameStarted: false },
  renderer,
  canvas,
  context,
  map
});

game.startGame();