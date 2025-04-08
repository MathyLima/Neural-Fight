import asyncio
import websockets


async def handler(websocket):
    print('cliente conectado')
    try:
        async for message in websocket:
            print(f"Mensagem recebida: {message}")
            await websocket.send(f"Mensagem recebida: {message}")
    except websockets.ConnectionClosed:
        print('Cliente desconectado')



async def main():
    async with websockets.serve(handler, "localhost", 8080):
        print("Servidor WebSocket rodando em ws://localhost:8080")
        await asyncio.Future()  # executa para sempre

asyncio.run(main())
