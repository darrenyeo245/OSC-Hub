from pythonosc.dispatcher import Dispatcher
from pythonosc import osc_server, udp_client
from dotenv import load_dotenv
import os
load_dotenv()

RASPI_IP = os.getenv("RASPI_IP")
RASPI_PORT = int(os.getenv("RASPI_PORT", 9001))

clients = [
    (RASPI_IP, RASPI_PORT), #RL-System/OSC-Interface
    ("127.0.0.1", 9002), #State-Simulator
    ("127.0.0.1", 9004), #Visualizer
    ("127.0.0.1", 9005), #ADM-Tester
]

def broadcast_handler(address, *args):
    for ip, port in clients:
        client = udp_client.SimpleUDPClient(ip, port)
        client.send_message(address, list(args))


dispatcher = Dispatcher()
dispatcher.set_default_handler(broadcast_handler)

server = osc_server.ThreadingOSCUDPServer(("0.0.0.0", 8000), dispatcher)
print("Server started at 0.0.0.0:8000")
server.serve_forever()