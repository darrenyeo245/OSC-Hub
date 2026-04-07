import threading
import numpy as np
from pythonosc import osc_server, udp_client
from pythonosc.dispatcher import Dispatcher

IN_IP = "127.0.0.1"
IN_PORT = 9002
HUB_IP = "127.0.0.1"
HUB_PORT = 8000

STATE_OUT_ADDR = "/adm/obj/101/aed"
RESET_IN_ADDR = "/episode/reset"
STEP_TRIGGER_ADDRS = [
    "/reward",
    "/episode/end",
    "/episode/reset_manual",
    "/training/stop",
]


def clamp_state(values):
    s = np.array(values[:3], dtype=np.float32)
    s[0] = np.clip(s[0], -180.0, 180.0)  # azimuth (deg)
    s[1] = np.clip(s[1], -90.0, 90.0)    # elevation (deg)
    s[2] = np.clip(s[2], -1.0, 1.0)      # distance
    return s


state = clamp_state([0.0, 0.0, 0.0])
lock = threading.Lock()
client = udp_client.SimpleUDPClient(HUB_IP, HUB_PORT)

RNG = np.random.default_rng()

STEP_SCALE = 0.2
DRIFT_SCALE = 0.1


def publish_state():
    with lock:
        client.send_message(STATE_OUT_ADDR, state.tolist())


def reset_handler(address, *args):
    del address
    global state
    if len(args) < 3:
        return
    with lock:
        state = clamp_state(args)
    publish_state()


def step_handler(address, *args):
    del address, args
    with lock:
        # Random walk per step with light drift toward center (0,0,0).
        noise = RNG.normal(0.0, STEP_SCALE, size=3)
        drift = -state * DRIFT_SCALE
        state[:] = clamp_state(state + noise + drift)
    publish_state()


dispatcher = Dispatcher()
dispatcher.map(RESET_IN_ADDR, reset_handler)
for addr in STEP_TRIGGER_ADDRS:
    dispatcher.map(addr, step_handler)

server = osc_server.ThreadingOSCUDPServer((IN_IP, IN_PORT), dispatcher)
print(f"State simulator listening on {IN_IP}:{IN_PORT}")
publish_state()
server.serve_forever()
