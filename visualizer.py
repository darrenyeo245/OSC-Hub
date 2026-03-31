import argparse
import threading
import numpy as np

from vispy import app, scene
from vispy.scene import visuals
from pythonosc import dispatcher, osc_server

OSC_IP   = "127.0.0.1"
VISUALIZER_PORT = 9004

ACTOR_OSC_ADDR     = "/adm/obj/101/xyz"
SPOTLIGHT_OSC_ADDR = "/adm/obj/1/xyz"

SCALE = 1.0

X_MIN, X_MAX = -1.0, 1.0
Y_MIN, Y_MAX = -1.0, 1.0
Z_MIN, Z_MAX = 0.0, 1.0

STATIC_SPOTLIGHT_POS = np.array([1.0, 1.0, 1.0], dtype=np.float32)
GROUND_Z = 0.0


def extend_to_room_boundary(origin, target):
    direction = target - origin
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return target.copy()
    direction = direction / norm

    bounds = [
        (X_MIN, 0), (X_MAX, 0),
        (Y_MIN, 1), (Y_MAX, 1),
        (Z_MIN, 2), (Z_MAX, 2),
    ]

    t_min = np.inf
    for bound_val, axis in bounds:
        if abs(direction[axis]) > 1e-9:
            t = (bound_val - origin[axis]) / direction[axis]
            if t > 1e-6:
                t_min = min(t_min, t)

    if t_min == np.inf:
        return target.copy()

    return (origin + direction * t_min).astype(np.float32)

def clamp_xyz(values):
    return np.array(
        [
            np.clip(values[0], X_MIN, X_MAX),
            np.clip(values[1], Y_MIN, Y_MAX),
            np.clip(values[2], Z_MIN, Z_MAX),
        ],
        dtype=np.float32,
    )

class InitState:
    def __init__(self):
        self.lock          = threading.Lock()
        self.actor_pos     = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.aim_pos       = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.updated       = False

state = InitState()

def handle_actor(address, *args):
    if len(args) >= 3:
        with state.lock:
            state.actor_pos = clamp_xyz(np.array(args[:3], dtype=np.float32) * SCALE)
            state.updated = True

def handle_spotlight(address, *args):
    if len(args) >= 3:
        with state.lock:
            aim = clamp_xyz(np.array(args[:3], dtype=np.float32) * SCALE)
            state.aim_pos = aim
            state.updated = True

def start_osc_server():
    disp = dispatcher.Dispatcher()
    disp.map(ACTOR_OSC_ADDR,     handle_actor)
    disp.map(SPOTLIGHT_OSC_ADDR, handle_spotlight)
    server = osc_server.ThreadingOSCUDPServer((OSC_IP, OSC_PORT), disp)
    print(f"[OSC] Listening on {OSC_IP}:{OSC_PORT}")
    server.serve_forever()

def build_scene():
    canvas = scene.SceneCanvas(
        title="3D Visualizer",
        size=(900, 700),
        bgcolor="#13131f",
        show=True,
    )
    view = canvas.central_widget.add_view()

    view.camera = scene.cameras.TurntableCamera(
        elevation=30,
        azimuth=45,
        distance=3.0,
        fov=40,
    )
    view.camera.set_range(x=(X_MIN, X_MAX), y=(Y_MIN, Y_MAX), z=(Z_MIN, Z_MAX))

    visuals.XYZAxis(parent=view.scene)

    grid_pos = []
    for i in np.linspace(-1, 1, 11):
        grid_pos += [[i, -1, 0], [i,  1, 0]]
        grid_pos += [[-1, i, 0], [ 1, i, 0]]
    grid_pos = np.array(grid_pos, dtype=np.float32)
    visuals.Line(
        pos=grid_pos,
        color=(0.25, 0.25, 0.35, 1.0),
        connect="segments",
        parent=view.scene,
    )

    actor_marker = visuals.Markers(parent=view.scene)
    actor_marker.set_data(
        pos=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        face_color=(0.2, 0.6, 1.0, 1.0),
        edge_color=(1.0, 1.0, 1.0, 0.4),
        size=18,
        edge_width=1.5,
        symbol="disc",
    )

    spot_marker = visuals.Markers(parent=view.scene)
    spot_marker.set_data(
        pos=STATIC_SPOTLIGHT_POS.reshape(1, 3),
        face_color=(1.0, 0.8, 0.1, 1.0),
        edge_color=(1.0, 1.0, 1.0, 0.4),
        size=20,
        edge_width=1.5,
        symbol="star",
    )

    aim_marker = visuals.Markers(parent=view.scene)
    aim_marker.set_data(
        pos=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        face_color=(1.0, 0.25, 0.25, 1.0),
        edge_color=(1.0, 1.0, 1.0, 0.4),
        size=14,
        edge_width=1.2,
        symbol="disc",
    )

    _initial_end = extend_to_room_boundary(
        STATIC_SPOTLIGHT_POS, np.array([0.0, 0.0, 0.0], dtype=np.float32)
    )
    beam_line = visuals.Line(
        pos=np.array([STATIC_SPOTLIGHT_POS, _initial_end], dtype=np.float32),
        color=(0.6, 0.6, 0.6, 0.7),
        width=2.0,
        parent=view.scene,
    )

    visuals.Text(
        "Akteur",
        color=(0.2, 0.6, 1.0, 1.0),
        font_size=10,
        pos=(30, 20),
        parent=canvas.scene,
    )
    visuals.Text(
        "Scheinwerfer",
        color=(1.0, 0.8, 0.1, 1.0),
        font_size=10,
        pos=(60, 40),
        parent=canvas.scene,
    )
    visuals.Text(
        "Ausrichtung (XY-Ebene)",
        color=(1.0, 0.25, 0.25, 1.0),
        font_size=10,
        pos=(100, 60),
        parent=canvas.scene,
    )

    return canvas, actor_marker, spot_marker, aim_marker, beam_line

def run_visualizer():
    canvas, actor_marker, spot_marker, aim_marker, beam_line = build_scene()

    def on_timer(event):
        with state.lock:
            if not state.updated:
                return
            actor_pos     = state.actor_pos.copy()
            aim_pos       = state.aim_pos.copy()
            state.updated = False

        actor_marker.set_data(
            pos=actor_pos.reshape(1, 3),
            face_color=(0.2, 0.6, 1.0, 1.0),
            edge_color=(1.0, 1.0, 1.0, 0.4),
            size=18,
            edge_width=1.5,
            symbol="disc",
        )
        spot_marker.set_data(
            pos=STATIC_SPOTLIGHT_POS.reshape(1, 3),
            face_color=(1.0, 0.8, 0.1, 1.0),
            edge_color=(1.0, 1.0, 1.0, 0.4),
            size=20,
            edge_width=1.5,
            symbol="star",
        )
        aim_marker.set_data(
            pos=aim_pos.reshape(1, 3),
            face_color=(1.0, 0.25, 0.25, 1.0),
            edge_color=(1.0, 1.0, 1.0, 0.4),
            size=14,
            edge_width=1.2,
            symbol="disc",
        )
        beam_end = extend_to_room_boundary(STATIC_SPOTLIGHT_POS, aim_pos)
        beam_line.set_data(
            pos=np.array([STATIC_SPOTLIGHT_POS, beam_end], dtype=np.float32)
        )
        canvas.update()

    timer = app.Timer(interval=1/30, connect=on_timer, start=True)

    print("[VIS] Visualizer läuft. Fenster schließen zum Beenden.")
    app.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Visualizer für RL-ADM-OSC")
    parser.add_argument("--port",  type=int,   default=VISUALIZER_PORT,
                        help=f"Visualizer-Port (default: {VISUALIZER_PORT})")
    parser.add_argument("--scale", type=float, default=SCALE,
                        help="Skalierungsfaktor für ADM-Koordinaten (default: 1.0)")
    args = parser.parse_args()

    OSC_PORT = args.port
    SCALE    = args.scale

    osc_thread = threading.Thread(target=start_osc_server, daemon=True)
    osc_thread.start()

    run_visualizer()