import argparse
import threading
import numpy as np

from vispy import app, scene
from vispy.color import Color
from vispy.scene import visuals
from pythonosc import dispatcher, osc_server

OSC_IP = "127.0.0.1"
VISUALIZER_PORT = 9004

ACTOR_OSC_ADDR = "/adm/obj/101/aed"
SPOTLIGHT_OSC_ADDR = "/adm/obj/1/aed"

SCALE = 1.0

SPHERE_RADIUS = 1.0
SPOTLIGHT_ORIGIN_XYZ = np.array([-1.0, 1.0, 1.0], dtype=np.float32)


def aed_to_xyz(az_deg, el_deg, distance):
    az = np.radians(az_deg)
    el = np.radians(el_deg)
    r = distance * SPHERE_RADIUS
    x = r * np.sin(az) * np.cos(el)
    y = r * np.cos(az) * np.cos(el)
    z = r * np.sin(el)
    return np.array([x, y, z], dtype=np.float32)


class InitState:
    def __init__(self):
        self.lock = threading.Lock()
        self.actor_aed = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.aim_aed = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.updated = False


state = InitState()


def aed_values(values):
    az = np.clip(float(values[0]), -180, 180)
    el = np.clip(float(values[1]), -90, 90)
    d = np.clip(float(values[2]) * SCALE, 0.0, 1.0)
    return np.array([az, el, d], dtype=np.float32)


def handle_actor(address, *values):
    print("ACTOR UPDATE:", values)
    if len(values) == 3:
        with state.lock:
            state.actor_aed = aed_values(values)
            state.updated = True


def handle_spotlight(address, *values):
    print("SPOTLIGHT UPDATE:", values)
    if len(values) == 3:
        with state.lock:
            state.aim_aed = aed_values(values)
            state.updated = True


def start_osc_server(port):
    disp = dispatcher.Dispatcher()
    disp.map(ACTOR_OSC_ADDR, handle_actor)
    disp.map(SPOTLIGHT_OSC_ADDR, handle_spotlight)
    server = osc_server.ThreadingOSCUDPServer((OSC_IP, port), disp)
    print(f"[OSC] Listening on {OSC_IP}:{port}")
    server.serve_forever()


def azimuth_circle(el_deg=0.0, radius=1.0, n_pts=180):
    pts = [aed_to_xyz(az, el_deg, radius) for az in np.linspace(-180, 180, n_pts, endpoint=False)]
    segs = []
    for i in range(len(pts)):
        segs += [pts[i], pts[(i + 1) % len(pts)]]
    return np.array(segs, dtype=np.float32)


def elevation_circle(az_deg=0.0, radius=1.0, n_pts=120):
    segs = []
    pts = [aed_to_xyz(az_deg, el, radius) for el in np.linspace(-180, 180, n_pts)]
    for i in range(len(pts) - 1):
        segs += [pts[i], pts[i + 1]]
    return np.array(segs, dtype=np.float32)


def build_scene():
    canvas = scene.SceneCanvas(
        title="RL-ADM AED Sphere Visualizer",
        size=(960, 720),
        bgcolor=Color("#0d0d1a"),
        show=True,
    )
    view = canvas.central_widget.add_view()
    view.camera = scene.cameras.TurntableCamera(
        elevation=25,
        azimuth=30,
        distance=3.2,
        fov=38,
    )

    lat_segs = azimuth_circle(el_deg=0.0, radius=1.0)
    visuals.Line(
        pos=lat_segs,
        color=(0.30, 0.30, 0.50, 1.0),
        connect="segments",
        width=2.0,
        parent=view.scene,
    )

    lon_segs = elevation_circle(az_deg=0.0, radius=1.0)
    visuals.Line(
        pos=lon_segs,
        color=(0.30, 0.30, 0.50, 1.0),
        connect="segments",
        width=2.0,
        parent=view.scene,
    )

    visuals.XYZAxis(parent=view.scene)

    front = aed_to_xyz(0, 0, 1.0)
    front_m = visuals.Markers(parent=view.scene)
    front_m.set_data(
        pos=front.reshape(1, 3),
        face_color=(0.5, 0.5, 0.8, 0.5),
        size=7,
        symbol="disc",
    )

    origin_m = visuals.Markers(parent=view.scene)
    origin_m.set_data(
        pos=SPOTLIGHT_ORIGIN_XYZ.reshape(1, 3),
        face_color=(1.0, 0.85, 0.2, 1.0),
        edge_color=(1.0, 1.0, 1.0, 0.5),
        size=12,
        edge_width=1.5,
        symbol="star",
    )

    init_end = state.aim_aed
    beam_line = visuals.Line(
        pos=np.array([SPOTLIGHT_ORIGIN_XYZ, init_end], dtype=np.float32),
        color=(1.0, 0.85, 0.25, 0.9),
        width=2.5,
        parent=view.scene,
    )

    aim_marker = visuals.Markers(parent=view.scene)
    aim_marker.set_data(
        pos=init_end.reshape(1, 3),
        face_color=(1.0, 0.3, 0.3, 1.0),
        edge_color=(1.0, 1.0, 1.0, 0.5),
        size=16,
        edge_width=1.5,
        symbol="disc",
    )

    init_actor = state.actor_aed
    actor_marker = visuals.Markers(parent=view.scene)
    actor_marker.set_data(
        pos=init_actor.reshape(1, 3),
        face_color=(0.3, 0.85, 0.3, 1.0),
        edge_color=(1.0, 1.0, 1.0, 0.5),
        size=22,
        edge_width=1.5,
        symbol="disc",
    )

    actor_line = visuals.Line(
        pos=np.array([[0.0, 0.0, 0.0], init_actor], dtype=np.float32),
        color=(0.3, 0.8, 0.3, 0.3),
        width=1.2,
        parent=view.scene,
    )

    legend_items = [
        ("Scheinwerfer", (1.0, 0.85, 0.2, 1.0)),
        ("Scheinwerfer-Ziel", (1.0, 0.3, 0.3, 1.0)),
        ("Akteur", (0.3, 0.85, 0.3, 1.0)),
    ]

    for i, (text, color) in enumerate(legend_items):
        visuals.Text(
            text,
            color=color,
            font_size=9,
            pos=(14, 20 + i * 20),
            anchor_x="left",
            parent=canvas.scene,
        )

    return canvas, actor_marker, actor_line, aim_marker, beam_line


def run_visualizer():
    canvas, actor_marker, actor_line, aim_marker, beam_line = build_scene()

    def on_timer(event):
        with state.lock:
            actor_aed = state.actor_aed.copy()
            aim_aed = state.aim_aed.copy()
            state.updated = False

        actor_xyz = aed_to_xyz(*actor_aed)
        aim_xyz = aed_to_xyz(*aim_aed)

        actor_marker.set_data(
            pos=actor_xyz.reshape(1, 3),
            face_color=(0.3, 0.85, 0.3, 1.0),
            edge_color=(1.0, 1.0, 1.0, 0.5),
            size=22,
            edge_width=1.5,
            symbol="disc",
        )

        actor_line.set_data(
            pos=np.array([[0.0, 0.0, 0.0], actor_xyz], dtype=np.float32)
        )

        aim_marker.set_data(
            pos=aim_xyz.reshape(1, 3),
            face_color=(1.0, 0.3, 0.3, 1.0),
            edge_color=(1.0, 1.0, 1.0, 0.5),
            size=16,
            edge_width=1.5,
            symbol="disc",
        )

        beam_line.set_data(
            pos=np.array([SPOTLIGHT_ORIGIN_XYZ, aim_xyz], dtype=np.float32)
        )

        canvas.update()

    app.Timer(interval=1 / 30, connect=on_timer, start=True)

    print("[VIS] Visualizer läuft. Fenster schließen zum Beenden.")
    app.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AED Sphere Visualizer")
    parser.add_argument("--port", type=int, default=VISUALIZER_PORT)
    parser.add_argument("--scale", type=float, default=SCALE)
    args = parser.parse_args()

    SCALE = args.scale

    osc_thread = threading.Thread(
        target=start_osc_server,
        args=(args.port,),
        daemon=True,
    )
    osc_thread.start()

    run_visualizer()