"""Habitat-Sim warehouse: build assets, sample episodes, render camera images, check aisles.

The warehouse (top view, camera at the dock looking toward the back wall):

    back wall    [bay 1] [bay 2] [bay 3] [bay 4]      crates sit in bays (3 crates, 1 empty bay)
                 ---------- back zone -----------
                 rack | left aisle | rack | right aisle | rack
                 ---------- front zone ----------
                              dock (robot, overview camera)

An episode samples a layout and a target crate, then one change:
  moved   : the target crate is moved to the empty bay
  blocked : a barrier is put across the aisle on the target bay's side (bays 1-2 left, 3-4 right)
For each episode we render  t0 (initial), t1_static (no change), t1_changed (after the change).
t1 images get a small camera jitter, so "no change" never means a pixel-identical image.
Which aisles are passable is decided by Habitat's navmesh path planner, not by hand.

Usage (env-hab, GPU node):  python sim.py --n 200 --out data
Writes data/episodes.jsonl and data/img/*.png
"""
import argparse, json, math, random
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

# Layout in meters, Habitat frame: y is up, the camera at the dock looks toward -z.
COLORS = {"red": (200, 30, 30), "blue": (30, 70, 200), "green": (30, 150, 50), "yellow": (255, 230, 40)}
Z_BACK, Z_FRONT = -11.0, 3.0           # back wall, open front edge; side walls at x = -7 and 7
RACK_X, RACK_Z, RACK_H = [(-7, -3), (-1, 1), (3, 7)], (-7.0, -3.0), 1.0
AISLE_X = {"left": -2.0, "right": 2.0}  # the two 2 m aisles between the racks
BAY_X, BAY_Z, CRATE = [-5.25, -1.75, 1.75, 5.25], -9.6, 1.0   # bays 1..4, left to right from the dock
BARRIER_Z = -5.0
CAM_POS, CAM_HEIGHT, CAM_PITCH_DEG, HFOV = (0.0, 0.0, -0.5), 7.0, -45.0, 75   # overview camera above the dock
W, H = 768, 512
FONT = "DejaVuSans-Bold.ttf"


# ---------- assets (trimesh -> glb) ----------

def font(size):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default(size=size)


def textured(verts, faces, uv, img):
    # glTF defaults to fully metallic, which renders almost black; say matte explicitly
    mat = trimesh.visual.material.PBRMaterial(baseColorTexture=img, metallicFactor=0.0, roughnessFactor=1.0)
    return trimesh.Trimesh(np.array(verts, float), np.array(faces), process=False,
                           visual=trimesh.visual.TextureVisuals(uv=np.array(uv, float), material=mat))


def quad(corners, img):
    """Quad from 4 corners (counter-clockwise seen from the visible side); image fills it."""
    return textured(corners, [[0, 1, 2], [0, 2, 3]], [[0, 0], [1, 0], [1, 1], [0, 1]], img)


def box(x0, x1, y0, y1, z0, z1, img):
    """Axis-aligned box as one mesh; every face shows the full image and points outward."""
    c = lambda x, y, z: (x, y, z)
    faces = [
        [c(x0, y0, z1), c(x1, y0, z1), c(x1, y1, z1), c(x0, y1, z1)],  # +z
        [c(x1, y0, z0), c(x0, y0, z0), c(x0, y1, z0), c(x1, y1, z0)],  # -z
        [c(x1, y0, z1), c(x1, y0, z0), c(x1, y1, z0), c(x1, y1, z1)],  # +x
        [c(x0, y0, z0), c(x0, y0, z1), c(x0, y1, z1), c(x0, y1, z0)],  # -x
        [c(x0, y1, z1), c(x1, y1, z1), c(x1, y1, z0), c(x0, y1, z0)],  # +y
        [c(x0, y0, z0), c(x1, y0, z0), c(x1, y0, z1), c(x0, y0, z1)],  # -y
    ]
    verts = [v for f in faces for v in f]
    tris = [[4 * k + i for i in t] for k in range(6) for t in ([0, 1, 2], [0, 2, 3])]
    return textured(verts, tris, [[0, 0], [1, 0], [1, 1], [0, 1]] * 6, img)


def floor_texture():
    """Concrete floor (50 px/m) with bay boxes and numbers, aisle center lines, and the dock."""
    ppm = 50
    img = Image.new("RGB", (14 * ppm, int(Z_FRONT - Z_BACK) * ppm), (150, 150, 145))
    d = ImageDraw.Draw(img)
    px = lambda x, z: ((x + 7) * ppm, (z - Z_BACK) * ppm)  # row 0 = back wall, so text reads upright from the dock
    for i, bx in enumerate(BAY_X):
        d.rectangle([px(bx - 1.0, BAY_Z - 1.0), px(bx + 1.0, BAY_Z + 1.0)], outline=(240, 200, 0), width=8)
        d.text(px(bx, BAY_Z + 1.5), str(i + 1), fill=(240, 200, 0), font=font(45), anchor="mm")
    for ax in AISLE_X.values():
        d.line([px(ax, RACK_Z[1]), px(ax, RACK_Z[0])], fill=(240, 200, 0), width=5)
    d.rectangle([px(-2.5, 0.5), px(2.5, 2.5)], outline=(255, 255, 255), width=6)  # dock
    return img


def sign_texture():
    """Back wall 14 m x 3 m with a numbered sign above each bay."""
    ppm = 50
    img = Image.new("RGB", (14 * ppm, 3 * ppm), (205, 205, 200))
    d = ImageDraw.Draw(img)
    for i, bx in enumerate(BAY_X):
        cx, cy = (bx + 7) * ppm, (3 - 2.0) * ppm
        d.rectangle([cx - 30, cy - 30, cx + 30, cy + 30], fill=(255, 255, 255), outline=(0, 0, 0), width=4)
        d.text((cx, cy), str(i + 1), fill=(0, 0, 0), font=font(48), anchor="mm")
    return img


def rack_texture():
    img = Image.new("RGB", (256, 128), (60, 60, 60))
    d = ImageDraw.Draw(img)
    for y in (10, 70):
        d.rectangle([0, y, 255, y + 8], fill=(230, 110, 20))       # orange beams
    for x in range(0, 256, 64):
        d.rectangle([x, 0, x + 8, 127], fill=(30, 60, 160))         # blue uprights
        d.rectangle([x + 14, 20, x + 58, 68], fill=(170, 130, 80))  # cardboard on the shelf
    return img


def solid(rgb, border=True):
    img = Image.new("RGB", (64, 64), rgb)
    if border:
        ImageDraw.Draw(img).rectangle([0, 0, 63, 63], outline=tuple(int(c * 0.55) for c in rgb), width=5)
    return img


def barrier_texture():
    img = Image.new("RGB", (256, 64), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for x in range(-64, 320, 48):
        d.polygon([(x, 64), (x + 24, 64), (x + 88, 0), (x + 64, 0)], fill=(240, 90, 0))
    return img


def build_assets(out):
    out.mkdir(parents=True, exist_ok=True)
    wall = solid((205, 205, 200), border=False)
    zb, zf = Z_BACK, Z_FRONT
    parts = [
        quad([(-7, 0, zf), (7, 0, zf), (7, 0, zb), (-7, 0, zb)], floor_texture()),
        quad([(-7, 0, zb), (7, 0, zb), (7, 3, zb), (-7, 3, zb)], sign_texture()),
        quad([(-7, 0, zf), (-7, 0, zb), (-7, 3, zb), (-7, 3, zf)], wall),   # left wall, faces +x
        quad([(7, 0, zb), (7, 0, zf), (7, 3, zf), (7, 3, zb)], wall),       # right wall, faces -x
    ]
    for x0, x1 in RACK_X:
        parts.append(box(x0, x1, 0, RACK_H, RACK_Z[0], RACK_Z[1], rack_texture()))
    # Habitat reads a bare stage .glb as z-up; rotate our y-up mesh so it lands upright.
    zup = trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0])
    trimesh.Scene([p.apply_transform(zup) for p in parts]).export(out / "warehouse.glb")
    h = CRATE / 2
    objs = {f"crate_{c}": box(-h, h, -h, h, -h, h, solid(rgb)) for c, rgb in COLORS.items()}
    objs["barrier"] = box(-1.0, 1.0, -0.6, 0.6, -0.15, 0.15, barrier_texture())  # spans the whole 2 m aisle
    for name, mesh in objs.items():
        trimesh.Scene(mesh).export(out / f"{name}.glb")
        (out / f"{name}.object_config.json").write_text(json.dumps({"render_asset": f"{name}.glb"}))


# ---------- episodes ----------

def sample_episode(i, rng):
    """Initial layout, target, and one change. Even ids: moved, odd ids: blocked (balanced)."""
    colors = rng.sample(list(COLORS), 3)
    bays = colors + [None]
    rng.shuffle(bays)
    target = rng.choice(colors)
    tbay = bays.index(target)
    init = {"bays": bays, "barrier": None}
    if i % 2 == 0:
        new = bays.copy()
        empty = new.index(None)
        new[empty], new[tbay] = target, None
        changed = {"bays": new, "barrier": None}
    else:
        changed = {"bays": bays, "barrier": "left" if tbay < 2 else "right"}
    return {"id": i, "target": target, "change": "moved" if i % 2 == 0 else "blocked",
            "states": {"static": init, "changed": changed}}


# ---------- habitat ----------

def make_sim(assets, gpu=0):
    import habitat_sim
    sc = habitat_sim.SimulatorConfiguration()
    sc.scene_id = str(assets / "warehouse.glb")
    sc.enable_physics = False
    sc.gpu_device_id = gpu
    cam = habitat_sim.CameraSensorSpec()
    cam.uuid, cam.sensor_type = "rgb", habitat_sim.SensorType.COLOR
    cam.resolution, cam.hfov = [H, W], HFOV
    cam.position, cam.orientation = [0.0, CAM_HEIGHT, 0.0], [math.radians(CAM_PITCH_DEG), 0.0, 0.0]
    ag = habitat_sim.agent.AgentConfiguration(sensor_specifications=[cam])
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sc, [ag]))
    # Light from above and from the camera side, so crate faces seen by the camera are not in shadow
    # (with the default lights the yellow crate's front face read as "brown").
    light = lambda v: habitat_sim.gfx.LightInfo(vector=[float(x) for x in v] + [0.0], color=[1.3, 1.3, 1.3],
                                                model=habitat_sim.gfx.LightPositionModel.Global)
    sim.set_light_setup([light(v) for v in ([0, -1, 0], [0, 1, 0], [0, -0.5, -1], [0, 0.5, 1])])
    otm = sim.get_object_template_manager()
    otm.load_configs(str(assets))
    handles = {Path(h).name.split(".")[0]: h for h in otm.get_template_handles()}
    return sim, handles


def set_state(sim, handles, state):
    """Remove all objects, then place crates and (optionally) the barrier as static objects."""
    import habitat_sim
    rom = sim.get_rigid_object_manager()
    rom.remove_all_objects()
    placed = [(f"crate_{c}", (BAY_X[b], CRATE / 2, BAY_Z)) for b, c in enumerate(state["bays"]) if c]
    if state["barrier"]:
        placed.append(("barrier", (AISLE_X[state["barrier"]], 0.6, BARRIER_Z)))
    for name, pos in placed:
        obj = rom.add_object_by_template_handle(handles[name])
        obj.translation = np.array(pos)  # move first: STATIC objects cannot be moved
        obj.motion_type = habitat_sim.physics.MotionType.STATIC
    ns = habitat_sim.NavMeshSettings()
    ns.set_defaults()
    ns.agent_radius, ns.include_static_objects = 0.35, True
    assert sim.recompute_navmesh(sim.pathfinder, ns)


def aisle_clear(sim, side):
    """Passable if the planner's path from aisle entrance to exit stays close to the straight run."""
    import habitat_sim
    x, (z_end, z_start) = AISLE_X[side], RACK_Z
    p = habitat_sim.ShortestPath()
    p.requested_start = sim.pathfinder.snap_point(np.array([x, 0.0, z_start + 1.0]))
    p.requested_end = sim.pathfinder.snap_point(np.array([x, 0.0, z_end - 1.0]))
    straight = (z_start - z_end) + 2.0
    return bool(sim.pathfinder.find_path(p)) and p.geodesic_distance < 1.3 * straight


def render(sim, path, jitter_rng=None):
    import habitat_sim
    from habitat_sim.utils.common import quat_from_angle_axis
    pos, yaw = np.array(CAM_POS), 0.0
    if jitter_rng is not None:  # small camera shake: +-10 cm, +-2 degrees yaw
        pos = pos + np.array([jitter_rng.uniform(-0.1, 0.1), 0.0, jitter_rng.uniform(-0.1, 0.1)])
        yaw = math.radians(jitter_rng.uniform(-2, 2))
    st = habitat_sim.AgentState()
    st.position, st.rotation = pos, quat_from_angle_axis(yaw, np.array([0.0, 1.0, 0.0]))
    sim.get_agent(0).set_state(st)
    Image.fromarray(sim.get_sensor_observations()["rgb"][..., :3]).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("data"))
    a = ap.parse_args()
    (a.out / "img").mkdir(parents=True, exist_ok=True)
    build_assets(a.out / "assets")
    sim, handles = make_sim((a.out / "assets").resolve())
    rng, jit = random.Random(a.seed), random.Random(a.seed + 1)
    with open(a.out / "episodes.jsonl", "w") as f:
        for i in range(a.n):
            ep = sample_episode(i, rng)
            ep["img"] = {}
            for key, state_name, jitter in [("t0", "static", None), ("t1_static", "static", jit),
                                            ("t1_changed", "changed", jit)]:
                state = ep["states"][state_name]
                set_state(sim, handles, state)
                state["clear"] = {s: aisle_clear(sim, s) for s in AISLE_X}
                # self-check: the planner must agree with the intended layout
                assert all(state["clear"][s] == (state["barrier"] != s) for s in AISLE_X), (i, state)
                ep["img"][key] = f"img/{i:04d}_{key}.png"
                render(sim, a.out / ep["img"][key], jitter)
            f.write(json.dumps(ep) + "\n")
    print(f"wrote {a.n} episodes to {a.out}")


if __name__ == "__main__":
    main()
