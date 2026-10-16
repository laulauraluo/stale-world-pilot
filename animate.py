"""Render the episode-1 animation in Habitat: a memory-only agent vs the replay check.

Story (episode 1; the decisions are Qwen3-VL-8B's, read from results/qwen3vl8b.jsonl):
  1. 09:00: the robot's camera image and the notes the agent wrote from it (its turn-1 reply)
  2. 09:20: a barrier appears in the right aisle; task: bring the blue crate to the dock
  3. side by side, same 09:20 world:
     memory only : route planned on the 09:00 navmesh (no barrier); the robot follows it in the 09:20 world
                   and stops at the last point that is navigable there, in front of the barrier
     replay check: the plan contradicts the fresh facts; route re-planned on the 09:20 navmesh: to bay 4
                   (it goes through the left aisle by itself), load the crate, back to the dock
  4. end card with numbers copied from results/summary.csv
All routes come from Habitat's navmesh planner. In the experiment the agent only picks a bay and an aisle;
the moving robot here is a visualization of those choices, not a robot controller.

Usage (env-hab, GPU node):  python animate.py --data data --results results --out anim
Writes anim/animation.mp4 and anim/animation.gif.
"""
import argparse, json, math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import sim as S

EP = 1
FPS, SPEED = 15, 4.0                    # frames per second, robot speed in m/s
CAM = dict(pos=(0.0, 10.0, 3.5), pitch=-54.0, hfov=70.0)   # wide view that also shows the dock
DOCK, BAY4_FRONT = (0.0, 1.0), (S.BAY_X[3], S.BAY_Z + 1.2)
W, H, BAND = S.W, S.H, 96               # one panel is W x H; caption band on top
RED, GREEN, WHITE, GREY = (220, 50, 40), (40, 170, 70), (255, 255, 255), (30, 30, 30)
REGULAR = "DejaVuSans.ttf"


def font(size, bold=False):
    return S.font(size) if bold else S.ImageFont.truetype(REGULAR, size)


def project(p):
    """World point -> pixel in the wide camera (pinhole; the camera looks along -z, pitched down)."""
    cx, cy, cz = CAM["pos"]
    v = np.array(p, float) - [cx, cy, cz]
    a = math.radians(-CAM["pitch"])     # undo the pitch: rotate about x by -pitch
    y, z = v[1] * math.cos(a) - v[2] * math.sin(a), v[1] * math.sin(a) + v[2] * math.cos(a)
    f = (W / 2) / math.tan(math.radians(CAM["hfov"] / 2))
    return W / 2 + f * v[0] / -z, H / 2 - f * y / -z


def plan(sim, start, goal):
    import habitat_sim
    p = habitat_sim.ShortestPath()
    p.requested_start = sim.pathfinder.snap_point(np.array([start[0], 0.0, start[1]]))
    p.requested_end = sim.pathfinder.snap_point(np.array([goal[0], 0.0, goal[1]]))
    assert sim.pathfinder.find_path(p), (start, goal)
    return [np.array(q) for q in p.points]


def resample(points, step):
    """Points every `step` meters along a polyline."""
    out = [points[0]]
    for a, b in zip(points, points[1:]):
        n = max(1, int(np.linalg.norm(b - a) / step))
        out += [a + (b - a) * (i / n) for i in range(1, n + 1)]
    return out


def yaw_of(a, b):
    d = b - a
    return math.atan2(-d[0], -d[2]) if np.linalg.norm(d) > 1e-6 else None


def poses(points, step):
    """(position, yaw) per frame along a path; yaw follows the direction of travel."""
    pts = resample(points, step)
    out, yaw = [], 0.0
    for i, p in enumerate(pts):
        y = yaw_of(p, pts[min(i + 1, len(pts) - 1)])
        yaw = yaw if y is None else y
        out.append((p, yaw))
    return out


def text_block(draw, xy, lines, size, fill=WHITE, bold_first=False, gap=8):
    x, y = xy
    for i, line in enumerate(lines):
        f = font(size, bold=bold_first and i == 0)
        draw.text((x, y), line, fill=fill, font=f)
        y += size + gap


def canvas(caption):
    img = Image.new("RGB", (2 * W, H + BAND), GREY)
    ImageDraw.Draw(img).text((24, BAND // 2), caption, fill=WHITE, font=font(34, bold=True), anchor="lm")
    return img


def label(panel, text, color=WHITE):
    d = ImageDraw.Draw(panel)
    f = font(26, bold=True)
    w = d.textlength(text, font=f)
    d.rectangle([12, 12, 32 + w, 54], fill=(0, 0, 0))
    d.text((22, 33), text, fill=color, font=f, anchor="lm")
    return panel


def draw_route(panel, points, color, upto=None):
    d = ImageDraw.Draw(panel)
    pts = [project((p[0], 0.05, p[2])) for p in resample(points, 0.25)]
    d.line(pts if upto is None else pts[:upto], fill=color, width=6, joint="curve")
    return panel


def main():
    import habitat_sim, magnum as mn
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("anim"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    ep = [json.loads(l) for l in open(a.data / "episodes.jsonl")][EP]
    qwen = {r["proc"]: r for r in map(json.loads, open(a.results / "qwen3vl8b.jsonl"))
            if r["id"] == EP and r["world"] == "changed"}
    assert ep["change"] == "blocked" and ep["states"]["changed"]["barrier"] == "right" and ep["target"] == "blue"
    assert qwen["memory"]["final"]["aisle"] == "right" and qwen["replay"]["final"]["aisle"] == "left"
    notes = [l.strip(" -*") for l in qwen["memory"]["replies"][0].splitlines() if l.strip()]

    # Scene: experiment assets plus a robot; the wide camera replaces the experiment camera.
    assets = a.out / "assets"
    S.build_assets(assets)
    robot = S.box(-0.45, 0.45, -0.175, 0.175, -0.45, 0.45, S.solid((120, 60, 170)))
    S.trimesh.Scene(robot).export(assets / "robot.glb")
    (assets / "robot.object_config.json").write_text(json.dumps({"render_asset": "robot.glb"}))
    S.CAM_HEIGHT, S.CAM_PITCH_DEG, S.HFOV, S.CAM_POS = CAM["pos"][1], CAM["pitch"], CAM["hfov"], (0.0, 0.0, CAM["pos"][2])
    sim, handles = S.make_sim(assets.resolve())
    cam = habitat_sim.AgentState()
    cam.position = np.array([0.0, 0.0, CAM["pos"][2]])  # the sensor sits CAM_HEIGHT above this point
    sim.get_agent(0).set_state(cam)

    # Routes on the real navmesh. Stale route: 09:00 world. Replay route: 09:20 world.
    S.set_state(sim, handles, ep["states"]["static"])
    stale = plan(sim, DOCK, BAY4_FRONT)
    S.set_state(sim, handles, ep["states"]["changed"])
    out_leg, back_leg = plan(sim, DOCK, BAY4_FRONT), plan(sim, BAY4_FRONT, DOCK)
    mid = lambda path: [p for p in path if S.RACK_Z[0] < p[2] < S.RACK_Z[1]]  # points inside the aisles
    assert all(p[0] > 0 for p in mid(stale)) and all(p[0] < 0 for p in mid(out_leg) + mid(back_leg))
    step = SPEED / FPS
    stale_poses = poses(stale, step)
    blocked_at = next(i for i, (p, _) in enumerate(stale_poses) if not sim.pathfinder.is_navigable(p)) - 1
    stale_poses = stale_poses[:blocked_at + 1]  # stop at the last point navigable in the 09:20 world
    replay_poses = poses(out_leg, step) + poses(back_leg, step)
    pickup = len(poses(out_leg, step))

    rom = sim.get_rigid_object_manager()
    bot = rom.add_object_by_template_handle(handles["robot"])
    bot.motion_type = habitat_sim.physics.MotionType.KINEMATIC
    crate = rom.get_object_by_handle(rom.get_object_handles("crate_blue")[0])
    crate.motion_type = habitat_sim.physics.MotionType.KINEMATIC
    crate_home = np.array(crate.translation)

    def shot(pose, carry=False, show_robot=True):
        p, yaw = pose
        bot.translation = mn.Vector3(p[0], 0.175, p[2]) if show_robot else mn.Vector3(0, -5, 0)
        bot.rotation = mn.Quaternion.rotation(mn.Rad(yaw), mn.Vector3(0, 1, 0))
        crate.translation = mn.Vector3(p[0], 0.35 + S.CRATE / 2, p[2]) if carry else mn.Vector3(*crate_home)
        return Image.fromarray(sim.get_sensor_observations()["rgb"][..., :3])

    frames = []
    hold = lambda img, sec: frames.extend([img] * int(sec * FPS))
    start = (np.array([DOCK[0], 0.0, DOCK[1]]), 0.0)

    # 1. 09:00: what the robot's camera saw, and what the agent wrote down.
    img = canvas("09:00  The agent looks once and writes down the layout")
    img.paste(label(Image.open(a.data / ep["img"]["t0"]).convert("RGB"), "robot camera, 09:00"), (0, BAND))
    d = ImageDraw.Draw(img)
    text_block(d, (W + 40, BAND + 40), ["Agent's notes (Qwen3-VL-8B):"] + notes[:8], 30, bold_first=True, gap=14)
    hold(img, 3.5)

    # 2. 09:20: the right aisle gets blocked (the sim already holds the 09:20 world).
    wide = shot(start)
    img = canvas("09:20  The right aisle is blocked.  Task: bring the blue crate to the dock")
    img.paste(label(wide.copy(), "warehouse, 09:20"), (0, BAND))
    img.paste(label(Image.open(a.data / ep["img"]["t1_changed"]).convert("RGB"), "robot camera, 09:20"), (W, BAND))
    hold(img, 3.5)

    # 3. Side by side: memory only vs replay check.
    n = max(len(stale_poses), len(replay_poses)) + int(1.0 * FPS)
    for i in range(n):
        left = shot(stale_poses[min(i, len(stale_poses) - 1)])
        draw_route(left, stale, RED)
        if i >= len(stale_poses):
            x, y = project(stale_poses[-1][0] + [0, 1.2, -0.6])
            ImageDraw.Draw(left).text((x, y), "X", fill=RED, font=font(64, bold=True), anchor="mm")
        right = shot(replay_poses[min(i, len(replay_poses) - 1)], carry=i >= pickup)
        draw_route(right, out_leg + back_leg, GREEN)
        img = canvas("Same world, two agents")
        img.paste(label(left, "memory only: right aisle, blocked", RED), (0, BAND))
        img.paste(label(right, "replay check: left aisle, delivered", GREEN), (W, BAND))
        frames.append(img)

    # 4. End card: numbers copied from results/summary.csv (changed world, 200 episodes per cell).
    img = Image.new("RGB", (2 * W, H + BAND), GREY)
    d = ImageDraw.Draw(img)
    text_block(d, (60, 70), [
        "Three open VLMs, 200 episodes per condition (Habitat-Sim warehouse)",
        "Planning from 09:00 notes: acted on the old layout in 75% to 90.5% of changed scenes",
        "With the replay check, the same agents did so in 0% to 1.5%",
        "",
        "Preliminary simulation study. The agent only picks a bay and an aisle;",
        "the robot motion is a visualization of that choice on Habitat navmesh paths.",
    ], 32, bold_first=True, gap=22)
    hold(img, 3.0)

    import imageio
    # Main profile + index at the front (faststart): plays in browsers, phones and QuickTime while streaming.
    imageio.mimsave(a.out / "animation.mp4", [np.asarray(f) for f in frames], fps=FPS, codec="libx264",
                    quality=8, macro_block_size=16, pixelformat="yuv420p",
                    ffmpeg_params=["-profile:v", "main", "-movflags", "+faststart"])
    small = [f.resize((2 * W * 5 // 8, (H + BAND) * 5 // 8)) for f in frames[::2]]  # GIF: 7.5 fps, smaller
    small[0].save(a.out / "animation.gif", save_all=True, append_images=small[1:], duration=int(2000 / FPS), loop=0)
    print(f"{len(frames)} frames ({len(frames) / FPS:.1f} s); stale route stops after {blocked_at} steps;"
          f" replay route {len(replay_poses)} steps")


if __name__ == "__main__":
    main()
