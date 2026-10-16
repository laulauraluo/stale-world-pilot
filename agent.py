"""Run a VLM agent on the warehouse episodes: 4 procedures x 2 worlds per episode.

Every episode is a short chat with the agent:
  turn 1  camera image at 09:00 -> the agent describes the floor (this is its memory)
  turn 2  "It is now 09:20 ... bring the <color> crate" -> the agent picks a bay (1-4) and an aisle

Procedures (how turn 2 runs):
  memory  : no new image; the agent may answer {"action": "look"} once to get the current image
  recheck : the current image is attached and the agent is told to check what changed
  replay  : the agent plans from memory; then the facts the plan depends on (which bay holds the
            target crate, whether each aisle is clear) are asked again on the current image in a
            fresh chat, and code compares them with the plan. Only if a fact contradicts the plan
            does the agent get a note and decide again. This ports Snapshot Replay: re-run the same
            query on the current state and let code, not the model, decide whether a change matters.
  facts   : ablation of replay. Same plan and same fresh-chat facts (copied from the replay case), but
            the agent always sees all the facts and the current image and compares them itself; code
            compares nothing. recheck -> facts isolates asking the facts separately; facts -> replay
            isolates the code comparison.
Reasoning models (GLM-4.1V-Thinking) write <think>...</think> first; that part is stripped before the
reply goes back into the chat or the log.
Worlds: static (nothing changes except slight camera jitter) and changed (crate moved / aisle blocked).

Usage (Python with vllm, one GPU):  python agent.py --model qwen3vl8b --data data --out results [--limit 10]
Writes <out>/<model>.jsonl: one line per (episode, world, procedure) with the final action and every
raw reply. Scoring is in analyze.py, so results can be re-scored without a GPU.
"""
import argparse, base64, json, re
from pathlib import Path

MODELS = {
    "qwen3vl8b": dict(model="Qwen/Qwen3-VL-8B-Instruct"),
    "mistral24b": dict(model="RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-FP8",
                       tokenizer_mode="mistral", config_format="mistral", load_format="mistral"),
    "glm41v9b": dict(model="zai-org/GLM-4.1V-9B-Thinking", max_tokens=8192, max_model_len=16384),  # GLM card: 8192
}
WORLDS, PROCS = ("static", "changed"), ("memory", "recheck", "replay")  # facts cases are copied from replay

SYSTEM = ("You control a delivery robot in a warehouse. A camera above the loading dock shows the floor. "
          "Four storage bays along the back wall are numbered 1 to 4 from left to right (signs on the wall, "
          "numbers on the floor). Crates sit in the bays. Two aisles, the left aisle and the right aisle, lead "
          "from the dock to the bays between shelving racks. The robot can reach any bay through either aisle, "
          "but it cannot drive through a blocked aisle. Other robots and workers also work on this floor.")
DESCRIBE = ("Camera image at 09:00. Describe the floor so you can plan later: for each bay 1-4, the color of "
            "the crate in it or 'empty', and whether each aisle is clear or blocked.")
# Neutral time cue. A smoke test that said "others have been moving things" made Qwen ask to look 10/10 times,
# which tests hint-following, not whether the agent decides on its own to re-check.
TASK = "It is now 09:20. New task: bring the COLOR crate to the dock."
GO = ('{"action": "go", "bay": <1-4>, "aisle": "left" or "right", "world_changed": true or false}, where '
      'world_changed says whether you think the floor changed since 09:00 in a way that matters for this task.')
REPLY = " Reply with one JSON object and nothing else: "
P_MEMORY = TASK + " You can act now, or first ask for a new camera image." + REPLY + 'either {"action": "look"} or ' + GO
P_LOOKED = "Camera image at 09:20. Now act." + REPLY + GO
P_RECHECK = (TASK + " The attached image is the current camera image (09:20). Before acting, check whether "
             "anything changed since 09:00 that matters for this task." + REPLY + GO)
P_PLAN = TASK + " Plan from what you saw at 09:00." + REPLY + '{"bay": <1-4>, "aisle": "left" or "right"}'
P_FACTS = ("Camera image." + REPLY + '{"bay": <the bay (1-4) holding the COLOR crate, or null if there is none>, '
           '"left_aisle_clear": true or false, "right_aisle_clear": true or false}')
P_NOTE = ("Before the robot moves, a re-check on the current camera image (09:20, attached) found that ISSUES. "
          "Your plan was bay BAY via the AISLE aisle. Decide again." + REPLY + GO)
P_ALL_FACTS = ("Before the robot moves, a separate check of the current camera image (09:20, attached) reports: "
               "FACTS. Your plan was bay BAY via the AISLE aisle. Check whether anything changed since 09:00 "
               "that matters for this task, then decide." + REPLY + GO)


def answer(text):
    """Drop a reasoning model's <think>...</think> part and GLM's answer/box markers; others pass through."""
    text = re.sub(r"<think>.*?(</think>|$)", "", text, flags=re.S)
    return re.sub(r"</?answer>|<\|(begin|end)_of_box\|>", "", text).strip()


def describe_facts(facts, color):
    """The fresh-chat facts as a sentence, for the facts ablation."""
    where = f"bay {facts['bay']}" if facts.get("bay") in (1, 2, 3, 4) else "no bay"
    state = lambda side: {True: "clear", False: "blocked"}.get(facts.get(f"{side}_aisle_clear"), "unknown")
    return f"the {color} crate is in {where}, the left aisle is {state('left')}, the right aisle is {state('right')}"


def parse(text):
    """The last {...} in a reply as a dict, or {} if none parses."""
    for m in reversed(re.findall(r"\{[^{}]*\}", text)):
        try:
            return json.loads(m)
        except json.JSONDecodeError:
            pass
    return {}


def plan_issues(plan, facts, color):
    """Facts from the current image that contradict the plan (empty list: the plan still holds)."""
    issues = []
    if facts.get("bay") != plan.get("bay"):
        where = f"bay {facts['bay']}" if facts.get("bay") else "no bay"
        issues.append(f"the {color} crate is in {where}")
    if facts.get(f"{plan.get('aisle')}_aisle_clear") is not True:
        issues.append(f"the {plan.get('aisle')} aisle is blocked" if plan.get("aisle") in ("left", "right")
                      else "your plan names no valid aisle")
    return issues


def final_action(d):
    bay, aisle, changed = d.get("bay"), d.get("aisle"), d.get("world_changed")
    return {"bay": bay if bay in (1, 2, 3, 4) else None,
            "aisle": aisle if aisle in ("left", "right") else None,
            "world_changed": changed if isinstance(changed, bool) else None}


def image(path):
    url = "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": url}}


def user(text, img=None):
    return {"role": "user", "content": ([image(img)] if img else []) + [{"type": "text", "text": text}]}


def assistant(text):
    return {"role": "assistant", "content": text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    eps = [json.loads(l) for l in open(a.data / "episodes.jsonl")][: a.limit]
    img = lambda ep, key: a.data / ep["img"][key]

    from vllm import LLM, SamplingParams
    cfg = {"max_tokens": 512, "max_model_len": 8192, **MODELS[a.model]}
    greedy = SamplingParams(temperature=0.0, max_tokens=cfg.pop("max_tokens"))
    llm = LLM(**cfg, limit_mm_per_prompt={"image": 2}, gpu_memory_utilization=0.85, seed=0)
    chat = lambda convs: [answer(o.outputs[0].text) for o in llm.chat(convs, greedy, use_tqdm=True)] if convs else []
    system = {"role": "system", "content": SYSTEM}

    # Turn 1: the memory, shared by every world and procedure of an episode.
    t1 = [[system, user(DESCRIBE, img(ep, "t0"))] for ep in eps]
    memory = {ep["id"]: conv + [assistant(r)] for ep, conv, r in zip(eps, t1, chat(t1))}

    cases = [{"ep": ep, "world": w, "proc": p, "looked": False, "replay_fired": None}
             for ep in eps for w in WORLDS for p in PROCS]
    for c in cases:
        ep, now = c["ep"], img(c["ep"], f"t1_{c['world']}")
        prompt = {"memory": P_MEMORY, "recheck": P_RECHECK, "replay": P_PLAN}[c["proc"]]
        c["now"] = now
        c["conv"] = memory[ep["id"]] + [user(prompt.replace("COLOR", ep["target"]),
                                             now if c["proc"] == "recheck" else None)]

    # Turn 2 for everyone; replay also asks the plan's facts on the current image in a fresh chat.
    for c, r in zip(cases, chat([c["conv"] for c in cases])):
        c["conv"].append(assistant(r))
    replay = [c for c in cases if c["proc"] == "replay"]
    fact_convs = [[system, user(P_FACTS.replace("COLOR", c["ep"]["target"]), c["now"])] for c in replay]
    for c, r in zip(replay, chat(fact_convs)):
        c["facts_reply"] = r
    # Ablation: same plan and facts as replay, but the model always sees the facts and compares itself.
    ablation = [{**c, "proc": "facts", "conv": list(c["conv"])} for c in replay]
    cases += ablation

    # Turn 3 only where needed: memory agents that asked to look, replay plans that the facts contradict.
    turn3 = []
    for c in cases:
        reply = parse(c["conv"][-1]["content"])
        if c["proc"] == "memory" and reply.get("action") == "look":
            c["looked"] = True
            c["conv"].append(user(P_LOOKED, c["now"]))
            turn3.append(c)
        elif c["proc"] == "replay":
            issues = plan_issues(reply, parse(c["facts_reply"]), c["ep"]["target"])
            c["replay_fired"] = bool(issues)
            if issues:
                note = (P_NOTE.replace("ISSUES", " and ".join(issues))
                        .replace("BAY", str(reply.get("bay"))).replace("AISLE", str(reply.get("aisle"))))
                c["conv"].append(user(note, c["now"]))
                turn3.append(c)
            else:  # no contradiction: keep the plan, report no change
                c["final"] = final_action({**reply, "world_changed": False})
        elif c["proc"] == "facts":
            note = (P_ALL_FACTS.replace("FACTS", describe_facts(parse(c["facts_reply"]), c["ep"]["target"]))
                    .replace("BAY", str(reply.get("bay"))).replace("AISLE", str(reply.get("aisle"))))
            c["conv"].append(user(note, c["now"]))
            turn3.append(c)
    for c, r in zip(turn3, chat([c["conv"] for c in turn3])):
        c["conv"].append(assistant(r))

    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / f"{a.model}.jsonl", "w") as f:
        for c in cases:
            final = c.get("final") or final_action(parse(c["conv"][-1]["content"]))
            replies = [m["content"] for m in c["conv"] if m["role"] == "assistant"]
            f.write(json.dumps({"model": a.model, "id": c["ep"]["id"], "world": c["world"], "proc": c["proc"],
                                "final": final, "looked": c["looked"], "replay_fired": c["replay_fired"],
                                "facts_reply": c.get("facts_reply"), "replies": replies}) + "\n")
    print(f"wrote {len(cases)} cases to {a.out / (a.model + '.jsonl')}")


if __name__ == "__main__":
    main()
