# Stale World Pilot: does a VLM agent notice that the warehouse changed?

A preliminary study in simulation (Habitat-Sim). A robot agent looks at a warehouse floor at 09:00, the
floor may change (a crate is moved, or an aisle is blocked), and at 09:20 the agent gets a task that depends
on what it saw. We measure whether it notices the change or acts on the old observation.

This is a simulated decision task for vision-language models, not a robot experiment: the agent picks a bay
and an aisle, and Habitat's navmesh planner decides whether that route is passable. There is no robot
control. Project page: https://laulauraluo.github.io/stale-world-pilot/ (source in `docs/`). Results: `results/table.md`, `results/summary.csv`,
`results/fig_results.pdf`, and every raw reply in `results/<model>.jsonl`.

## Reproduce

Needs Linux with NVIDIA GPUs (headless rendering via EGL), Slurm, conda, and internet for the one-time setup.
We used one NVIDIA L40S (46 GB) per model.

```bash
bash setup.sh                                  # once: env-hab (habitat-sim), env-vlm (vLLM 0.11.0), models
SBATCH_ARGS="-p <gpu partition>" bash run_all.sh   # render, run the three models in parallel, score
```

Optional environment variables: `SBATCH_ARGS` (extra `sbatch` options such as the partition), `VLM_PY` (a
Python that already has `vllm==0.11.0`; default `env-vlm/bin/python`), `HF_HOME` (Hugging Face cache).
Rendering takes under a minute, Qwen and Mistral a few minutes each, GLM about 35 minutes.

Single steps: `mkdir -p logs` (Slurm needs it), then `sbatch $SBATCH_ARGS render.sbatch` or
`sbatch $SBATCH_ARGS --export=ALL,MODEL=qwen3vl8b agent.sbatch`; give smoke tests `OUT=results_smoke,LIMIT=10`
so they do not overwrite `results/`. The animation:
`sbatch $SBATCH_ARGS --gres=gpu:1 --wrap "env-hab/bin/python animate.py --out anim"`.
Scoring alone runs anywhere with Python 3 and matplotlib:
`python analyze.py --data data --results results --compare results_v1`.

The repository keeps `data/episodes.jsonl` and `results/*.jsonl`, which is enough to re-score; the 600
rendered images (`data/img/`) are re-created by `render.sbatch` (seed 0).

Result folders: `results/` is the reported run (three models, four procedures). `results_v1/` is a first
run (two models, no facts procedure), kept to check that re-running gives the same actions; `--compare`
writes `results/compare.txt` (391 to 400 of 400 final actions identical per cell). `results_glm4k/` is a GLM
run capped at 4096 new tokens; 66/1600 replies ran out of tokens before answering, so the reported GLM run
uses 8192 (the value in GLM's model card example).

## Files

| file | what it does |
| --- | --- |
| `sim.py` | builds the warehouse (trimesh -> glb), samples episodes, renders the camera images in Habitat (EGL, headless), and decides with Habitat's navmesh planner whether each aisle is passable. Self-check: the planner must agree with where the barrier is. |
| `agent.py` | runs one VLM (vLLM, greedy decoding) on every episode: 4 procedures x 2 worlds. Saves every reply (a reasoning model's <think> part is dropped). |
| `analyze.py` | scores the replies, writes `results/table.md`, `results/summary.csv`, `results/fig_results.pdf` and `.svg`. Starts with a self-test of the scoring rules. Also counts 09:00 descriptions that call an aisle blocked (none is), a check on the models' perception. |
| `animate.py` | renders the episode-1 animation (`docs/animation.mp4`, `.gif`) in Habitat on navmesh routes; same environment as `sim.py`. |
| `setup.sh`, `run_all.sh`, `*.sbatch` | environment setup and the one-command pipeline. |
| `docs/` | the static project page (`index.html`) with the animation and `fig_results.svg`. |

## Design

**Scene.** A procedurally built warehouse: four numbered bays on the back wall, three colored crates in
three of them, and two aisles (left, right) between shelving racks. An overview camera above the loading
dock sees the whole floor (768x512 RGB).

**Episode.** `sim.py` samples a layout and a target crate (seed 0). Each episode has two worlds:

- static: nothing changes. The 09:20 image still differs slightly (camera jitter: up to 10 cm in x and in z,
  up to 2 degrees of yaw).
- changed: even ids move the target crate to the empty bay; odd ids put a barrier across the aisle on the
  target bay's side (bays 1-2: left, 3-4: right). So the changed world is half moved, half blocked.

**Agent.** A chat: turn 1 shows the 09:00 image and asks for a description (the agent's memory). Turn 2
gives only the time (09:20) as a cue, no hint that anything moved (the system prompt says other robots and
workers share the floor), and asks it to bring the target crate. The exact prompts are at the top of `agent.py`. The agent answers `{"bay", "aisle", "world_changed"}`. Four procedures for turn 2:

| procedure | what the agent gets |
| --- | --- |
| memory | no new image; it may answer `{"action": "look"}` once to get the current image |
| recheck | the current image, and the instruction to check what changed before acting |
| replay | it plans from memory; the facts the plan depends on (where the target crate is, whether each aisle is clear) are asked again on the current image in a fresh chat; code compares them with the plan and, only if they contradict it, shows the agent a note and the image and lets it decide again |
| facts | ablation of replay: the same plan and the same fresh-chat facts (copied from the replay case), but the agent always sees all the facts and the image and compares them itself; code compares nothing |

Replay follows one idea: re-run the same query on the current state and let code decide whether the
change matters for this task. The query here needs perception, so the re-run is a second model call.

**Metrics** (definitions in the `analyze.py` docstring): task success (sim-checked), detection rate and
false-alarm rate (the agent's `world_changed`), stale-action rate (the action fits the old world but not
the new one), and how often the memory agent asked to look. For replay, the detection rate is mostly set by
code: with no contradiction `world_changed` is set to false, and with one the agent is told what changed. Each cell has n = 200 episodes and a 95%
Wilson interval.

**Models.** Qwen3-VL-8B-Instruct (Alibaba), Mistral-Small-3.2-24B-Instruct-2506 (FP8 by RedHatAI) and
GLM-4.1V-9B-Thinking (Zhipu; a reasoning model, up to 8192 new tokens per reply), three open models from
different families, run with vLLM 0.11.0 on one L40S each.

**Ablation.** recheck -> facts -> replay adds one step at a time: facts adds the separate fresh-chat question
about the plan's facts (and the plan-first step); replay adds the code comparison, which also decides
whether the agent is told anything at all.

**Changes made after smoke tests, before the full run.** (1) The 09:20 prompt first said that other robots
and workers had been moving things; Qwen then asked to look in 10/10 smoke episodes, which tests
hint-following, so the prompt now gives only the time (the system prompt says others share the floor).
(2) Extra lights in `make_sim`, because under the default lights the yellow crate's front face looked brown.
(3) GLM's limit of new tokens went from 4096 to 8192: at 4096, 6/80 smoke replies ran out before answering.

**Known perception issue.** Mistral calls an aisle blocked in 137/200 of its 09:00 descriptions although no
aisle is blocked then (Qwen: 0/200). It does not hurt static-world success (both aisles are clear), but some
of Mistral's blocked-aisle successes are accidental. `analyze.py` prints this count under the table, and also
the replies with no valid answer: GLM has 32/1600 (greedy reasoning sometimes loops until the 8192-token
limit) and 9/200 empty 09:00 descriptions; Qwen and Mistral have none.
