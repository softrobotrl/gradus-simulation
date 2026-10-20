"""Train SAC on the Gradus dog with a nominal controller and residual actions.

The task and observation/action layout are in :mod:`pipeline.dog_env`; the
model is generated from :mod:`pipeline.dog_model`.  This script is the dog
counterpart of ``policy/train_sac.py`` and reuses the same SAC, replay buffer
and logging structure.

Quick start (from the repo root, in the ``gradus-rl`` env)::

    python -m pipeline.dog_model                          # build pipeline/build/gradus_dog/robot.xml
    python -m pipeline.validate_dog_env                   # model + env acceptance checks
    python -m policy.train_dog_sac --run-name stand --total-steps 200000

Watch it::

    tensorboard --logdir logs/dog_sac

Stand first, then locomotion::

    python -m policy.train_dog_sac --run-name walk --command-mode velocity \\
        --vx-min 0.0 --vx-max 0.4 --residual-scale 0.5 --total-steps 5000000 \\
        --sac.action-cost 1.0 --sac.actor-lr 1e-4

The defaults are tuned for standing: the nominal PD controller already stands,
and a strong L2 action cost (``--sac.action-cost``) keeps the residual policy
close to it.  For walking, lower the action cost so the policy can move the legs.

Every ``DogTrainConfig`` field is a CLI flag (underscores become dashes).  The
``sac`` fields use a ``sac.`` prefix, e.g. ``--sac.gamma 0.98``.
Output is in ``logs/dog_sac/<run-name>/``: config.yaml, metrics.csv,
TensorBoard events, checkpoints/{latest,best,step_N}.pt.
"""

import argparse
import csv
import dataclasses
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch
import yaml

from pipeline.dog_env import make_dog_env
from policy.replay_buffer import ReplayBuffer
from policy.sac import SAC, SACHyperparams


@dataclass
class DogTrainConfig:
    run_name: str = ""             # default: date_time
    seed: int = 0
    # environment
    backend: str = "cpu"           # the C MuJoCo thread backend; the MJCF is MJX-ready
    num_envs: int = 64
    num_threads: int = 0           # 0 = one per CPU (capped at num_envs)
    episode_s: float = 8.0
    command_mode: str = "stand"    # "stand" or "velocity"
    vx_min: float = 0.0
    vx_max: float = 0.4
    vy_min: float = 0.0
    vy_max: float = 0.0
    wz_min: float = 0.0
    wz_max: float = 0.0
    # nominal controller + residual action
    nominal_mode: str = "pd"       # "none", "constant" or "pd"
    stand_angle: float = 0.10
    gait_hz: float = 0.0
    gait_amp: float = 0.0
    kp_q: float = 0.30
    kd_q: float = 0.01
    residual_scale: float = 0.25
    action_filter: float = 0.50
    max_speed: float = 8.0
    # reward
    alive_reward: float = 1.0
    height_reward: float = 1.0
    upright_reward: float = 0.5
    velocity_reward: float = 0.5
    yaw_reward: float = 0.2
    posture_reward: float = 0.5
    tension_cost: float = 0.001
    rate_cost: float = 0.01
    residual_cost: float = 0.05
    height_target: float = 0.215
    height_sigma: float = 0.020
    upright_sigma: float = 0.15
    velocity_sigma: float = 0.20
    yaw_sigma: float = 0.50
    posture_sigma: float = 0.35
    terminate_height: float = 0.12
    terminate_tilt: float = 0.70
    init_height_noise: float = 0.003
    init_tilt_noise: float = 0.02
    init_joint_noise: float = 0.05
    init_vel_noise: float = 0.02
    init_yaw_noise: float = 0.0
    # learner
    learner_device: str = "auto"   # auto: cuda if PyTorch sees a GPU, else cpu
    total_steps: int = 2_000_000   # transitions to collect (num_envs per env step)
    learning_starts: int = 10_000  # transitions of random actions before learning
    buffer_size: int = 1_000_000
    batch_size: int = 1024
    updates_per_step: int = 2      # gradient updates per env step
    sac: SACHyperparams = field(default_factory=lambda: SACHyperparams(
        init_alpha=0.02, init_log_std=-1.0, action_cost=10.0, actor_lr=5e-5))
    # bookkeeping
    log_every: int = 25
    save_every: int = 500_000


def parse_args() -> tuple[DogTrainConfig, str | None]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resume", help="checkpoint to continue from (its config is reused; flags override it)")

    def add(cls, prefix=""):
        for f in dataclasses.fields(cls):
            if dataclasses.is_dataclass(f.type):
                continue
            kind = {bool: lambda s: s.lower() in ("1", "true", "yes"), int: int, float: float}.get(f.type, str)
            parser.add_argument(f"--{prefix}{f.name.replace('_', '-')}", dest=prefix + f.name, type=kind, default=None,
                                help=f"default: {f.default}")

    add(DogTrainConfig)
    add(SACHyperparams, "sac.")
    args = vars(parser.parse_args())
    resume = args.pop("resume")
    cfg = load_config(torch.load(resume, weights_only=False)["config"]) if resume else DogTrainConfig()
    for key, value in args.items():
        if value is not None:
            obj, name = (cfg.sac, key[4:]) if key.startswith("sac.") else (cfg, key)
            setattr(obj, name, value)
    return cfg, resume


def load_config(d: dict) -> DogTrainConfig:
    return DogTrainConfig(**{**d, "sac": SACHyperparams(**d["sac"])})


def env_kwargs(cfg: DogTrainConfig) -> dict:
    names = (
        "episode_s", "command_mode", "nominal_mode", "stand_angle", "gait_hz", "gait_amp", "kp_q", "kd_q",
        "residual_scale", "action_filter", "max_speed", "alive_reward", "height_reward", "upright_reward",
        "velocity_reward", "yaw_reward", "posture_reward", "tension_cost", "rate_cost", "residual_cost",
        "height_target", "height_sigma", "upright_sigma", "velocity_sigma", "yaw_sigma", "posture_sigma",
        "terminate_height", "terminate_tilt", "init_height_noise", "init_tilt_noise", "init_joint_noise",
        "init_vel_noise", "init_yaw_noise",
    )
    out = {name: getattr(cfg, name) for name in names}
    out["vx_range"] = (cfg.vx_min, cfg.vx_max)
    out["vy_range"] = (cfg.vy_min, cfg.vy_max)
    out["wz_range"] = (cfg.wz_min, cfg.wz_max)
    if cfg.num_threads > 0:
        out["num_threads"] = cfg.num_threads
    return out


def pick_learner(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


class EpisodeTracker:
    """Per-env running return and length; collects the totals of episodes as they finish."""

    def __init__(self, num_envs: int, device):
        self.ret = torch.zeros(num_envs, device=device)
        self.len = torch.zeros(num_envs, device=device)
        self.finished_returns: list[float] = []
        self.finished_lengths: list[float] = []
        self.failures = 0

    def add(self, reward, done, time_out) -> None:
        self.ret += reward
        self.len += 1
        if done.any():
            self.finished_returns += self.ret[done].tolist()
            self.finished_lengths += self.len[done].tolist()
            self.failures += int((done & (time_out == 0)).sum())
            self.ret[done] = 0
            self.len[done] = 0

    def pop(self) -> dict[str, float]:
        if not self.finished_returns:
            return {}
        n = len(self.finished_returns)
        out = {"episode/return": sum(self.finished_returns) / n, "episode/length": sum(self.finished_lengths) / n,
               "episode/failure_rate": self.failures / n, "episode/count": n}
        self.finished_returns, self.finished_lengths, self.failures = [], [], 0
        return out


def save(path: Path, agent: SAC, cfg: DogTrainConfig, step: int, best: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"agent": agent.state_dict(), "config": dataclasses.asdict(cfg), "step": step, "best": best},
               path.with_suffix(".tmp"))
    path.with_suffix(".tmp").replace(path)


def main() -> None:
    cfg, resume = parse_args()
    cfg.run_name = cfg.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("logs/dog_sac") / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=False))
    torch.manual_seed(cfg.seed)
    learner = pick_learner(cfg.learner_device)

    env = make_dog_env(cfg.num_envs, backend=cfg.backend, seed=cfg.seed, **env_kwargs(cfg))
    print(f"learner on {learner}; run dir {run_dir}")
    agent = SAC(env.num_obs, env.num_actions, cfg.sac, learner)
    step, best = 0, -float("inf")
    if resume:
        ckpt = torch.load(resume, weights_only=False, map_location=learner)
        agent.load_state_dict(ckpt["agent"])
        step, best = ckpt["step"], ckpt["best"]
        print(f"resumed from {resume} at {step:,} transitions (the replay buffer starts empty again)")
    buffer = ReplayBuffer(cfg.buffer_size, env.num_obs, env.num_actions, learner)
    random_until = step + cfg.learning_starts if not resume else step
    learn_from = step + cfg.learning_starts

    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(run_dir)
    csv_file = (run_dir / "metrics.csv").open("a", newline="")
    csv_writer = None

    obs = env.reset()["policy"].to(learner)
    env.episode_length_buf = torch.randint(0, env.max_episode_length, (cfg.num_envs,))
    tracker = EpisodeTracker(cfg.num_envs, learner)
    sums: dict[str, torch.Tensor | float] = {}
    counts: dict[str, int] = {}
    t_sim = t_learn = 0.0
    t_log, step_log = time.perf_counter(), step
    iteration = 0
    saved_bucket = step // cfg.save_every

    def accumulate(stats: dict) -> None:
        for k, v in stats.items():
            sums[k] = sums.get(k, 0.0) + v
            counts[k] = counts.get(k, 0) + 1

    try:
        while step < cfg.total_steps:
            t0 = time.perf_counter()
            if step < random_until:
                action = torch.rand(cfg.num_envs, env.num_actions, device=learner) * 2 - 1
            else:
                action = agent.act(obs)
            next_obs, reward, done, extras = env.step(action.to(env.device))
            next_obs, reward, done = next_obs["policy"].to(learner), reward.to(learner), done.to(learner).bool()
            time_out = extras["time_outs"].to(learner)
            t_sim += time.perf_counter() - t0

            real_next = extras["final_observation"].to(learner)
            terminal = done.float() * (1 - time_out)
            buffer.add(obs, action, reward, real_next, terminal)
            agent.normalizer.update(real_next)
            tracker.add(reward, done, time_out)
            accumulate({f"env{k}": v for k, v in extras["log"].items()})
            obs = next_obs
            step += cfg.num_envs
            iteration += 1

            t0 = time.perf_counter()
            if step >= learn_from:
                for _ in range(cfg.updates_per_step):
                    accumulate(agent.update(buffer.sample(cfg.batch_size)))
            if learner == "cuda":
                torch.cuda.synchronize()
            t_learn += time.perf_counter() - t0

            if iteration % cfg.log_every == 0 or step >= cfg.total_steps:
                now = time.perf_counter()
                row = {"step": step, "time/transitions_per_s": (step - step_log) / (now - t_log),
                       "time/sim_fraction": t_sim / max(t_sim + t_learn, 1e-9)}
                row |= {k: float(v) / counts[k] for k, v in sums.items()}
                row |= tracker.pop()
                sums.clear(), counts.clear()
                t_sim = t_learn = 0.0
                t_log, step_log = now, step
                for k, v in row.items():
                    if k != "step":
                        writer.add_scalar(k, v, step)
                if csv_writer is None:
                    later = ["episode/return", "episode/length", "episode/failure_rate", "episode/count", "loss/critic",
                             "loss/actor", "loss/alpha", "sac/alpha", "sac/entropy", "sac/q", "sac/target_q"]
                    csv_writer = csv.DictWriter(csv_file, fieldnames=list(dict.fromkeys(list(row) + later)),
                                                extrasaction="ignore")
                    if csv_file.tell() == 0:
                        csv_writer.writeheader()
                csv_writer.writerow(row)
                csv_file.flush()

                ret = row.get("episode/return")
                print(f"{step:>10,}  {row['time/transitions_per_s']:>7,.0f}/s  "
                      f"return {ret if ret is not None else float('nan'):7.2f}  "
                      f"height {row.get('env/base_height', 0):.3f} m  "
                      f"speed {row.get('env/speed', 0):.2f} m/s  "
                      f"tilt {row.get('env/tilt', 0):.3f}  "
                      f"tension {row.get('env/tension', 0):.2f}  "
                      f"alpha {row.get('sac/alpha', float('nan')):.3f}  "
                      f"Q {row.get('sac/q', float('nan')):7.2f}  "
                      f"sim {row['time/sim_fraction']:.0%}", flush=True)

                ckpt_dir = run_dir / "checkpoints"
                if ret is not None and ret > best and step >= learn_from:
                    best = ret
                    save(ckpt_dir / "best.pt", agent, cfg, step, best)
                save(ckpt_dir / "latest.pt", agent, cfg, step, best)
                if step // cfg.save_every > saved_bucket:
                    saved_bucket = step // cfg.save_every
                    save(ckpt_dir / f"step_{step}.pt", agent, cfg, step, best)
    finally:
        env.close()
        writer.close()
        csv_file.close()

    print(f"done: {step:,} transitions, best episode return {best:.2f}; checkpoints in {run_dir / 'checkpoints'}")


if __name__ == "__main__":
    main()
