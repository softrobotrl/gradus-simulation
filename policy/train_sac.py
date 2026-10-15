"""Train SAC on the board-pushing task.

Usage (from the repo root, in the gradus-rl env):
  python -m policy.train_sac                                   # defaults; run name from the date and time
  python -m policy.train_sac --run-name first_try --total-steps 2000000
  python -m policy.train_sac --resume logs/sac/first_try/checkpoints/latest.pt
  tensorboard --logdir logs/sac                                # watch the curves (http://localhost:6006)

Every TrainConfig field below is a command-line flag (underscores become dashes): --num-envs 512, --gamma 0.98 ...
Output in logs/sac/<run-name>/: config.yaml, metrics.csv, TensorBoard events, checkpoints/{latest,best,step_N}.pt
Step by step explanation of this loop: policy/README.md, "The training loop".
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

from pipeline.board_env import make_board_env
from policy.replay_buffer import ReplayBuffer
from policy.sac import SAC, SACHyperparams


@dataclass
class TrainConfig:
    config: str = "pipeline/configs/opencr_tdcr_spatial.yaml"  # robot config
    run_name: str = ""            # default: date_time
    seed: int = 0
    # environment
    backend: str = "auto"         # auto: MJX on a GPU if JAX can use one, else C MuJoCo threads ("mjx" / "cpu")
    num_envs: int = 256           # parallel envs; each env step adds this many transitions
    random_direction: bool = False  # True: a new push direction every episode
    episode_s: float = 8.0
    # reward weights (see pipeline/BOARD_TASK.md, "Reward")
    progress_reward: float = 100.0  # per m/s of board speed along the direction
    lateral_cost: float = 50.0    # per m/s sideways
    rate_cost: float = 0.05       # servo command change per step, as a fraction of the max change, squared
    effort_cost: float = 0.01     # servo angle as a fraction of its range, squared
    tension_cost: float = 0.0     # tendon tension as a fraction of the servo's max pull, squared; 0 = off
    action_filter: float = 0.2    # servo response per step (1 = instant; 0.2 ~ 0.1 s time constant, stops vibration)
    # learner
    learner_device: str = "auto"  # auto: cuda if PyTorch sees a GPU, else cpu
    total_steps: int = 10_000_000  # transitions to collect (num_envs per env step)
    learning_starts: int = 10_000  # transitions of random actions before learning starts
    buffer_size: int = 1_000_000
    batch_size: int = 4096        # on a GPU an update costs the same at 1024 or 4096 (launch overhead dominates)
    updates_per_step: int = 4     # gradient updates per env step (replay ratio = this * batch_size / num_envs)
    sac: SACHyperparams = field(default_factory=SACHyperparams)
    # bookkeeping
    log_every: int = 50           # env steps between log lines
    save_every: int = 1_000_000   # transitions between step_N checkpoints (latest.pt is saved at every log)


def parse_args() -> tuple[TrainConfig, str | None]:
    """Build the CLI from the dataclasses, so every field is a flag with its default shown in --help."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resume", help="checkpoint to continue from (its config is reused; flags override it)")

    def add(cls, prefix=""):
        for f in dataclasses.fields(cls):
            if dataclasses.is_dataclass(f.type):
                continue
            kind = {bool: lambda s: s.lower() in ("1", "true", "yes"), int: int, float: float}.get(f.type, str)
            parser.add_argument(f"--{f.name.replace('_', '-')}", dest=prefix + f.name, type=kind, default=None,
                                help=f"default: {f.default}")

    add(TrainConfig)
    add(SACHyperparams, "sac.")
    args = vars(parser.parse_args())
    resume = args.pop("resume")
    cfg = load_config(torch.load(resume, weights_only=False)["config"]) if resume else TrainConfig()
    for key, value in args.items():
        if value is not None:
            obj, name = (cfg.sac, key[4:]) if key.startswith("sac.") else (cfg, key)
            setattr(obj, name, value)
    return cfg, resume


def load_config(d: dict) -> TrainConfig:
    return TrainConfig(**{**d, "sac": SACHyperparams(**d["sac"])})


def env_kwargs(cfg: TrainConfig) -> dict:
    """The TrainConfig fields that configure the board env."""
    names = ("random_direction", "episode_s", "progress_reward", "lateral_cost", "rate_cost", "effort_cost",
             "tension_cost", "action_filter")
    return {name: getattr(cfg, name) for name in names}


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


def save(path: Path, agent: SAC, cfg: TrainConfig, step: int, best: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"agent": agent.state_dict(), "config": dataclasses.asdict(cfg), "step": step, "best": best},
               path.with_suffix(".tmp"))
    path.with_suffix(".tmp").replace(path)  # write then rename: a crash mid-save never corrupts the old file


def main() -> None:
    cfg, resume = parse_args()
    cfg.run_name = cfg.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("logs/sac") / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=False))
    torch.manual_seed(cfg.seed)
    learner = pick_learner(cfg.learner_device)

    env = make_board_env(cfg.config, cfg.num_envs, backend=cfg.backend, seed=cfg.seed,
                         **env_kwargs(cfg))
    print(f"learner on {learner}; run dir {run_dir}")
    agent = SAC(env.num_obs, env.num_actions, cfg.sac, learner)
    step, best = 0, -float("inf")
    if resume:
        ckpt = torch.load(resume, weights_only=False, map_location=learner)
        agent.load_state_dict(ckpt["agent"])
        step, best = ckpt["step"], ckpt["best"]
        print(f"resumed from {resume} at {step:,} transitions (the replay buffer starts empty again)")
    buffer = ReplayBuffer(cfg.buffer_size, env.num_obs, env.num_actions, learner)
    random_until = step + cfg.learning_starts if not resume else step  # resumed agents act with their policy
    learn_from = step + cfg.learning_starts

    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(run_dir)
    csv_file = (run_dir / "metrics.csv").open("a", newline="")
    csv_writer = None

    obs = env.reset()["policy"].to(learner)
    # Spread episode start times, so episodes don't all end (and reset) on the same step.
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

    while step < cfg.total_steps:
        # 1. act: random at first (to fill the buffer with varied data), then sample from the policy.
        t0 = time.perf_counter()
        if step < random_until:
            action = torch.rand(cfg.num_envs, env.num_actions, device=learner) * 2 - 1
        else:
            action = agent.act(obs)
        next_obs, reward, done, extras = env.step(action.to(env.device))
        next_obs, reward, done = next_obs["policy"].to(learner), reward.to(learner), done.to(learner).bool()
        time_out = extras["time_outs"].to(learner)
        t_sim += time.perf_counter() - t0

        # 2. store: the real post-step obs, and terminal only for failures (time-outs still bootstrap).
        real_next = extras["final_observation"].to(learner)
        terminal = done.float() * (1 - time_out)
        buffer.add(obs, action, reward, real_next, terminal)
        agent.normalizer.update(real_next)
        tracker.add(reward, done, time_out)
        accumulate({f"env{k}": v for k, v in extras["log"].items()})
        obs = next_obs
        step += cfg.num_envs
        iteration += 1

        # 3. learn
        t0 = time.perf_counter()
        if step >= learn_from:
            for _ in range(cfg.updates_per_step):
                accumulate(agent.update(buffer.sample(cfg.batch_size)))
        if learner == "cuda":
            torch.cuda.synchronize()
        t_learn += time.perf_counter() - t0

        # 4. log and save
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
                                            extrasaction="ignore")  # columns that appear only once learning starts
                if csv_file.tell() == 0:
                    csv_writer.writeheader()
            csv_writer.writerow(row)
            csv_file.flush()

            ret = row.get("episode/return")
            print(f"{step:>10,}  {row['time/transitions_per_s']:>7,.0f}/s  "
                  f"return {ret if ret is not None else float('nan'):7.2f}  "
                  f"board {row.get('env/board_speed', 0) * 1e3:5.2f} mm/s  lifted {row.get('env/lifted', 0):4.2f}  "
                  f"alpha {row.get('sac/alpha', float('nan')):.3f}  Q {row.get('sac/q', float('nan')):7.2f}  "
                  f"sim {row['time/sim_fraction']:.0%}", flush=True)

            ckpt_dir = run_dir / "checkpoints"
            if ret is not None and ret > best and step >= learn_from:
                best = ret
                save(ckpt_dir / "best.pt", agent, cfg, step, best)
            save(ckpt_dir / "latest.pt", agent, cfg, step, best)
            if step // cfg.save_every > saved_bucket:
                saved_bucket = step // cfg.save_every
                save(ckpt_dir / f"step_{step}.pt", agent, cfg, step, best)

    writer.close()
    csv_file.close()
    print(f"done: {step:,} transitions, best episode return {best:.2f}; checkpoints in {run_dir / 'checkpoints'}")


if __name__ == "__main__":
    main()
