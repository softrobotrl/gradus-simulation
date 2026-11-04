"""PPO timeout handling and recoverable checkpoints for the leg task."""

from pathlib import Path

import torch
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from rsl_rl.runners import OnPolicyRunner


class LegPPO(PPO):
    def process_env_step(self, obs, rewards, dones, extras):
        # rsl-rl 5.5 bootstraps timeouts with the value before the action. Use
        # the actual final observation, which env.step preserves before reset.
        timeouts = extras["time_outs"].to(self.device).bool()
        if timeouts.any():
            final_obs = extras["final_observation"].to(self.device)[timeouts]
            final_obs = TensorDict({"policy": final_obs}, batch_size=[len(final_obs)])
            with torch.no_grad():
                final_values = self.critic(final_obs).squeeze(-1)
            rewards = rewards.clone()
            rewards[timeouts] += self.gamma * final_values
        extras = {key: value for key, value in extras.items() if key != "time_outs"}
        super().process_env_step(obs, rewards, dones, extras)


class LegRunner(OnPolicyRunner):
    def save(self, path, infos=None):
        checkpoint = self.alg.save()
        checkpoint["iter"] = self.current_learning_iteration
        checkpoint["infos"] = infos
        path = Path(path)

        # Replace only after the full file is written, preserving the previous
        # checkpoint if a save is interrupted.
        for destination in dict.fromkeys((path, path.with_name("latest.pt"))):
            temporary = destination.with_suffix(".tmp")
            torch.save(checkpoint, temporary)
            temporary.replace(destination)
        self.logger.save_model(str(path), self.current_learning_iteration)
