"""Policy/critic network with clipped normalized inputs.

rsl_rl normalizes inputs by a running mean and std, ``(x - mean) / (std + 0.01)``,
with no limit.  An input that barely varies in simulation (e.g. the rail gyro)
gets a tiny std, so a modest real-world reading could arrive as dozens of
standard deviations and push the network far outside anything it was trained
on.  Clipping the normalized input to ±``OBS_CLIP`` bounds that.

Same weights as ``MLPModel``, so existing checkpoints load unchanged.
"""

from __future__ import annotations

import torch
from rsl_rl.models import MLPModel

OBS_CLIP = 5.0   # standard deviations


class ClippedMLPModel(MLPModel):
    def get_latent(self, obs, masks=None, hidden_state=None) -> torch.Tensor:
        return super().get_latent(obs, masks, hidden_state).clamp(-OBS_CLIP, OBS_CLIP)
