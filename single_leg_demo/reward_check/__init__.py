"""Reward checks that need no training (see single_leg_demo/REWARD.md).

* ``preference``: score fixed behaviours (a scripted walk, a jerky walk, a
  noisy walk, standing still) under a reward.  A reward worth training must
  score the walk above standing still.  About a minute per reward.
* ``lockstep``: score trained checkpoints under two rewards on exactly the
  same motion, and check which reward ranks them by how well they do the task.
* ``gait``: the scripted side-swing walk both checks use.
"""
