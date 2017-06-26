import random

class ExperienceReplay:
  def __init__(self, num_replays):
    self.num_replays = num_replays
    self.replays = []

  def add_replay(self, replay):
    if len(self.replays) < self.num_replays:
      self.replays.append(replay)
    else:
      self.replays[random.randrange(self.num_replays)] = replay
