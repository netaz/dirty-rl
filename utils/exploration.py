"""Epsilon-Greedy exploration schedules."""


import math


class LinearExplorationSchedule(object):
    def __init__(self, start_eps=1., end_eps=0.1, duration=1000000):
        self.eps = start_eps
        self.end_eps = end_eps
        self.step_size = (start_eps - end_eps) / duration

    def step(self):
        if self.eps <= self.end_eps:
            return self.eps
        self.eps -= self.step_size
        return self.eps


class ExponentialExplorationSchedule(object):
    def __init__(self, start_eps, end_eps, eps_decay):
        self.start_eps = self.eps = start_eps
        self.end_eps = end_eps
        self.steps_done = 0
        self.eps_decay = eps_decay

    def step(self):
        eps_threshold = self.end_eps + (self.start_eps - self.end_eps) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        self.eps = eps_threshold
        return self.eps
