import torch
from functools import lru_cache


def discount_rewards(rewards, reward_discount):
    """Compute the rewards-to-go.  Based on Karpathy's code.

    Args:
      rewards: (list) a list of rewards collected while playing an episode.  Remember that
        an episode in Pong is composed of many games; each ending with +1 or -1.  The
        `rewards` array contains many games, and for each game we compute the running sum of
        discounted rewards.
    """
    discounted_r = torch.zeros_like(rewards)
    running_add = 0
    with torch.no_grad():
        # Run the computation backwards, starting at the last reward received in this episode.
        for t in reversed(range(0, rewards.size(0))):
            if rewards[t] != 0:
                running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
            running_add = running_add * reward_discount + rewards[t]
            discounted_r[t] = running_add

        # # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        # discounted_r -= discounted_r.mean()
        # reward_std = discounted_r.std()
        # if not reward_std == 0 and not torch.isnan(reward_std):
        #     discounted_r /= discounted_r.std()
        return discounted_r


def baseline_mean_reward(rewards):
    """The most basic baseline function.

    It is wrapped in a function, just to give it some meaning.
    """
    return rewards.mean()


def entropy(logp, p):
    """Compute the entropy of `p` - probability density function approximation.

    We need this in order to compute the entropy-bonus.
    """
    H = -(logp * p).sum(dim=1).mean()
    return H


def long_term_entropy(r, logp, p, reward_discount):
    """Discounted entropy"""
    lt_H = torch.zeros_like(r)
    running_H = 0
    for t in reversed(range(0, r.size(0))):
        if r[t] != 0:
            running_H = 0  # reset the sum, since this was a game boundary (pong specific!)
        H = -(logp[t] * p[t]).sum()
        running_H = running_H * reward_discount + H
        lt_H[t] = running_H
    return lt_H.mean()


@lru_cache(maxsize=2000)
def gamma(exponent):
    return gamma ** exponent
