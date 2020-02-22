""" Vanilla (Stochastic) Policy Gradients to control Atari Pong - Multiprocess version

Based on Andrej Karpathy's blog (http://karpathy.github.io/2016/05/31/rl/)
and code (https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5), and
adapted to PyTorch, and with some changes.

This implementation will execute only on the CPU.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchviz
from utils.memory import FastMemory
from utils.utils import GameProgressBar, load_checkpoint, save_checkpoint,\
                        get_device, ExponentialMovingAvg
from utils.rl import discount_rewards, baseline_mean_reward, entropy,\
                     long_term_entropy
from utils.log import TBWrapper
from utils.preprocess import pre_process_game_frame
from  utils.atari import encode_action
from functools import partial


experiment_name = "final_v11_mp_v4"

# Define the game frame pre-processing pipeline
PRE_PROCESS_OUTPUT_DIM = 80
pre_process_atari = partial(pre_process_game_frame,
                            n_channels=1,
                            output_shape=(PRE_PROCESS_OUTPUT_DIM, PRE_PROCESS_OUTPUT_DIM))


def empty_frame(device):
    """Generate an empty game frame"""
    return torch.zeros(PRE_PROCESS_OUTPUT_DIM**2,
                       dtype=torch.uint8, device=device)


class KarpathyPongPolicy(nn.Module):
    """Action policy approximation

    Inputs are 80x80 and the output is binary.
    """
    H = 200  # number of hidden layer neurons
    def __init__(self, device, input_dim, n_actions):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim,
                                   KarpathyPongPolicy.H,
                                   bias=False)
        self.fc2 = torch.nn.Linear(KarpathyPongPolicy.H,
                                   n_actions,
                                   bias=False)
        self._init_weights()
        self.to(device)

    def _init_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight, a=1, nonlinearity='relu')
        # Following John Schulman's advice to maximize initial entropy
        # (https://www.youtube.com/watch?v=jmMsNQ2eug4)
        torch.nn.init.constant_(self.fc2.weight, 0)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        logits = self.fc2(h)
        action_prob = F.softmax(logits, dim=0)
        return action_prob, logits

    def select_action(self, current_state):
        """Stochastically select an action to take"""
        action_prob, logits = self(current_state)
        C = torch.distributions.Categorical(logits=logits)
        action = C.sample()
        logprob = C.log_prob(action)
        # logprob = torch.log(action_prob)
        return action.to(get_device()), logprob.to(get_device()), action_prob.to(get_device())


def collect_experience(env,
                       policy,
                       render,
                       progress_bar):
    """Play a full episode, and collect experience, following `policy`.

    Args:
        env: (gym.Env) environment
        policy: (nn.Module) agent policy
        render: (bool) set to True to render game frames
        progress_bar: (GameProgressBar) progress indicator
    """
    device = get_device()
    memory = FastMemory()
    done = False
    episode_reward = 0

    # Run a full episode trajectory
    observation = env.reset()
    prev_x = None  # used in computing the difference frame
    while not done:
        if render:
            env.render()

        # preprocess the observation, set input to network to be difference image
        with torch.no_grad():
            cur_x = pre_process_atari(observation).flatten()
            x = cur_x - prev_x if prev_x is not None else empty_frame(device)
            prev_x = cur_x

        # forward the policy network and sample an action from the returned probability
        action, action_logprob, action_prob = policy.select_action(x.float())

        # step the environment and get new measurements
        observation, reward, done, info = env.step(encode_action(action.item()))
        episode_reward += reward

        action_mask = F.one_hot(action, encode_action.n_actions)
        memory.append((action_mask * action_logprob).to(device),
                      torch.tensor(reward, device=device),
                      (action_prob).to(device))

        if reward != 0 and progress_bar is not None:
            # Game is done, but episode may still be in progress
            progress_bar(reward)
    return memory, episode_reward


def compute_and_accumulate_policy_gradients(experience):
    """Compute the policy gradients using the provided experience and
       accumulate in the graph.

    Args:
        experience: (utils.memory.*) experience trajectories from playing
            one or more episodes.  Each episode may be composed of several
            (game) trajectories.

    Returns:
        policy_loss: policy loss
        H: entropy
        lt_entropy: long-term entropy (discounted)
    """
    batch_action_logp, batch_rewards, batch_action_probs = experience.vertical_batch()

    # Compute the rewards-to-go
    discounted_epr = discount_rewards(batch_rewards, args.reward_discount).unsqueeze(1)
    with torch.no_grad():
        discounted_epr -= baseline_mean_reward(discounted_epr)
    batch_action_logp = batch_action_logp.squeeze()
    H = entropy(batch_action_logp, batch_action_probs)
    lt_entropy = long_term_entropy(batch_rewards, batch_action_logp,
                                   batch_action_probs, args.reward_discount)
    policy_loss = (-batch_action_logp * discounted_epr).sum(dim=1).mean()
    #entropy_bonus = 0.1 * H
    #policy_loss -= (entropy_bonus + lt_entropy)
    if False:
        torchviz.make_dot(policy_loss,
                          params=dict(policy.named_parameters())).render("pg_loss_backward",
                                                                         format="png")
        exit()
    policy_loss.backward()  # accumulate gradients
    return policy_loss, H, lt_entropy


def update_policy(policy,
                  optimizer,
                  episode_number,
                  tb_writer,
                  log_params):
    """Update the agent policy using the accumulated policy gradients.

    Args:
        policy: (nn.Module) the agent policy
        optimizer: (torch.optim) loss optimizer
        episode_number: (int) current episode (used for logging)
        tb_writer: (utils.log.TBWrapper) TensorBoard wrapper
        log_params: (boolean) controls whether we log the policy parameters.
    """
    def _update_policy_weights(optimizer):
        print('\nUpdating policy...')
        optimizer.step()
        optimizer.zero_grad()

    if log_params:
        tb_writer.log_parameters(episode_number, policy,
                                 [name for name, _ in policy.named_parameters()])
    _update_policy_weights(optimizer)
    save_checkpoint(policy, optimizer, episode_number, tb_writer.tb_log_dir)


"""Configuration"""
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--policy-lr', default=1e-4, type=float,
                       help='policy optimizer learning rate')
argparser.add_argument('--render', action='store_true',
                       help='render flag')
argparser.add_argument('--reward-eam-factor', default=0.01, type=float,
                       help='reward exponential-moving-average factor (default: 0.01)')
argparser.add_argument('--reward-discount', '--gamma', default=0.99, type=float,
                       help='reward discount factor')
argparser.add_argument('--resume', type=str, default=None,
                       help='load a previously serialized checkpoint')
argparser.add_argument('--profile', action='store_true',
                       help='enable when using python -m torch.utils.bottleneck')
argparser.add_argument('--log-params', action='store_true',
                       help='log parameter histograms (warning: this requires large storage space)')
argparser.add_argument('--num-actions', default=3, type=int, choices=(2, 3),
                        help='the number of actions in the action space')
argparser.add_argument('--device', default=None, type=str,
                        help='the device to use (cpu, cuda, cuda:n)')
optimizer_grp = argparser.add_argument_group('Optimizer Arguments')
optimizer_grp.add_argument('--rms-decay-rate', default=0.99, type=float,
                           help='decay factor for RMSProp leaky sum of grad^2 (default: 0.99)')

pg_grp = argparser.add_argument_group('Policy Gradient Arguments')
pg_grp.add_argument('-b', '--batch-size', default=10, type=int,
                    help='mini-batch size (default: 10)')

mp_grp = argparser.add_argument_group('Multi-Processing Arguments')
mp_grp.add_argument('--world-size', default=2, type=int,
                    help='number of processes to use for rollout and backprop')
mp_grp.add_argument('--gradient-reduce', default='sum', type=str, choices=('sum', 'mean'),
                    help='Gradient reduction operator')
mp_grp.add_argument('--mp-master-port', default=29500, type=int,
                    help='TCP port used by the master node for inter-process communication')



def reduce_gradients(model, reduction, size):
    """ Gradient reduction. """
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        if reduction == 'mean':
            param.grad.data /= size


def vanilla_policy_gradient(args, reward_ema=None, writer=None):
    """This is where the VPG magic happens"""
    device = get_device(args.device)
    policy = KarpathyPongPolicy(device, PRE_PROCESS_OUTPUT_DIM**2, args.num_actions)
    optimizer = torch.optim.RMSprop(policy.parameters(),
                                    lr=args.policy_lr,
                                    alpha=args.rms_decay_rate,
                                    centered=True)
    if args.resume:
        load_checkpoint(policy, optimizer, args.resume)
    policy.train()
    env = gym.make("Pong-v0")
    encode_action.n_actions = args.num_actions
    rank = dist.get_rank()
    episode_number = rank
    progress_bar = None
    world_size = dist.get_world_size()
    while True:
        optimizer.zero_grad()
        for _ in range(args.batch_size // world_size):
            experience, episode_reward = collect_experience(env,
                                                            policy,
                                                            args.render,
                                                            progress_bar)

            # optimizer.zero_grad()
            policy_loss, H, lt_entropy = compute_and_accumulate_policy_gradients(experience)
            reward_ema.update(episode_reward)
            if writer is not None:
                writer.log_kvdict({'episode_reward': episode_reward,
                                   'running_return': reward_ema.value_mt(),
                                   'policy_loss': policy_loss.item(),
                                   'ep_action_entropy': H.item(),
                                   'lt_action_entropy': lt_entropy.item()}, episode_number)
            print(f'episode={episode_number} '
                  f'reward={episode_reward} running mean={reward_ema.value_mt():.2f} '
                  f'loss={policy_loss.item():.5f}',
                  end='\n')
            episode_number += world_size

        # Update policy parameters every batch_size episodes
        reduce_gradients(policy, args.gradient_reduce, args.batch_size)
        if rank == 0:
            # Emit a log only for the Master process
            print('\nUpdating policy...')
        optimizer.step()

        #update_policy(policy, optimizer, episode_number, writer, args.log_params)


import torch.distributed as dist
from torch.multiprocessing import Process
from multiprocessing.managers import BaseManager


def init_processes(rank, size, args, fn, port, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)

    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(args)


if __name__ == "__main__":
    args = argparser.parse_args()
    size = args.world_size
    # Force CPU
    backend = 'gloo' if get_device('cpu') == 'cpu' else 'nccl'
    processes = []

    # https://stackoverflow.com/questions/3671666/sharing-a-complex-object-between-python-processes
    BaseManager.register('ExponentialMovingAvg', ExponentialMovingAvg)
    BaseManager.register('TBWrapper', TBWrapper)
    manager = BaseManager()
    manager.start()
    reward_ema = manager.ExponentialMovingAvg(args.reward_eam_factor)
    writer = manager.TBWrapper(experiment_name)

    vanilla_policy_gradient_mt = partial(vanilla_policy_gradient,
                                         reward_ema=reward_ema,
                                         writer=writer)
    for rank in range(size):
        p = Process(target=init_processes,
                    args=(rank, size, args, vanilla_policy_gradient_mt, args.mp_master_port, backend))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()