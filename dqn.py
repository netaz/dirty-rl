"""Deep Q-Learning

Based on the code cited below with various changes to make it more conformant with the DQN paper.
https://github.com/pytorch/tutorials/blob/master/intermediate_source/reinforcement_q_learning.py
Also described in https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html.

Changes:
    - The Q-value function input (observation) is composed of 4 RGB frames, instead of a 1D
        difference frame.
    - The replay-buffer is more sophisticated because it supports extracting the history-context
        of a sampled frame (i.e. when sampling a frame we also get the 3 temporally-neighboring
        frames.  The buffer implementation is from the Berkeley CS294 RL course.
    - Gradient debugging code
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
import gym
import torch
import torch.nn as nn
import itertools
from utils.exploration import LinearExplorationSchedule
from utils.utils import GameProgressBar, load_checkpoint, save_checkpoint, \
                        get_device, ExponentialMovingAvg, dtype
from utils.memory import ReplayBuffer
from utils.log import TBWrapper
from utils.preprocess import pre_process_game_frame
import torchviz
import torch.nn.functional as F
import numpy as np
import random


experiment_name = "dqn-v10_huber"
writer = TBWrapper(experiment_name)


class QNet(nn.Module):
    def __init__(self, n_actions, n_input_ch, input_shape, bias=True, bn=True):
        """Q-value function approximator: compute the expected value of an input state.

        Given an input state, produce a tensor with an expected value for each action.
        The input to the DNN is a group of 80x80 image frames produced by the environment,
        and pre-processed to scale and crop.

        Args:
            n_actions: (int) the number of actions in the action-space
            n_input_ch: (int) number of input channels
            input_shape: (tuple) the shape of the input
            bias: (boolean) add bias parameters to the convolution and linear layers.
            bn: (boolean)

        This is a copy of the model provided here:
        https://github.com/pytorch/tutorials/blob/master/intermediate_source/reinforcement_q_learning.py
        Changes:
            - configurable BN and bias settings
            - gradient hooks for debug
        """
        super().__init__()
        self.n_actions = n_actions
        self.conv1 = nn.Conv2d(n_input_ch, 16, kernel_size=3, stride=2, bias=bias)
        self.bn1 = nn.BatchNorm2d(16) if bn else nn.Identity()
        self.relu_conv1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, bias=bias)
        self.bn2 = nn.BatchNorm2d(32) if bn else nn.Identity()
        self.relu_conv2 = nn.ReLU()

        def conv2d_size_out(size, kernel_size=3, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(input_shape[0]))
        convh = conv2d_size_out(conv2d_size_out(input_shape[1]))
        linear_input_size = convw * convh * 32

        self.fc1 = nn.Linear(linear_input_size, 256, bias=bias)
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(256, n_actions, bias=bias)

        # Various gradient hooks - enable if you are debugging
        # self.relu_fc1_hook_handle = self.relu_fc1.register_backward_hook(relu_fc1_backward_hook)
        # self.fc1_hook_handle = self.fc1.register_backward_hook(fc1_backward_hook)
        # self.fc2_hook_handle = self.fc2.register_backward_hook(fc2_backward_hook)
        # self.fc2.weight.register_hook(param_backward_hook)

    def forward(self, x):
        x = x / 255. #- 0.5
        x = self.relu_conv1(self.bn1(self.conv1(x)))
        x = self.relu_conv2(self.bn2(self.conv2(x)))
        x = x.flatten(start_dim=1)
        x = self.relu_fc1(self.fc1(x))
        # Debug placeholder to identify empty feature-maps
        #if x.data.abs().sum() == 0:
        #    debug = True
        x = self.fc2(x)
        return x


def param_backward_hook(grad_output):
    print(grad_output[0].data.abs().sum(), grad_output[0].data.sum())


# this is per: https://arxiv.org/pdf/1312.5602.pdf
# see also: https://github.com/transedward/pytorch-dqn/blob/master/dqn_model.py
def relu_fc1_backward_hook(module, grad_input, grad_output):
    print(f"relu_fc1: grad_input = {grad_input[0].data.sum()}  grad_output = {grad_output[0].data.sum()}")
    # if grad_input[0].data.abs().sum() == 0:
    #     print("in zero - fc1")
    # if grad_output[0].data.abs().sum() == 0:
    #     print("out zero - fc1")


def fc1_backward_hook(module, grad_input, grad_output):
    print(f"fc1: grad_input = {grad_input[0].data.sum()}  grad_output = {grad_output[0].data.sum()}")
    # if grad_input[0].data.abs().sum() == 0:
    #     print("in zero - fc1")
    # if grad_output[0].data.abs().sum() == 0:
    #     print("out zero - fc1")


def fc2_backward_hook(module, grad_input, grad_output):
    # if grad_input[0].data.abs().sum() == 0:
    #     print("in zero - fc2")
    # if grad_output[0].data.abs().sum() == 0:
    #     print("out zero - fc2")
    print(grad_input[0].shape)
    print(grad_output[0].shape)
    print(f"fc2: grad_input = {grad_input[0].data.sum()}  grad_output = {grad_output[0]}")


def epsilon_greedy_action(state, eps, n_actions, q_behavior):
    """Select an action, under an epsilon-greedy exploration schedule.

    Arguments:
        state: (torch.Tensor) observation
        eps: (float) epsilon-greedy threshold value
        n_actions: (int) number of actions to choose from
        q_behavior: (nn.Module) Q-value model used for acting

    Supports discrete action-spaces only.
    """
    if np.random.random() <= eps:
        # Random
        action = random.randrange(n_actions)
        action = torch.tensor([[action]], device=get_device(), dtype=torch.long)
    else:
        # Greedy
        with torch.no_grad():
            state_action_values = q_behavior(state.type(dtype))
            # Take argmax of the action row-tensor - this is the index of the
            # action with the largest expected value for state (s)
            action = state_action_values.max(1)[1].view(1, 1)
    return action


def is_heating_up(replay_buf, cnt_transitions):
    """Are we still heating up (acting randomly).

    During heat-up we act randomly and collect trajectories in
    the replay buffer.

    Args:
        replay_buf: replay buffer
        cnt_transitions: number of transitions/steps/iterations
            played so far.
    """
    return cnt_transitions < args.heatup_transitions or \
           not replay_buf.can_sample(args.batch_size)


def run_episode(episode_number,
                replay_buf,
                cnt_transitions,
                q_behavior,
                q_target,
                optimizer,
                criterion,
                exploration,
                progress_bar):
    episode_reward = 0
    discount = 0.99
    s0 = env.reset()
    s0 = pre_process_atari(s0)
    for t in itertools.count(1):
        if args.render:
            env.render()

        heating_up = is_heating_up(replay_buf, cnt_transitions + t)
        eps = 1. if heating_up else exploration.step()

        last_idx = replay_buf.store_frame(s0)
        recent_observations = replay_buf.encode_recent_observation()
        recent_observations = torch.from_numpy(recent_observations).type(dtype).unsqueeze(0)

        action = epsilon_greedy_action(recent_observations, eps, q_behavior.n_actions, q_behavior)
        s1, reward, done, _ = env.step(encode_action(action.item()))
        s1 = pre_process_atari(s1)
        replay_buf.store_effect(last_idx, action, reward, done)
        s0 = s1
        episode_reward += reward

        if not heating_up and t % 4 == 0:
            train_on_batch(args.batch_size, q_behavior, q_target, replay_buf,
                           discount, optimizer, criterion, episode_number)  # cnt_transitions + t)

        if reward != 0:
            # Game is done, but episode may still be in progress
            progress_bar(reward)

        if done:
            break

    writer.add_scalar('epsilon', eps, episode_number)
    writer.add_scalar('episode_reward', episode_reward, episode_number)
    return episode_reward, t


def train_on_batch(batch_size, q_behavior, q_target, replay_buf, discount, optimizer, criterion, episode_number):
    optimize_model(batch_size, q_behavior, q_target, replay_buf, discount, optimizer, criterion, episode_number)
    #debug = writer.log_parameters(episode_number, q_behavior, ["conv1.weight", "conv2.weight"])


def optimize_model(batch_size, q_behavior, q_target, memory, discount, optimizer, criterion, episode_number):
    if not memory.can_sample(batch_size):
        return

    obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = memory.sample(batch_size)
    # Convert numpy nd_array to torch variables for calculation
    start_state_batch = torch.from_numpy(obs_batch).type(dtype)
    action_batch = torch.from_numpy(act_batch).long()
    reward_batch = torch.from_numpy(rew_batch).to(get_device())
    next_states_batch = torch.from_numpy(next_obs_batch).type(dtype)
    #not_done_mask = torch.from_numpy(1 - done_mask).type(dtype)
    is_terminal = torch.from_numpy(done_mask).bool()


    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    #batch = [*zip(*transitions)]

    #start_state_batch = torch.cat(batch[0]).type(dtype) # we convert to float only when we must - in order to save memory
    # action_batch = torch.cat(batch[1])
    # reward_batch = torch.tensor(batch[2], device=get_device())

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    start_state_values = q_behavior(start_state_batch).to(get_device())
    action_mask = F.one_hot(action_batch.squeeze(), q_behavior.n_actions).to(get_device())
    predicted_start_state_Q_values = (start_state_values * action_mask).sum(dim=1)
    #predicted_start_state_Q_values = start_state_values.gather(1, action_batch)

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
    #                                         batch[3])), device=device, dtype=torch.bool)
    # next_states_batch = torch.cat([s for s in batch[3]
    #                                     if s is not None])

    #next_states_batch = torch.cat(batch[3]).type(dtype)
    #is_terminal = batch[4]
    #non_final_mask = torch.tensor([not is_terminal for is_terminal in batch[4]], device=device, dtype=torch.bool)

    with torch.no_grad():
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for next_states_batch are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_action_values = q_target(next_states_batch).to(get_device())
        next_state_Q_values = next_state_action_values.max(1)[0].detach()
        next_state_Q_values[is_terminal] = 0
        # Compute the expected Q values
        discount = 0.99
        target_state_values = next_state_Q_values * discount + reward_batch

    # Compute Huber loss
    #loss = criterion(predicted_start_state_Q_values, target_state_values.unsqueeze(1))
    loss = criterion(predicted_start_state_Q_values, target_state_values)
    #torchviz.make_dot(loss, params=dict(q_behavior.named_parameters())).render("dqn_backward-3", format="png")
    writer.add_scalar('loss', loss, episode_number)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for name, param in q_behavior.named_parameters():
        if param.grad.data.abs().sum() == 0:
            debug = True
        # clip gradients
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def update_target_network(q_behavior, q_target, rate=1.):
    for target_p, behavior_p in zip(q_target.parameters(), q_behavior.parameters()):
        assert not torch.all(torch.eq(target_p, behavior_p))
    for target_p, behavior_p in zip(q_target.parameters(), q_behavior.parameters()):
        target_p.data = (1. - rate) * target_p.data + rate * behavior_p.data
    for target_p, behavior_p in zip(q_target.parameters(), q_behavior.parameters()):
        assert torch.all(torch.eq(target_p, behavior_p))

    #q_target.load_state_dict(q_behavior.state_dict())
    #pass

"""
v0 vs v4: v0 has repeat_action_probability of 0.25 (meaning 25% of the time the previous action will be used instead of the new action), while v4 has 0 (always follow your issued action)
Deterministic: a fixed frameskip of 4, while for the env without Deterministic, frameskip is sampled from (2,5) (code here)
"""
env = gym.make("Pong-v0") # ("PongDeterministic-v4")  # Pong-ram-v0 #"BreakoutDeterministic-v4"
"""
For Gym Pong n_actions == 6, which includes 4 NoOps, and 2 motion actions.
    I don't know why they did they, but we don't want to be biased to NoOps.

    We will use:
    0 - noop
    2 - up
    5 - down

    See: https://github.com/openai/atari-py/blob/master/doc/manual/manual.pdf
"""
#assert env.env.game == 'pong'
#actions_encoding = {0: 0, 1: 2, 2: 5}
#actions_encoding = {0: 2, 1: 5}
#n_actions = len(actions_encoding)


def encode_action(a):
    """Map action from 0..n to game-specific semantic action values.

    Pong-specific:
        For Gym Pong n_actions == 6, which includes 4 NoOps, and 2 motion actions.
        I don't know why they did that, but we don't want to be biased to NoOps,
        so we constrain the action space to 2 or 3 actions. We will use:
            0 - noop
            2 - up
            5 - down

        See: https://github.com/openai/atari-py/blob/master/doc/manual/manual.pdf
    """
    assert hasattr(encode_action, 'n_actions')
    if encode_action.n_actions == 2:
        actions_encoding = {0: 2, 1: 5}
    else:
        actions_encoding = {0: 0, 1: 2, 2: 5}
    return actions_encoding[a]


def DQN(args):
    # Initialize replay memory D to capacity N
    memory = ReplayBuffer(size=args.replay_mem_size, frame_history_len=4)
    exploration = LinearExplorationSchedule(args.eps_start, args.eps_end, args.eps_decay)
    #exploration = ExponentialExplorationSchedule(args.eps_start, args.eps_end, args.eps_decay)

    # Initialize action-value function Q with random weights
    D = PRE_PROCESS_OUTPUT_DIM
    n_actions = encode_action.n_actions = args.num_actions
    q_target = QNet(n_actions=n_actions,
                    n_input_ch=history_len*n_channels,
                    input_shape=(D, D)).to(get_device())
    q_behavior = QNet(n_actions=n_actions,
                      n_input_ch=history_len*n_channels,
                      input_shape=(D, D)).to(get_device())
    q_target.eval()
    # Freeze target network
    for p in q_target.parameters():
        p.requires_grad = False
    q_behavior.train()
    # Copy the weights, so both Q-value approximators initialize the same
    q_behavior.load_state_dict(q_target.state_dict())
    criterion = nn.MSELoss()
    #criterion = nn.SmoothL1Loss() # Huber loss
    # “Human-level control through deep reinforcement learning” - rmsprop config
    LEARNING_RATE = 0.00025
    ALPHA = 0.95
    EPS = 0.01
    optimizer = torch.optim.RMSprop(q_behavior.parameters(),
                                    lr=LEARNING_RATE, alpha=ALPHA, eps=EPS)  # , lr=0.00025, momentum=0.95, eps=0.01)

    reward_ema = ExponentialMovingAvg(args.reward_eam_factor)
    max_return = -np.inf
    cnt_transitions = 0

    for episode in itertools.count():
        with GameProgressBar(episode) as progress_bar:
            episode_return, n_transitions = run_episode(episode,
                                                        memory,
                                                        cnt_transitions,
                                                        q_behavior,
                                                        q_target,
                                                        optimizer,
                                                        criterion,
                                                        exploration,
                                                        progress_bar)
            reward_ema.update(episode_return)
            cnt_transitions += n_transitions

            if episode % args.target_update_rate == 0:
                update_target_network(q_behavior, q_target)

            max_return = max(max_return, episode_return)
            writer.add_scalar('running_return', reward_ema.value, episode)
            # print(f"End of episode {episode} (G={episode_return} "
            #       f"transitions={n_transitions} max_return={max_return} "
            #       f"reward_ema={reward_ema.value})")
            print('  '.join([f'reward={episode_return:.2f}',
                             f'running mean={reward_ema.value:.2f}']), end='')

        env.close()


# see https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf

# The number of frames to input to the DQN model.
# This includes the latest frame from the game, plus some previous frames.
# When history_len is set to 1, we use frame difference as our observation.
history_len = 4
n_channels = 3

from functools import partial
PRE_PROCESS_OUTPUT_DIM = 80
pre_process_atari = partial(pre_process_game_frame,
                            n_channels=3,
                            output_shape=(PRE_PROCESS_OUTPUT_DIM, PRE_PROCESS_OUTPUT_DIM))


import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('-b', '--batch-size', default=32, type=int,
                       help='mini-batch size (default: 32)')
argparser.add_argument('--render', action='store_true',
                       help='render flag')
argparser.add_argument('--heatup-transitions', default=50000, type=int)
argparser.add_argument('--replay-mem-size', default=500000, type=int)
argparser.add_argument('--learning-freq', default=1, type=int,
                       help='the number iterations between Q-value trainings')
argparser.add_argument('--target-update-rate', default=10, type=int,
                       help='the number of episodes between updates of the approximated Q*')
argparser.add_argument('--reward-eam-factor', default=0.01, type=float,
                       help='reward exponential-moving-average factor (default: 0.01)')
argparser.add_argument('--eps-start', default=1.0, type=float,
                       help='epsilon-greedy exploration schedule: start value')
argparser.add_argument('--eps-end', default=0.1, type=float,
                       help='epsilon-greedy exploration schedule: end value')
argparser.add_argument('--eps-decay', default=1000000, type=int,
                       help='the number of iterations between updates of the approximated Q*')
argparser.add_argument('--num-actions', default=3, type=int, choices=(2, 3),
                       help='the number of actions in the action space')

args = argparser.parse_args()


if __name__ == "__main__":
    args = argparser.parse_args()
    DQN(args)
