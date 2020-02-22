import torch
import os
from colorama import Fore, Back, Style


dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


"""Various decorators"""

def cached_function(func):
    """Use this wrapper for functions that run often, but always return the same result.

    After executing the decorated function once, we cache the result and henceforth
    only return this cached value.  You can take advantage of the one-time
    initialization, to run specific code once.
    """
    def fn_wrapper(*args, **kwargs):
        if fn_wrapper._cache is None:
            ret = func(*args, **kwargs)
            fn_wrapper._cache = ret
        return fn_wrapper._cache
    fn_wrapper._cache = None
    return fn_wrapper


def func_footer(freq, footer_task):
    """Run some code after the decorated function ends.

    Args:
        freq: (int) number of invocations of the decorated function between
            invocations of the `footer_task`
        footer_task: (callable) the footer function to invoke.  This
        function takes the return value of the decorated-function as its
        only input.  I.e.
            footer_task(decorated(*args, **kwargs))
    """
    def wrap(func):
        def fn_wrapper(*args, **kwargs):
            func_footer._counter += 1
            ret = wrap.func(*args, **kwargs)
            if func_footer._counter % func_footer._freq == 0:
                footer_task(ret)
            return ret
        wrap.func = func
        return fn_wrapper
    func_footer._freq = freq
    func_footer._counter = 0
    return wrap


def optimize_cuda():
    """Enable CUDNN benchmark mode for best performance.

    This is usually "safe" for image classification, regression and RL models
    as the input sizes don't change during the run.  See here:
    https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/3
    """
    torch.backends.cudnn.benchmark = True


@cached_function
def get_device(device=None):
    """Return the device used for model and backprop execution"""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        optimize_cuda()
    return device


class GameProgressBar(object):
    """A simple progress bar for Atari games with a binary outcome.

    This class is written to be used with a controlled-execution block.
    """
    def __init__(self, episode_number, is_win_func=None):
        """Args:
            episode_number: (int) episode number
            is_win_func: (callable) callable which takes as input the reward
            and determines if the game was won.  The default `is_win_func`
            assumes a win produces a positive reward.
        """
        if is_win_func is None:
            is_win_func = self.default_win_fn
        self.episode_number = episode_number
        self.is_win = is_win_func

    @staticmethod
    def default_win_fn(reward):
        return reward > 0

    def __enter__(self):
        print(f"{self.episode_number}: ", end='', flush=True)
        return self.update

    def __exit__(self, type, value, traceback):
        print()  # new-line at game end

    def update(self, reward):
        """Print the updated status of a game in episode.
        """
        print((Fore.GREEN + "-" if self.is_win(reward) else Fore.RED + ".") +
               Style.RESET_ALL, end='', flush=True)


def save_checkpoint(model, optimizer, episode, dir):
    fname = f"{dir}/checkpoint.pt"
    checkpoint = {'epoch': episode,
                  'model': model.state_dict(),
                  'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, fname)


def load_checkpoint(model, optimizer, chkpt_file, device):
    chkpt_file = os.path.expanduser(chkpt_file)
    checkpoint = torch.load(chkpt_file, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    optimizer.load_state_dict(checkpoint['optimizer'])


class ExponentialMovingAvg(object):
    """EMA helper.

    See: https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
    The exponential moving average for a series Y may be calculated recursively:
        ema (@ time=t) = new_sample if t = 1
                         (1-alpha) * ema + alpha * new_sample if t>1
    """
    def __init__(self, alpha):
        """
        Args:
            alpha: (float) a constant smoothing factor between 0 and 1. A higher
                alpha value discounts older observations faster.
        """
        self.alpha = alpha
        assert 1.0 >= alpha > 0.
        self.ema = None

    def update(self, new_sample):
        self.ema = new_sample if self.ema is None else \
            (1-self.alpha) * self.ema + self.alpha * new_sample
        return self.ema

    @property
    def value(self):
        return self.ema

    def value_mt(self):
        return self.ema