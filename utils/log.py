"""Logging utilities"""


from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


def log_parameters(writer, timestep, net, param_names):
    for name, param in net.named_parameters():
        if name in param_names:
            writer.add_histogram(name+"_grad", param.grad.data, timestep)
            writer.add_histogram(name, param.data, timestep)
        if param.grad.data.sum() == 0:
            print("%d: zero grad for %s " % (timestep, name))
            #writer.add_scalar(name+"_grad", param.grad.data.sum(), t)


class TBWrapper(object):
    """Simple wrapper for Tensorboard SummaryWriter"""
    def __init__(self, experiment_name):
        now = datetime.now()  # current date and time
        now_str = now.strftime("%m-%d-%Y_%H-%M-%S")
        tb_log_dir = f"runs/{experiment_name}/{now_str}"
        self.writer = SummaryWriter(tb_log_dir)
        self.tb_log_dir = tb_log_dir

    def log_kvdict(self, kv_dict, timestep):
        """Log a dictionary of key-value pairs"""
        for k, v in kv_dict.items():
            self.writer.add_scalar(k, v, timestep)

    def add_scalar(self, k, v, timestep):
        self.writer.add_scalar(k, v, timestep)

    def log_parameters(self, timestep, net, param_names):
        """Log parameters and their gradients"""
        debug = False
        for name, param in net.named_parameters():
            if name in param_names:
                self.writer.add_histogram(name, param.data, timestep)
                self.writer.add_histogram(name+"_grad", param.grad.data, timestep)
            if param.grad.data.abs().sum() == 0:
                print("%d: zero grad for %s " % (timestep, name))
                debug = True
                #writer.add_scalar(name+"_grad", param.grad.data.sum(), timestep)
        return debug