from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def log_metrics(self, metrics, step):
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, step)

    def close(self):
        self.writer.close()