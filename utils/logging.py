import os
import sys
import os.path as osp
import time
from torch.utils.tensorboard import SummaryWriter

class TensorBoardLogger:
    
    def __init__(self, fpath=None, filename=''):
        
        if not osp.exists(osp.join(fpath, filename)):
            os.makedirs(osp.join(fpath, filename))

        self.writer = SummaryWriter(osp.join(fpath, filename))
    
    def update(self, tb_dict, it):
        
        for key, value in tb_dict.items():
            self.writer.add_scalar(key, value, it)

class ConsoleLogger:
    """Write console output to external text file.
    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py>`_
    Args:
        fpath (str): directory to save logging file.
    Examples::
       >>> import sys
       >>> import os.path as osp
       >>> save_dir = 'output/experiment-1'
       >>> log_name = 'train.log'
       >>> sys.stdout = Logger(osp.join(save_dir, log_name))
    """

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            if not os.path.exists(osp.dirname(fpath)):
                os.makedirs(osp.dirname(fpath))
            self.file = open(fpath, "w")

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

def setup_logger(output=None):
    if output is None:
        return
    
    if not osp.exists(output):    
        os.makedirs(output)
    
    if output.endswith(".txt") or output.endswith(".log"):
        fpath = output
    else:
        fpath = osp.join(output, "log.txt")

    # make sure the existing log file is not over-written
    fpath += time.strftime("-%Y-%m-%d-%H-%M-%S")

    sys.stdout = ConsoleLogger(fpath)