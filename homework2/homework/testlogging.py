import torch.utils.tensorboard as tb
from homework.logger import test_logging  # assuming it's in homework/logger.py

logger = tb.SummaryWriter('cnn')
test_logging(logger)
