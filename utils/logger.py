import os
import time

from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter


class Logger(SummaryWriter):
    def __init__(self, log_root='', name='', logger_name=''):
        if logger_name == '':
            date = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            self.log_name = '{}_{}'.format(name, date)
            log_dir = os.path.join(log_root, self.log_name)
            super(Logger, self).__init__(log_dir, flush_secs=1)
        else:
            self.log_name = 'test_' + logger_name
            log_dir = os.path.join(log_root, self.log_name)
            super(Logger, self).__init__(log_dir, flush_secs=1)


class TbLoggerParser(object):
    def __init__(self, fname):
        self.ea = event_accumulator.EventAccumulator(fname)
        self.ea.Reload()

        self.keys_scalars = self.ea.scalars.Keys()
        self.keys_images = self.ea.images.Keys()

    def scalars(self, key):
        assert (key in self.keys_scalars)

        steps = []
        values = []
        for item in self.ea.scalars.Items(key):
            steps.append(item.step)
            values.append(item.value)

        return steps, values

    def images(self, key):
        steps = []
        images = []
        for item in self.ea.images.Items(key):
            print(item)
            steps.append(item.step)
            images.append(item.value)

        return steps, images


if __name__ == '__main__':
    parse = TbLoggerParser('../log/xxx')
    # draw gamma/(gamma-1)
    steps = parse.text('method')
    # y = [i / (i - 1) for i in values]
    # file = open('trans.txt', 'a+')
    # for i in y:
    #     file.writelines(str(i) + '\n')
    # file.close()
    print(steps)
