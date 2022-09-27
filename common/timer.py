import time
from collections import OrderedDict

class ModuleTimer(object):

    def __init__(self):
        self.last_time = time.time()
        self.module_times = OrderedDict()

    def end(self, name):
        end_time = time.time()
        if self.last_time is not None:
            self.module_times.setdefault(name, 0)
            self.module_times[name] += end_time - self.last_time
        self.last_time = end_time

    def output(self, sep=' | '):
        total = sum(self.module_times.values())
        ret = 'total: {:.2f}ms'.format(total * 1000) + sep
        ret += sep.join('{}: {:.2f}ms ({:.0f}%)'.format(k, v*1000, v*100/total) for k, v in self.module_times.items())
        return ret
