from common.timer import ModuleTimer
from log import logger
import time


timer = ModuleTimer()

a = 0

for i in range(1000000):
    a += 10

timer.end('stage1')

for j in range(10500000):
    a += 10

timer.end('stage2')

logger.warning(timer.output())

for it in range(30):
    import pdb
    pdb.set_trace()
    logger.critical('test mail handler {}'.format(it))
    time.sleep(2)
