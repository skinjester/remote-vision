import time
from threading import Thread
import logging
import logging.config
import LogSettings

logging.config.dictConfig(LogSettings.LOGGING_CONFIG)
log = logging.getLogger('logtest-info') # factory method



def myfunc(i):
    log.info("sleeping 5 sec from thread #{}".format(i))
    time.sleep(5)
    log.info("finished sleeping 5 sec from thread #{}".format(i))

for i in range(10):
    threadname = 'threadname{}'.format(i)
    t = Thread(target=myfunc, name=threadname, args=(i,))
    # print "starting thread named:{}".format(t.name)
    t.start()


'''
log.info('Something')
log.warn('The object')
log.error('Citizens')
log.critical('Civilization')
log.debug('Space this time')
logtest_mylib.doSomething()
'''