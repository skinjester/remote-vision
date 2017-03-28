import logging
import logging.config
import LogSettings

logging.config.dictConfig(LogSettings.LOGGING_CONFIG)
log = logging.getLogger('logtest-info') # factory method

def doSomething():
    log.info('Something')
    log.warn('The object')
    log.error('Citizens')
    log.critical('Civilization')
    log.debug('Space this time')
