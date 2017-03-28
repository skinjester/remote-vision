import logging
import logging.config
import logtest_mylib
import LogSettings



logging.config.dictConfig(LogSettings.LOGGING_CONFIG)
log = logging.getLogger('logtest-critical') # factory method


def main():
    log.info('Something')
    log.warn('The object')
    log.error('Citizens')
    log.critical('Civilization')
    log.debug('Space this time')
    logtest_mylib.doSomething()

if __name__ == '__main__':
    main()
