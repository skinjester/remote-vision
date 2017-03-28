import logging
import logtest_mylib

# Get the top-level logger object
log = logging.getLogger() # the 'root' logger

# make it print to the console.
console = logging.StreamHandler()

# set the message format
format_str = '%(asctime)s\t%(levelname)s %(processName)s %(filename)s:%(lineno)s -- %(message)s'
console.setFormatter(logging.Formatter(format_str))


log.addHandler(console)
log.setLevel(logging.DEBUG) # show anything ERROR or above
logging.Filter

# emit a warning to the puny Humans
log.info('Something was spotted near Saturn this afternoon')
log.warn('The object near Saturn has changed course for Earth')
log.error('Citizens of Earth, be warned!')
log.critical('There goes the civilization')
log.debug('We will rebuild in space this time')

## Citizens of Earth, be warned!
