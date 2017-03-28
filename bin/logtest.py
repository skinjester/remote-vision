import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("ex")

log.debug("This is a debug message")
log.info("Informational message")
log.error("An error has happened!")

try:
    print 'lolwat'
    #raise RuntimeError
except Exception, err:
    logging.exception("Error!")