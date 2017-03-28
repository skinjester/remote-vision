# logtest_myapp.py
import logging
import logtest_mylib
 
#----------------------------------------------------------------------
def main():
    """
    The main entry point of the application
    """
    logger = logging.getLogger().addHandler(logging.StreamHandler())
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()

    fileHandler = logging.FileHandler("{0}/{1}.log".format('./', 'myapp'))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
 

 
    rootLogger.info("Program started")
    # result = logtest_mylib.add(7, 8)
    # logger.info("Done!")
 
if __name__ == "__main__":
    main()