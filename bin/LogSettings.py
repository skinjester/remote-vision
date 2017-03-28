LOGGING_CONFIG = {
    'version': 1, # required
    'disable_existing_loggers': True, # this config overrides all other loggers
    'formatters': {
        'simple': {
            'format': '%(relativeCreated)010.2f %(message)s'
        },
        'detailed': {
            'format': '%(relativeCreated)010.2f [%(funcName)-s] %(message)s'
        },
        'showthread': {
            'format': '%(relativeCreated)010.2f %(threadName)-s [%(funcName)s]: %(message)s'
        }

    },
    'handlers': {
        'console-simple': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'simple'
        },
        'console-detailed': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'detailed'
        },
        'console-thread-detail': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'showthread'
        }
    },
    'loggers': {
        'logtest-simple': {
            'level': 'INFO',
            'handlers': ['console-simple']
        },
        'logtest-debug': {
            'level': 'DEBUG',
            'handlers': ['console-detailed']
        },
        'logtest-debug-thread': {
            'level': 'DEBUG',
            'handlers': ['console-thread-detail']
        }
    }
}

'''
\t%(funcName)s\t%(threadName)s\t%(processName)s\t%(filename)s:%(lineno)s\t%(message)s
'''