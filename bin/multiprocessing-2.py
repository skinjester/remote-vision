import time
import multiprocessing

def event_func(num):
    print '\t%r is waiting' % multiprocessing.current_process()
    event.wait()
    print '\t%r has woken up' % multiprocessing.current_process()

if __name__ == "__main__":
    event = multiprocessing.Event()

    pool = multiprocessing.Pool()

    # call event_func for every process in the pool
    a = pool.map_async(event_func, [i for i in range(pool._processes)])

    print 'main is sleeping'
    time.sleep(2)

    print 'main is setting event'
    event.set()

    pool.close()
    pool.join()