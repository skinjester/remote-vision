# this program manages several workers who consume data
# from a JoinableQueue and pass results back to the parent process
# the "poison pill" technique is used to stop the workers
# after setting up the real tasks, the main program adds
# one "stop" value per worker to the joinableQueue
# each of the workers adds its results to a normal (non joinable)
# multiprocessing queue the main process uses the joinableQueue's
# join() method to wait for all the tasks to finish before
# reading out the results from the results queue

def main():
    """ Main entry point of the app """
    print "hello world"


if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()

import multiprocessing
import time

class Consumer(multiprocessing.Process):
    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                print '%s: Exiting' % proc_name
                self.task_queue.task_done()
                break
            print '%s: %s' % (proc_name, next_task)
            answer = next_task()
            self.task_queue.task_done()
            self.result_queue.put(answer)
        return

class Task(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def __call__(self):
        time.sleep(0.1) # pretend to take some time to do the work
        return '%s * %s = %s' % (self.a, self.b, self.a * self.b)
    def __str__(self):
        return '%s * %s' % (self.a, self.b)

if __name__ == '__main__':
    # Establish communication queues
    task_queue = multiprocessing.JoinableQueue()
    results_queue = multiprocessing.Queue()

    # Start consumers
    num_consumers = multiprocessing.cpu_count() * 2
    print 'Creating %d consumers' % num_consumers
    consumers = [ Consumer(task_queue, results_queue)
                  for i in xrange(num_consumers) ]

    for w in consumers:
        w.start()

    # Enqueue jobs
    num_jobs = 17
    for i in xrange(num_jobs):
        task_queue.put(Task(i, i))

    # Add a poison pill for each consumer
    for i in xrange(num_consumers):
        task_queue.put(None)

    # Wait for all of the tasks to finish
    task_queue.join()

    # Print results from the queue until done
    while num_jobs:
        result = results_queue.get()
        print 'Result:', result
        num_jobs -= 1


