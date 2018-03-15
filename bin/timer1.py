# Program to cancel the timer
import threading
 
def gfg():
    print("GeeksforGeeks\n")
 
timer = threading.Timer(5.0, gfg)
timer.start()
print("Cancelling timer\n")
timer.cancel()
print("Exit\n")