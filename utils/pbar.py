
import sys

class Pbar(object):
    """
    A simple util to track the progress of loop execution.
    Create a separate instance for every loop you want to track.
    Args:
        loop_size - number of iterations in the loop
        step - number of iterations between updates

    Example:
        pb = pbar.Pbar(n, 100)
        
        for i in range(n):
            < do something here >
            pb.update(i)
        pb.finish()
    """

    def __init__(self, loop_size, freq):
        """ initializes counter, sets step size
        
        Args:
            loop_size - number of steps in the loop
            freq - frequency of process updates (100 for an update on every percent, 
                    10 for an update on every 10 percent)
        """
        self.loop_size = loop_size
        self.step = loop_size/freq

    def start(self):
        print(('Progress: 0% '), end=' ')

    def update(self, current_step):
        """ updates the progressbar if it is time to. """
        if current_step % self.step == 0: 
            print(('\b\b\b\b%02d%%' % int((float(current_step)/self.loop_size)*100)), end=' ')
            sys.stdout.flush()

    def finish(self):
        print('\b\b\b\b100%')

    


