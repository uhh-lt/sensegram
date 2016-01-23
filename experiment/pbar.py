'''simple util function to track the progress of loop execution'''
#TODO rewrite into a class
import time, sys

def start_progressbar(loop_size, n_steps):
	""" starts counter, returns step size
	loop_size - number of steps in the loop
	n_steps - frequency of process updates (100 for an update on every percent, 10 for an update on every 10 percent)
	"""
	print('Progress: 0% '),
	return loop_size/n_steps
def finish_progressbar():
	print('\b\b\b\b100%')
def update_progressbar(current_step, loop_size):
	print('\b\b\b\b%02d%%' % int((float(current_step)/loop_size)*100)),
	sys.stdout.flush()
	
# usage example

# step = start_progressbar(n, 10)
# for i in range(n):
# 	do_task()
# 	if i%step == 0:
# 		update_progressbar(i, n)
# finish_progressbar()

