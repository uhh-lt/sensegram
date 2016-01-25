
import fileinput as fi
from string import Template
import os
import shutil
import argparse as ap
import logging as log
log.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=log.INFO)

def prepare_job(input_name,output_name,output_dir,py_script,start,end,sub_dir,no_of_files):
	with open(input_name,"r") as input:
		with open(output_name,"w") as output:
			for line in input.readlines():
				t = Template(line)
				tmp_line = t.substitute(output_dir=output_dir,script=py_script,start=start,end=end,sub_dir=sub_dir,no_of_files=no_of_files)
				output.write(tmp_line)
	

def create_job_directories(dirs):
	for dir in dirs:
		os.makedirs(dir)		

def fill_job_directories(job_script,dirs,py_script,r,no_of_files):
	i = 0
	if(len(dirs)+ 1 == len(r)):	
		for dir in dirs:
			prepare_job(input_name=job_script,output_name=dir + "/" + job_script,output_dir=dir,py_script=py_script,start=r[i],end=r[i+1],sub_dir='tmp',no_of_files=no_of_files)
			shutil.copyfile(py_script,dir + "/" + py_script)
			i = i + 1
	else:
		log.info("Problem while filling the job directories!")

def start_jobs(dirs,job_file):
	for dir in dirs:
		os.chdir(dir)
		os.system('bsub < ' + job_file)

def prepare_indices(num_of_jobs):
	i = 3000000
	r = i//num_of_jobs
	out =  [elem for elem in range(0,i,r)]
	out.append(i)
	return out


if __name__ == "__main__":
	parser = ap.ArgumentParser(description="Prepare cluster jobs for multiprocessing of the provided script.\nA directory subtree is build with one directory for each job.")	
	parser.add_argument('--base_dir',default="/home/kurse/jm18magi/sensegram/src/output",type=str, help='The subdirectory in which the directory subtree should build.')
	parser.add_argument('--py_script', type=str, help='The python script to be executed.')
	parser.add_argument('--num_of_jobs',type=int,help='The number of jobs that should be started.')
	parser.add_argument('--job_file',type=str,help='The template file for the job preparation.')
	args = parser.parse_args()
	base_dir = args.base_dir
	py_script = args.py_script
	num_of_jobs = args.num_of_jobs
	job_file = args.job_file
	r = prepare_indices(num_of_jobs)
	# Create directory names of subdirectories
	dirs = [(base_dir + "/" + str(i))for i in range(num_of_jobs)]	
	create_job_directories(dirs)

	fill_job_directories(job_file,dirs,py_script,r,50)
	start_jobs(dirs,job_file)	
	
