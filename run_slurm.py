'''
Script for running many trials of the same algorithm with SLURM.
Edit the script according to your needs (number of trials, log folder, memory/time requested, ...).
The algorithm must take only the trial number as argument (edit the call otherwise).

python3 run_slurm <ALG_NAME>
'''

import os, errno, time, sys

name = sys.argv[1] # name of the algorithm to run

for i in range(1,51):

    text = """\
#!/bin/bash

# job name
#SBATCH -J job_name

# logfiles
#SBATCH -o mips/log/stdout_""" + name + """_""" + str(i) + """\
#SBATCH -e mips/log/stderr_""" + name + """_""" + str(i) + """\

# request computation time hh:mm:ss
#SBATCH -t 8:00:00

# request virtual memory in MB per core
#SBATCH --mem-per-cpu=1000

# nodes for a single job
#SBATCH -n 1

#SBATCH -C avx2
#SBATCH -c 4

cd mips
module load matlab
matlab -nosplash -nodesktop -nodisplay -r "INSTALL; """ + name + """(""" + str(i) + """); exit"
    """

    text_file = open('r.sh', "w")
    text_file.write(text)
    text_file.close()

    os.system('sbatch r.sh')
