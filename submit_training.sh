#!/bin/bash
## Notes: - comments start with '#' ;
## - the first line of the script must be a 'shebang' line (i.e. '#!' followed
##     by the shell flavor for interpreting this script) ;
## - then follow Slurm sbatch directives starting with '#SBATCH'
##
#SBATCH -n 1   # request 1 main task running python or jupyter
#SBATCH -c 12  # number of cores per task (threads)
#SBATCH -t 2:30:00  # max. run time ('Walltime') in format d-HH:MM:SS
## or: #SBATCH -t MMMM  #max.time in minutes  
# #SBATCH --mail-type=ALL  #send a mail when job starts and ends
##
## - you may change the number of cores or walltime
## - also if you need more memory than the default 4GB/core, you may uncomment:
# #SBATCH -mem-per-cpu=8192  #or =8G or other value in MB
##
## - to run on a GPU node, uncomment these lines:
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100
## - if you want a fractional GPU instead of a full Nvidia A100, replace the last line with 
# #SBATCH --gres=gpu:a100_1g.5gb
##
#SBATCH --output=jobs/slurm.%j/slurm-%j.out
## - some printout:
echo "Running $SLURM_NTASKS tasks ($SLURM_CPUS_PER_TASK cores per task) on $SLURM_NODELIST"

## - prevent any loaded software to interfere
module purge

## - load the software environment 
##   here: define a flag as 'base' or some other virtual environment name to 
##     activate a conda environment, 
##     if the flag is an empty string, use system software
mysoftwaresrc=""
if [ "X$mysoftwaresrc" = "X" ]; then
    module load cuda cudnn
    module load python
else
  . ~/.bashrc
  if [ "$mysoftwaresrc" = "base" ]; then
    conda activate
  else
    conda activate $mysoftwaresrc
  fi
fi

## - limit the openMP threadpool to the number of allocated cores
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

## - optional: 
##   define a working directory (default is the folder where you start the job)
WORKDIR="$HOME/scratch/wildfires"
# e.g: WORKDIR=$HOME/scratch/test
##   or any other folder 
##   (note: the path to all input files must be absolute  or relative to WORKDIR) 
[ -d $WORKDIR ] || mkdir -p $WORKDIR
cd $WORKDIR

## - name of the python script or notebook you'd like to run:
myscript="train.ipynb"

## note: files containing spaces must be in quotes; better replace spaces with underscores!!
##       this is done with the following commands:
myscriptfixed=`echo $myscript | tr ' ' '_' `
mv "$myscript" $myscriptfixed
 
## - if $myscript is a notebook (extension .ipynb), convert it to a python script (extension .py)
##    variable 'myscriptname' is the filename without extension
if [ "${myscriptfixed##*.}" = "ipynb" ]; then
    myscriptname="${myscriptfixed%.*}"
    jupyter nbconvert --to=python $myscriptfixed
    myscriptfixed=${myscriptname}.py
fi

## - optional: get timing information (first set counter to 0):
SECONDS=0

## - run the python code:
python $myscriptfixed > "jobs/slurm.$SLURM_JOB_ID/program_output.log" 2>&1

## - optional: get timing information and print out:
Tend=$SECONDS
Ncores=`expr $SLURM_NTASKS \* $SLURM_CPUS_PER_TASK`
echo "Running for $Tend seconds on $Ncores cores on $SLURM_NODELIST"
## I hope that works!
