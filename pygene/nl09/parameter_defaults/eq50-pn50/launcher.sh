#!/bin/bash

#SBATCH -J job_name -t 30
#SBATCH -N 8 --ntasks-per-node=32 --cpus-per-task=2 --gpus-per-node=1
#SBATCH --mail-user=drsmith@pppl.gov --mail-type=END,FAIL
#SBATCH -o run-%j.out

module load gene
module -l list

# verify existence of parameters and checkpoint
if [[ -s checkpoint && -s parameters ]]; then
    echo "checkpoint and parameters are valid"
else
    echo "invalid checkpoint or parameters, exiting"
    exit 1
fi

# create tmp directory, copy parameters and checkpoint
echo "preparing scratch directory"
rundir=run-$SLURM_JOB_ID
scrdir=${SCRATCH}/$rundir
mkdir $scrdir
if [[ -d $scrdir ]]; then
    echo "scrdir: ${scrdir}"
else
    echo "scrdir does not exist"
    exit 1
fi
cp parameters checkpoint $scrdir
sleep 5

# run in tmp directory
cd $scrdir
ls -lt
echo "running GENE"
srun ${GENEBIN}/gene_arch_pgi_199_debug0_cuda1.acc > gene.out
exitcode=$?
sleep 5

# check for 0 exit code
if [[ $exitcode -eq 0 ]]; then
    echo "good exitcode: ${exitcode}"
else
    echo "bad exitcode: ${exitcode}, exiting"
    exit $exitcode
fi
ls -lt

# move tmp directory to submit directory
cd $SLURM_SUBMIT_DIR
mv -f $scrdir .
if [[ $? -eq 0 ]]; then
    echo "moving output to submit directory is successful"
else
    echo "error moving output, exiting"
    exit 1
fi
sleep 5

# verify checkpoint file
cd $rundir
if [[ -s checkpoint && -s s_checkpoint && checkpoint -nt s_checkpoint ]]; then
    echo "checkpoint and s_checkpoint are valid"
else
    echo "bad checkpoint or s_checkpoint file, exiting"
    exit 1
fi
mv -f checkpoint ..
sleep 5

exit $exitcode
