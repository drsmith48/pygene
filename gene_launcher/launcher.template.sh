#!/bin/tcsh
#SBATCH -J {{ jobname|default('GENE', true) }}
#SBATCH -p {{ partition|default('kruskal', true) }}
#SBATCH --time={{ walltime|default('8:0:00', true) }}
#SBATCH --ntasks={{ ntasks|default('64', true) }}
#SBATCH --ntasks-per-node=32
#SBATCH --mem-per-cpu={{ mempercpu|default('2000', true) }}
#SBATCH --mail-type=ALL --mail-user=drsmith@pppl.gov
#SBATCH -o std.out -e std.err

echo "--------------------------------"
echo "--------------------------------"
echo "Cluster: ${SLURM_CLUSTER_NAME}"
echo "Partition: ${SLURM_JOB_PARTITION}"
echo "Submit directory: ${SLURM_SUBMIT_DIR}"
echo " "
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job name: ${SLURM_JOB_NAME}"
echo " "
echo "Nodes/tasks/CPUs: ${SLURM_NNODES}/${SLURM_NTASKS}/${SLURM_NPROCS}"
echo "Mem/CPU (MB): ${SLURM_MEM_PER_CPU}"
echo "--------------------------------"
echo "Node list: ${SLURM_NODELIST}"
echo "--------------------------------"
echo " "

echo "Sourcing env.csh"
source /p/gene/drsmith/env.csh

echo "Running scanscript"
set t1=`date +%s`
./scanscript --np=${SLURM_NTASKS} --ppn=32 --mps=1
set t2=`date +%s`
set delt=`expr ${t2} - ${t1}`
set hr=`expr ${delt} / 3600`
set min=`expr ${delt} % 3600 / 60`
echo "Elapsed time: ${hr} hr ${min} min"

echo "Finished"
exit