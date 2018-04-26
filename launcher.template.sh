#!/bin/tcsh
#SBATCH -J {{ jobname|default('GENE', true) }}
#SBATCH -p {{ partition|default('kruskal', true) }}
#SBATCH --time={{ walltime|default('24:0:00', true) }}
#SBATCH -o std.out -e std.err
#SBATCH --mail-type=ALL --mail-user=drsmith@pppl.gov
#SBATCH --mem-per-cpu={{ mempercpu|default('2000', true) }}
#SBATCH --ntasks={{ ntasks|default('128', true) }} --ntasks-per-node=32

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

module load gene/1.3
module -l list

echo "\nRunning scanscript"
set t1=`date +%s`
{{ mpiexec_comment|default('##', true) }}mpiexec -n ${SLURM_NTASKS} ./gene_pppl_intel
{{ scanscript_comment|default('##', true) }}./scanscript --np=${SLURM_NTASKS} --ppn=32
set t2=`date +%s`
set delt=`expr ${t2} - ${t1}`
set hr=`expr ${delt} / 3600`
set min=`expr ${delt} % 3600 / 60`
echo "\nElapsed time: ${hr} hr ${min} min"

echo "\nFinished"
exit
