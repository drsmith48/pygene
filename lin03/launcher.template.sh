#!/bin/bash

#SBATCH --job-name=gene_scan
#SBATCH --account=pppl

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96

#SBATCH --time=1-0

#SBATCH --mem-per-cpu=7500M

#SBATCH --mail-type=ALL
#SBATCH --mail-user=drsmith@pppl.gov


echo "Submit directory: ${SLURM_SUBMIT_DIR}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job name: ${SLURM_JOB_NAME}"
echo "Nodes/tasks/CPUs: ${SLURM_NNODES}/${SLURM_NTASKS}/${SLURM_NPROCS}"
echo "Mem/CPU (MB): ${SLURM_MEM_PER_CPU}"
echo "Node list: ${SLURM_NODELIST}"

module -l list

GENE_HOME=${HOME}/gene/genecode
GENE_EXEC=${GENE_HOME}/bin/gene_stellar
GENE_SCANSCRIPT=${GENE_HOME}/tools/perl/scanscript

echo "GENE executable: ${GENE_EXEC}"
echo "GENE scanscript: ${GENE_SCANSCRIPT}"

# setup work area on scratch
RUN_DIR=${HOME}/gene/scratch/job_${SLURM_JOB_ID}
echo "Creating and moving to work area: ${RUN_DIR}"
mkdir ${RUN_DIR}
cp parameters ${RUN_DIR}
ls -l ${RUN_DIR}
cd ${RUN_DIR}
echo "PWD: ${PWD}"


echo "Running scanscript"
t1=`date +%s`

${GENE_SCANSCRIPT} \
  --n_pes=${SLURM_NTASKS} \
  --min_procs=12 \
  --syscall="srun ${GENE_EXEC}" \
  --verbose

EXIT_CODE=$?

t2=`date +%s`
delt=`expr ${t2} - ${t1}`
hr=`expr ${delt} / 3600`
min=`expr ${delt} % 3600 / 60`
echo "Elapsed time: ${hr} hr ${min} min"

# move results to tigress
echo "Moving results to submit dir: ${SLURM_SUBMIT_DIR}"
mv scanfiles0000/ ${SLURM_SUBMIT_DIR}
MV_EXIT_CODE=$?

cd ${SLURM_SUBMIT_DIR}

if [[ ${MV_EXIT_CODE} -eq 0 ]]; then
  rm -rf ${RUN_DIR}
fi


echo "Finished with exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
