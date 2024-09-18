#!/usr/local/bin/bash
#SBATCH --job-name=NohP
#SBATCH --nodes=1
#SBATCH --time=00:45:00
#SBATCH -o log/output.%A-%a
#SBATCH -e log/error.%A-%a
#SBATCH --account=s2190
#SBATCH --array=0-30

date
hostname
echo "---Running Simulation N="${SLURM_ARRAY_TASK_ID}"---"
python runRetrievalSimulation.py $((${SLURM_ARRAY_TASK_ID}+0))
# python runRetrievalSimulation.py 0 $((${SLURM_ARRAY_TASK_ID}+3)) &
# python runRetrievalSimulation.py 0 $((${SLURM_ARRAY_TASK_ID}+6)) &
# python runRetrievalSimulation.py 0 $((${SLURM_ARRAY_TASK_ID}+9)) &
# python runRetrievalSimulation.py 0 $((${SLURM_ARRAY_TASK_ID}+12)) &
# python runRetrievalSimulation.py 0 $((${SLURM_ARRAY_TASK_ID}+15)) &
# python runRetrievalSimulation.py 0 $((${SLURM_ARRAY_TASK_ID}+18)) &
# python runRetrievalSimulation.py 0 $((${SLURM_ARRAY_TASK_ID}+21)) &
# python runRetrievalSimulation.py 0 $((${SLURM_ARRAY_TASK_ID}+31)) &
# python runRetrievalSimulation.py 0 $((${SLURM_ARRAY_TASK_ID}+34)) &
# python runRetrievalSimulation.py 0 $((${SLURM_ARRAY_TASK_ID}+37)) &
# python runRetrievalSimulation.py 0 $((${SLURM_ARRAY_TASK_ID}+41)) &
# wait
exit 0
