#!/bin/bash
#SBATCH --account=monakhova
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=(6000ada|a6000)&avx2
#SBATCH --prefer=6000ada
#SBATCH --exclude=monakhova-compute-01,nikola-compute-15,bhattacharjee-compute-02
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180

set -euo pipefail
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
specialist_child=''
specialist_requeue=0
trap 'specialist_requeue=1; if [ -n "$specialist_child" ]; then kill -TERM "$specialist_child" 2>/dev/null || true; fi' USR1
trap 'if [ -n "$specialist_child" ]; then kill -TERM "$specialist_child" 2>/dev/null || true; fi' TERM
/home/jy2228/.conda/envs/convrml/bin/python -u "$@" &
specialist_child=$!
set +e
wait "$specialist_child"
specialist_status=$?
if kill -0 "$specialist_child" 2>/dev/null; then
    wait "$specialist_child"
    specialist_status=$?
fi
if [ "$specialist_requeue" -eq 1 ]; then
    /usr/local/slurm/current/bin/scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    exit 0
fi
exit "$specialist_status"
