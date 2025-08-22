#!/bin/bash
#SBATCH --job-name=SIE
#SBATCH -t 1-00:00:00                    # tiempo maximo en el cluster (D-HH:MM)
#SBATCH -o /home/araymond/%x_%j.out                 # STDOUT (A = )
#SBATCH -e /home/araymond/%x_%j.err                 # STDERR
#SBATCH --mail-type=END,FAIL         # notificacion cuando el trabajo termine o falle
#SBATCH --mail-user=afraymon@uc.cl    # mail donde mandar las notificaciones
#SBATCH --chdir=/home/araymond    # direccion del directorio de trabajo
#SBATCH --account=defaultacc
#SBATCH --partition=debug
#SBATCH --nodelist=peteroa            # forzamos ventress
#SBATCH --gres=gpu:4
#SBATCH --nodes=1                    # numero de nodos a usar
#SBATCH --ntasks-per-node=1          # numero de trabajos (procesos) por nodo
#SBATCH --cpus-per-task=8           # numero de cpus (threads) por trabajo (proceso)
#SBATCH --mem=90G
# 1. Cargar pyenv

# 2. Activar entorno pyenv
source /home/araymond/storage/torch/bin/activate
cd /home/araymond/storage/investigacion/SIE

python main.py --predictor_type lie --experience SIE --exp-dir results/lie --root-log-dir results/logs/ --epochs 2000 --arch resnet18 --equi 256 --batch-size 1024 --base-lr 1e-3 --dataset-root /home/araymond/storage/investigacion/datasets/3DIEBench --images-file ./data/train_images.npy --labels-file ./data/train_labels.npy --sim-coeff 10 --std-coeff 10 --cov-coeff 1 --mlp 2048-2048-2048 --equi-factor 0.45 --hypernetwork linear

wait

echo "Finished with job $SLURM_JOBID"
