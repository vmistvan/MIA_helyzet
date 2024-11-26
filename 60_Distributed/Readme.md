# distributed-pytorch
Kód a DDP tutorial sorozathoz https://pytorch.org/tutorials/beginner/ddp_series_intro.html

Minden kód fájl bővíti az előzőt. 
A sorozat egy nem elosztott szripttel kezdődik, ami egyszem GPU-n fut, és egy multinode tréninggel végződik egy Slurm fürtön (Slurm cluster)


Fájlok
single_gpu.py: nem elosztott tréning szkript

multigpu.py: DDP egy csomóponton (több gpu)

multigpu_torchrun.py: DDP egy csomóponton (több gpu) Torchrun használatával

multinode.py: DDP több csomóponton Torchrun (vagy Slurm) használatával

slurm/setup_pcluster_slurm.md: instructions to set up an AWS cluster
slurm/config.yaml.template: configuration to set up an AWS cluster
slurm/sbatch_run.sh: slurm script to launch the training job
