# distributed-pytorch - nem működik még, kipróbáláshoz ez ígéretes!
Kód a DDP tutorial sorozathoz https://pytorch.org/tutorials/beginner/ddp_series_intro.html

Minden kód fájl bővíti az előzőt. 
A sorozat egy nem elosztott szripttel kezdődik, ami egyszem GPU-n fut, és egy multinode tréninggel végződik egy Slurm fürtön (Slurm cluster)


Fájlok
single_gpu.py: nem elosztott tréning szkript

multigpu.py: DDP egy csomóponton (több gpu)

multigpu_torchrun.py: DDP egy csomóponton (több gpu) Torchrun használatával

multinode.py: DDP több csomóponton Torchrun (vagy Slurm) használatával

furcsa eltérés, hogy a torchrun-nal kell futtatni, ami egy alias, de én így hívtam:
python -m torch.distributed.run --standalone --nproc_per_node=gpu ./60_Distributed/multigpu_torchrun.py 10 100


slurm/setup_pcluster_slurm.md: instructions to set up an AWS cluster
slurm/config.yaml.template: configuration to set up an AWS cluster
slurm/sbatch_run.sh: slurm script to launch the training job


torch>=1.11.0
