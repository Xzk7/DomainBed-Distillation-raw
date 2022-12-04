cd ..
python -m domainbed.scripts.sweep launch\
       --data_dir=/data/zkxu/DG/dataset\
       --output_dir=/data/zkxu/DG/code/DomainBed/PACS_ERM_R50\
       --command_launcher multi_gpu\
       --algorithms ERM\
       --datasets PACS\
       --n_hparams 2\
       --n_trials 1\
       --seeds 0 1 2