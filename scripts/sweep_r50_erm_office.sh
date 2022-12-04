cd ..
python -m domainbed.scripts.sweep launch\
       --data_dir=/data1/zkxu/DG/dataset\
       --output_dir=/data1/zkxu/DG/code/DomainBed/OFFICE_ERM_R50\
       --command_launcher multi_gpu\
       --algorithms ERM\
       --datasets OfficeHome\
       --n_hparams 2\
       --n_trials 1\
       --seeds 0 1 2