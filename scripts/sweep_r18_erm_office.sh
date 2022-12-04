cd ..
python -m domainbed.scripts.sweep launch\
       --data_dir=/data/zkxu/DG/dataset\
       --output_dir=/data/zkxu/DG/code/DomainBed/OFFICE_ERM_R18\
       --command_launcher multi_gpu\
       --algorithms ERM\
       --datasets OfficeHome\
       --model resnet18\
       --n_hparams 3\
       --n_trials 1