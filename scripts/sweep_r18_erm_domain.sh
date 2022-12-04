cd ..
python -m domainbed.scripts.sweep launch\
       --data_dir=/data1/zkxu/DG/dataset\
       --output_dir=/data1/zkxu/DG/code/DomainBed/DOMAIN_ERM_R18\
       --command_launcher multi_gpu\
       --algorithms ERM\
       --datasets DomainNet\
       --model resnet18\
       --n_hparams 2\
       --single_test_envs \
       --n_trials 1