cd ..
python -m domainbed.scripts.sweep launch\
       --data_dir=/data1/zkxu/DG/dataset\
       --output_dir=/data1/zkxu/DG/code/DomainBed/DOMAIN_ERM_R50\
       --command_launcher multi_gpu\
       --algorithms ERM\
       --datasets DomainNet\
       --n_hparams 2\
       --n_trials 1 \
       --single_test_envs \
       --seeds 0 1 2