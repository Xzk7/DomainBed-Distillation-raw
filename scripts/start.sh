cd ..
CUDA_VISIBLE_DEVICES=0 python3 -m domainbed.scripts.train\
       --data_dir=/data/zkxu/DG/dataset\
       --output_dir=/data/zkxu/DG/code/DomainBed/Domain_ERM_R50\
       --algorithm ERM\
       --dataset DomainNet\
       --test_envs 1 \
       --hparams_seed 2 \
       --checkpoint_freq 1