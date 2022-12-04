cd ..
CUDA_VISIBLE_DEVICES=1 python3 -m domainbed.scripts.try_train\
       --data_dir=/data/zkxu/DG/dataset \
       --algorithm ERMDistill\
       --dataset PACS\
       --output_dir=/data/zkxu/DG/code/DomainBed/ERMDistill \
       --test_envs 1