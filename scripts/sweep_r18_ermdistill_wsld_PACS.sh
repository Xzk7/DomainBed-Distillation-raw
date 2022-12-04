cd ..
python -m domainbed.scripts.sweep launch\
       --data_dir=/data/zkxu/DG/dataset\
       --output_dir=/data/zkxu/DG/code/DomainBed/PACS_ERMDistill_R18\
       --input_dir=/data/zkxu/DG/code/DomainBed/PACS_ERM_R50\
       --command_launcher multi_gpu\
       --algorithms ERMDistill\
       --datasets PACS\
       --distillation_types wsld \
       --weight_wslds 3.5 4.0 \
       --n_hparams 2\
       --single_test_envs \
       --n_hparams_from 0\
       --n_trials 1 \
       --seeds 0 1 2