cd ..
python -u -m domainbed.scripts.collect_result_seed \
    --input_dir /data/zkxu/DG/code/DomainBed/PACS_ERMDistill_R18 --algorithm ERMDistill \
    --work_dir /data/zkxu/DG/code/DomainBed/collect_results/PACS_ERMDistill_R18_wsld_ \
    --dataset PACS --test_env 0 1 2 3  --distillation_type wsld --weight_wslds 3.0 2.8 2.6 2.5 2.4 2.2 2.0 --seeds 0 1 2