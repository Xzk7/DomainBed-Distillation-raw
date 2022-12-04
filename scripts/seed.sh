cd ..
python -u -m domainbed.scripts.collect_result_seed \
    --input_dir /data/zkxu/DG/code/DomainBed/PACS_ERMDistill_R18 --algorithm ERMDistill \
    --work_dir /data/zkxu/DG/code/DomainBed/collect_results/PACS_ERMDistill_R18 \
    --dataset PACS --test_env 0 1 2 3 4 5 --weight_mgd 7e-4 1e-4 7e-5 5e-5 --seeds 0 1 2