cd ..
python -u -m domainbed.scripts.collect_result \
    --input_dir /data1/zkxu/DG/code/DomainBed/VLCS_ERMDistill_R18 --algorithm ERMDistill \
    --work_dir /data1/zkxu/DG/code/DomainBed/collect_results/VLCS_ERMDistill_R18\
    --dataset VLCS --test_envs 0 1 2 3 --weight_mgds 9e-5 5e-5 1e-5 \