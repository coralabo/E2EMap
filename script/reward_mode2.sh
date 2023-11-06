#!/bin/bash

echo "start......"


  nohup python3 ../main.py --src_file_path=../data/cholesky.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=150000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=1\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=2 >../saving_log/saving_log_re_2/cholesky_4_4_GRF_4_LRF_4_ii_1_xyx2_re_2.log 2>&1 & \


  nohup python3 ../main.py --src_file_path=../data/gemm.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=150000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=1\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=2 >../saving_log/saving_log_re_2/gemm_4_4_GRF_4_LRF_4_ii_1_xyx2_re_2.log 2>&1 & \


  nohup python3 ../main.py --src_file_path=../data/atax.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=150000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=1\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=2 >../saving_log/saving_log_re_2/atax_4_4_GRF_4_LRF_4_ii_1_xyx2_re_2.log 2>&1 & \


  nohup python3 ../main.py --src_file_path=../data/syrk.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=150000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=1\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=2 >../saving_log/saving_log_re_2/syrk_4_4_GRF_4_LRF_4_ii_1_xyx2_re_2.log 2>&1 & \
wait
echo "finish cholesky......"
echo "finish gemm......"
echo "finish atax......"
echo "finish syrk......"

  nohup python3 ../main.py --src_file_path=../data/gesummv.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=150000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=2\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=2 >../saving_log/saving_log_re_2/gesummv_4_4_GRF_4_LRF_4_ii_2_xyx2_re_2.log 2>&1 & \


  nohup python3 ../main.py --src_file_path=../data/gemver.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=150000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=2\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=2 >../saving_log/saving_log_re_2/gemver_4_4_GRF_4_LRF_4_ii_2_xyx2_re_2.log 2>&1 & \


  nohup python3 ../main.py --src_file_path=../data/doitgen.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=150000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=1\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=2 >../saving_log/saving_log_re_2/doitgen_4_4_GRF_4_LRF_4_ii_1_xyx2_re_2.log 2>&1 & \


  nohup python3 ../main.py --src_file_path=../data/bicg.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=150000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=2\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=2 >../saving_log/saving_log_re_2/bicg_4_4_GRF_4_LRF_4_ii_2_xyx2_re_2.log 2>&1 & \
wait
echo "finish gesummv......"
echo "finish gemver......"
echo "finish doitgen......"
echo "finish bicg......"

  nohup python3 ../main.py --src_file_path=../data/mvt.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=150000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=1\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=2 >../saving_log/saving_log_re_2/mvt_4_4_GRF_4_LRF_4_ii_1_xyx2_re_2.log 2>&1 & \


  nohup python3 ../main.py --src_file_path=../data/symm.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=150000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=1\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=2 >../saving_log/saving_log_re_2/symm_4_4_GRF_4_LRF_4_ii_1_xyx2_re_2.log 2>&1 & \


  nohup python3 ../main.py --src_file_path=../data/trmm.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=150000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=2\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=2 >../saving_log/saving_log_re_2/trmm_4_4_GRF_4_LRF_4_ii_2_xyx2_re_2.log 2>&1 & \


  nohup python3 ../main.py --src_file_path=../data/syr2k.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=150000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=2\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=2 >../saving_log/saving_log_re_2/syr2k_4_4_GRF_4_LRF_4_ii_2_xyx2_re_2.log 2>&1 & \
wait
echo "finish mvt......"
echo "finish symm......"
echo "finish trmm......"
echo "finish syr2k......"

  nohup python3 ../main.py --src_file_path=../data/atax_unroll.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=150000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=2\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=2 >../saving_log/saving_log_re_2/atax_unroll_4_4_GRF_4_LRF_4_ii_2_xyx2_re_2.log 2>&1 & \


  nohup python3 ../main.py --src_file_path=../data/cholesky_unroll.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=150000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=1\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=2 >../saving_log/saving_log_re_2/cholesky_unroll_4_4_GRF_4_LRF_4_ii_1_xyx2_re_2.log 2>&1 & \


  nohup python3 ../main.py --src_file_path=../data/doitgen_unroll.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=150000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=2\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=2 >../saving_log/saving_log_re_2/doitgen_unroll_4_4_GRF_4_LRF_4_ii_2_xyx2_re_2.log 2>&1 & \


  nohup python3 ../main.py --src_file_path=../data/gemm_unroll.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=150000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=2\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=2 >../saving_log/saving_log_re_2/gemm_unroll_4_4_GRF_4_LRF_4_ii_2_xyx2_re_2.log 2>&1 & \
wait
echo "finish atax_unroll......"
echo "finish cholesky_unroll......"
echo "finish doitgen_unroll......"
echo "finish gemm_unroll......"

  nohup python3 ../main.py --src_file_path=../data/mvt_unroll.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=150000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=2\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=2 >../saving_log/saving_log_re_2/mvt_unroll_4_4_GRF_4_LRF_4_ii_2_xyx2_re_2.log 2>&1 & \


  nohup python3 ../main.py --src_file_path=../data/symm_unroll.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=150000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=2\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=2 >../saving_log/saving_log_re_2/symm_unroll_4_4_GRF_4_LRF_4_ii_2_xyx2_re_2.log 2>&1 & \


  nohup python3 ../main.py --src_file_path=../data/syrk_unroll.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=150000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=2\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=2 >../saving_log/saving_log_re_2/syrk_unroll_4_4_GRF_4_LRF_4_ii_2_xyx2_re_2.log 2>&1 & \


  nohup python3 ../main.py --src_file_path=../data/trmm_unroll.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=150000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=2\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=2 >../saving_log/saving_log_re_2/trmm_unroll_4_4_GRF_4_LRF_4_ii_2_xyx2_re_2.log 2>&1 & \
wait
echo "finish mvt_unroll......"
echo "finish symm_unroll......"
echo "finish syrk_unroll......"
echo "finish trmm_unroll......"

echo "finish all......"