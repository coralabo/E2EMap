#!/bin/bash
op=$1
if [ "$op" == "demo" ]; then
  python3 main.py --src_file_path=data/trmm.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=150000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=2\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=4


elif [ "$op" == "cholesky" ]; then
  nohup python3 main.py --src_file_path=data/cholesky.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=150000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=1\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=1 >saving_log/cholesky_4_4_GRF_4_LRF_4_ii_1_xyx2_re_1.log 2>&1 & \

elif [ "$op" == "gemm" ]; then
  nohup python3 main.py --src_file_path=data/gemm.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=150000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=1\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=1 >saving_log/gemm_4_4_GRF_4_LRF_4_ii_1_xyx2_re_1.log 2>&1 & \

elif [ "$op" == "atax" ]; then
  nohup python3 main.py --src_file_path=data/atax.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=150000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=1\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=0 --max_memory=0 --memory_mode=False --reward_mode=1 >saving_log/atax_4_4_GRF_0_LRF_4_ii_1_xyx2_re_1_4_test.log 2>&1 & \

elif [ "$op" == "syrk" ]; then
  nohup python3 main.py --src_file_path=data/syrk.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=150000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=1\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=0 --max_memory=0 --memory_mode=False --reward_mode=1 >saving_log/syrk_4_4_GRF_0_LRF_4_ii_1_xyx2_re_1_4_test.log 2>&1 & \

elif [ "$op" == "gesummv" ]; then
  nohup python3 main.py --src_file_path=data/gesummv.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=150000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=2\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=1 >saving_log/gesummv_4_4_GRF_4_LRF_4_ii_2_xyx2_re_1.log 2>&1 & \

elif [ "$op" == "gemver" ]; then
  nohup python3 main.py --src_file_path=data/gemver.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=150000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=2\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=1 >saving_log/gemver_4_4_GRF_4_LRF_4_ii_2_xyx2_re_1.log 2>&1 & \

elif [ "$op" == "doitgen" ]; then
  nohup python3 main.py --src_file_path=data/doitgen.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=150000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=1\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=1 >saving_log/doitgen_4_4_GRF_4_LRF_4_ii_1_xyx2_re_1.log 2>&1 & \

elif [ "$op" == "bicg" ]; then
  nohup python3 main.py --src_file_path=data/bicg.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=150000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=2\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=0 --max_memory=0 --memory_mode=False --reward_mode=1 >saving_log/bicg_4_4_GRF_0_LRF_4_ii_2_xyx2_re_1_4_test2.log 2>&1 & \

elif [ "$op" == "mvt" ]; then
  nohup python3 main.py --src_file_path=data/mvt.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=150000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=1\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=1 >saving_log/mvt_4_4_GRF_4_LRF_4_ii_1_xyx2_re_1.log 2>&1 & \

elif [ "$op" == "symm" ]; then
  nohup python3 main.py --src_file_path=data/symm.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=150000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=1\
  --beta=0 --layer_nums=5 --max_LRF=2 --max_GRF=2 --max_memory=0 --memory_mode=False --reward_mode=1 >saving_log/symm_4_4_GRF_2_LRF_2_ii_1_xyx2_re_1.log 2>&1 & \

elif [ "$op" == "trmm" ]; then
  nohup python3 main.py --src_file_path=data/trmm.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=150000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=2\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=1 >saving_log/trmm_4_4_GRF_4_LRF_4_ii_2_xyx2_re_1.log 2>&1 & \

elif [ "$op" == "syr2k" ]; then
  nohup python3 main.py --src_file_path=data/syr2k.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=150000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=2\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=1 >saving_log/syr2k_4_4_GRF_4_LRF_4_ii_2_xyx2_re_1.log 2>&1 & \


elif [ "$op" == "atax_u" ]; then
  nohup python3 main.py --src_file_path=data/atax_unroll.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=150000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=2\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=0 --max_memory=0 --memory_mode=False --reward_mode=1 >saving_log/atax_unroll_4_4_GRF_0_LRF_4_ii_2_xyx2_re_1_4_test.log 2>&1 & \

elif [ "$op" == "cholesky_u" ]; then
  nohup python3 main.py --src_file_path=data/cholesky_unroll.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=150000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=1\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=1 >saving_log/cholesky_unroll_4_4_GRF_4_LRF_4_ii_1_xyx2_re_1.log 2>&1 & \

elif [ "$op" == "doitgen_u" ]; then
  nohup python3 main.py --src_file_path=data/doitgen_unroll.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=150000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=3\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=0 --max_memory=0 --memory_mode=False --reward_mode=1 >saving_log/doitgen_unroll_4_4_GRF_0_LRF_4_ii_3_xyx2_re_1_4_test.log 2>&1 & \

elif [ "$op" == "gemm_u" ]; then
  nohup python3 main.py --src_file_path=data/gemm_unroll.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=150000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=2\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=1 >saving_log/gemm_unroll_4_4_GRF_4_LRF_4_ii_2_xyx2_test.log 2>&1 & \

elif [ "$op" == "mvt_u" ]; then
  nohup python3 main.py --src_file_path=data/mvt_unroll.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=150000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=2\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=1 >saving_log/mvt_unroll_4_4_GRF_4_LRF_4_ii_2_xyx2_re_1.log 2>&1 & \

elif [ "$op" == "symm_u" ]; then
  nohup python3 main.py --src_file_path=data/symm_unroll.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=150000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=2\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=1 >saving_log/symm_unroll_4_4_GRF_4_LRF_4_ii_2_xyx2_re_1.log 2>&1 & \

elif [ "$op" == "syrk_u" ]; then
  nohup python3 main.py --src_file_path=data/syrk_unroll.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=150000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=2\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=1 >saving_log/syrk_unroll_4_4_GRF_4_LRF_4_ii_2_xyx2_re_1.log 2>&1 & \

elif [ "$op" == "trmm_u" ]; then
  nohup python3 main.py --src_file_path=data/trmm_unroll.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=150000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=12 --mii=1\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=4 >saving_log/trmm_unroll_12_12_GRF_4_LRF_4_ii_1_xyx2_re_4_4.log 2>&1 & \

elif [ "$op" == "dummies" ]; then
  nohup python3 main.py --src_file_path=data/dummies.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=200000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=4\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=6 >saving_log6/dummies_4_4_GRF_4_LRF_4_ii_4_xyx2_re_6.log 2>&1 & \

elif [ "$op" == "deriche" ]; then
  nohup python3 main.py --src_file_path=data/deriche.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=200000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=3\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=6 >saving_log6/deriche_4_4_GRF_4_LRF_4_ii_3_xyx2_re_6.log 2>&1 & \

elif [ "$op" == "Inner1" ]; then
  nohup python3 main.py --src_file_path=data/Inner1.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=200000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=3\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=6 >saving_log6/Inner1_4_4_GRF_4_LRF_4_ii_3_xyx2_re_6.log 2>&1 & \

elif [ "$op" == "adi" ]; then
  nohup python3 main.py --src_file_path=data/adi.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=200000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=5\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=1 >saving_log/adi_4_4_GRF_4_LRF_4_ii_5_xyx2_re_1_2_4.log 2>&1 & \


elif [ "$op" == "h2v2" ]; then
  nohup python3 main.py --src_file_path=data/h2v2.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=200000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=7\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=1 >saving_log/h2v2_4_4_GRF_4_LRF_4_ii_7_xyx2_re_1_2_4.log 2>&1 & \

elif [ "$op" == "conv3_u" ]; then
  nohup python3 main.py --src_file_path=data/conv3_u.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=200000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=5\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=6 >saving_log/conv3_u_4_4_GRF_4_LRF_4_ii_5_xyx2_re_1.log 2>&1 & \

elif [ "$op" == "syr2k_u" ]; then
  nohup python3 main.py --src_file_path=data/syr2k_u.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=200000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=4\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=6 >saving_log/syr2k_u_4_4_GRF_4_LRF_4_ii_4_xyx2_re_1.log 2>&1 & \

elif [ "$op" == "gesummv_u" ]; then
  nohup python3 main.py --src_file_path=data/gesummv_unroll.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=200000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=5\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=6 >saving_log/gesummv_u_4_4_GRF_4_LRF_4_ii_5_xyx2_re_1.log 2>&1 & \

elif [ "$op" == "gemver_u" ]; then
  nohup python3 main.py --src_file_path=data/gemver_unroll.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=200000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=6\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=6 >saving_log/gemver_u2_4_4_GRF_4_LRF_4_ii_6_xyx2_re_1.log 2>&1 & \

elif [ "$op" == "mesa" ]; then
  nohup python3 main.py --src_file_path=data/mesa.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=200000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=5\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=6 >saving_log/mesa_4_4_GRF_4_LRF_4_ii_5_xyx2_re_1.log 2>&1 & \

elif [ "$op" == "jepg" ]; then
  nohup python3 main.py --src_file_path=data/jepg_dct.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=200000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=6\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=6 >saving_log/jepg_dct_4_4_GRF_4_LRF_4_ii_6_xyx2_re_12.log 2>&1 & \


elif [ "$op" == "clean" ]; then
  rm nohup.out train_reward.log test_reward.log max_train_reward.log
  rm -rf log/*
else
  echo "Please enter bash run.sh (The model you want to test || clean) Example: bash run.sh demo"
fi

