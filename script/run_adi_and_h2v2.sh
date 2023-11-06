#!/bin/bash
op=$1
if [ "$op" == "demo" ]; then
  python3 ../main.py --src_file_path=../data/trmm.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=150000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=2\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=4

elif [ "$op" == "adi_re_1" ]; then
  nohup python3 ../main.py --src_file_path=../data/adi.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=200000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=5\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=1 >../saving_log/adi_4_4_GRF_4_LRF_4_ii_5_xyx2_re_1.log 2>&1 & \


elif [ "$op" == "h2v2_re_1" ]; then
  nohup python3 ../main.py --src_file_path=../data/h2v2.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=200000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=7\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=1 >../saving_log/h2v2_4_4_GRF_4_LRF_4_ii_7_xyx2_re_1.log 2>&1 & \

elif [ "$op" == "adi_re_2" ]; then
  nohup python3 ../main.py --src_file_path=../data/adi.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=200000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=4\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=2 >../saving_log/adi_4_4_GRF_4_LRF_4_ii_4_xyx2_re_2.log 2>&1 & \


elif [ "$op" == "h2v2_re_2" ]; then
  nohup python3 ../main.py --src_file_path=../data/h2v2.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=200000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=4\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=2 >../saving_log/h2v2_4_4_GRF_4_LRF_4_ii_4_xyx2_re_2.log 2>&1 & \

elif [ "$op" == "adi_re_3" ]; then
  nohup python3 ../main.py --src_file_path=../data/adi.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=200000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=4\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=3 >../saving_log/adi_4_4_GRF_4_LRF_4_ii_4_xyx2_re_3.log 2>&1 & \


elif [ "$op" == "h2v2_re_3" ]; then
  nohup python3 ../main.py --src_file_path=../data/h2v2.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=200000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=4\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=3 >../saving_log/h2v2_4_4_GRF_4_LRF_4_ii_4_xyx2_re_3.log 2>&1 & \

elif [ "$op" == "adi_re_4" ]; then
  nohup python3 ../main.py --src_file_path=../data/adi.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=200000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=4\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=4 >../saving_log/adi_4_4_GRF_4_LRF_4_ii_4_xyx2_re_4.log 2>&1 & \


elif [ "$op" == "h2v2_re_4" ]; then
  nohup python3 ../main.py --src_file_path=../data/h2v2.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=200000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=4\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=4 >../saving_log/h2v2_4_4_GRF_4_LRF_4_ii_4_xyx2_re_4.log 2>&1 & \


elif [ "$op" == "adi_8_8" ]; then
  nohup python3 ../main.py --src_file_path=../data/adi.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=200000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=8 --mii=2\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=4 >../saving_log/adi_8_8_GRF_4_LRF_4_ii_2_xyx2_re_4.log 2>&1 & \


elif [ "$op" == "h2v2_8_8" ]; then
  nohup python3 ../main.py --src_file_path=../data/h2v2.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=200000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=8 --mii=4\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=4 >../saving_log/h2v2_8_8_GRF_4_LRF_4_ii_4_xyx2_re_4.log 2>&1 & \


elif [ "$op" == "adi_12_12" ]; then
  nohup python3 ../main.py --src_file_path=../data/adi.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=200000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=12 --mii=2\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=4 >../saving_log/adi_12_12_GRF_4_LRF_4_ii_2_xyx2_re_4.log 2>&1 & \


elif [ "$op" == "h2v2_12_12" ]; then
  nohup python3 ../main.py --src_file_path=../data/h2v2.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=200000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=12 --mii=4\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=4 >../saving_log/h2v2_12_12_GRF_4_LRF_4_ii_4_xyx2_re_4.log 2>&1 & \

elif [ "$op" == "adi_4grf_0lrf" ]; then
  nohup python3 ../main.py --src_file_path=../data/adi.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=200000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=5\
  --beta=0 --layer_nums=5 --max_LRF=0 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=1 >../saving_log/adi_4_4_GRF_4_LRF_0_ii_5_xyx2_re_1.log 2>&1 & \


elif [ "$op" == "h2v2_4grf_0lrf" ]; then
  nohup python3 ../main.py --src_file_path=../data/h2v2.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=200000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=7\
  --beta=0 --layer_nums=5 --max_LRF=0 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=1 >../saving_log/h2v2_4_4_GRF_4_LRF_0_ii_7_xyx2_re_1.log 2>&1 & \

elif [ "$op" == "adi_2grf_2lrf" ]; then
  nohup python3 ../main.py --src_file_path=../data/adi.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=200000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=5\
  --beta=0 --layer_nums=5 --max_LRF=2 --max_GRF=2 --max_memory=0 --memory_mode=False --reward_mode=1 >../saving_log/adi_4_4_GRF_2_LRF_2_ii_5_xyx2_re_1.log 2>&1 & \


elif [ "$op" == "h2v2_2grf_2lrf" ]; then
  nohup python3 ../main.py --src_file_path=../data/h2v2.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=200000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=7\
  --beta=0 --layer_nums=5 --max_LRF=2 --max_GRF=2 --max_memory=0 --memory_mode=False --reward_mode=1 >../saving_log/h2v2_4_4_GRF_2_LRF_2_ii_7_xyx2_re_1.log 2>&1 & \

elif [ "$op" == "adi_0grf_4lrf" ]; then
  nohup python3 ../main.py --src_file_path=../data/adi.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=200000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=6\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=0 --max_memory=0 --memory_mode=False --reward_mode=1 >../saving_log/adi_4_4_GRF_0_LRF_4_ii_6_xyx2_re_1.log 2>&1 & \


elif [ "$op" == "h2v2_0grf_4lrf" ]; then
  nohup python3 ../main.py --src_file_path=../data/h2v2.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=200000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=7\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=0 --max_memory=0 --memory_mode=False --reward_mode=1 >../saving_log/h2v2_4_4_GRF_0_LRF_4_ii_7_xyx2_re_1.log 2>&1 & \

elif [ "$op" == "clean" ]; then
  rm nohup.out train_reward.log test_reward.log max_train_reward.log
  rm -rf log/*
else
  echo "Please enter bash run_adi_and_h2v2.sh (The model you want to test || clean) Example: bash run_adi_and_h2v2.sh demo"
fi