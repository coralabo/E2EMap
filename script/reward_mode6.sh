echo "start......"

  nohup python3 ../main.py --src_file_path=../data/dummies.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=200000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=4\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=6 >../saving_log/saving_log_re_6/dummies_4_4_GRF_4_LRF_4_ii_4_xyx2_re_6.log 2>&1 & \


  nohup python3 ../main.py --src_file_path=../data/deriche.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=200000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=3\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=6 >../saving_log/saving_log_re_6/deriche_4_4_GRF_4_LRF_4_ii_3_xyx2_re_6.log 2>&1 & \


  nohup python3 ../main.py --src_file_path=../data/Inner1.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=200000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=3\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=6 >../saving_log/saving_log_re_6/Inner1_4_4_GRF_4_LRF_4_ii_3_xyx2_re_6.log 2>&1 & \


  nohup python3 ../main.py --src_file_path=../data/adi.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=200000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=4\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=6 >../saving_log/saving_log_re_6/adi_4_4_GRF_4_LRF_4_ii_4_xyx2_re_6.log 2>&1 & \
wait
echo "finish dummies......"
echo "finish deriche......"
echo "finish Inner1......"
echo "finish adi......"

  nohup python3 ../main.py --src_file_path=../data/h2v2.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=200000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=4\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=6 >../saving_log/saving_log_re_6/h2v2_4_4_GRF_4_LRF_4_ii_4_xyx2_re_6.log 2>&1 & \


  nohup python3 ../main.py --src_file_path=../data/conv3_u.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=200000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=5\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=6 >../saving_log/saving_log_re_6/conv3_u_4_4_GRF_4_LRF_4_ii_5_xyx2_re_6.log 2>&1 & \


  nohup python3 ../main.py --src_file_path=../data/gesummv_unroll.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=200000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=5\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=6 >../saving_log/saving_log_re_6/gesummv_u_4_4_GRF_4_LRF_4_ii_5_xyx2_re_6.log 2>&1 & \


  nohup python3 ../main.py --src_file_path=../data/gemver_unroll.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=200000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=6\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=6 >../saving_log/saving_log_re_6/gemver_u2_4_4_GRF_4_LRF_4_ii_6_xyx2_re_6.log 2>&1 & \
wait
echo "finish h2v2......"
echo "finish conv3_u......"
echo "finish gesummv_unroll......"
echo "finish gemver_unroll......"

  nohup python3 ../main.py --src_file_path=../data/mesa.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=200000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=5\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=6 >../saving_log/saving_log_re_6/mesa_4_4_GRF_4_LRF_4_ii_5_xyx2_re_6.log 2>&1 & \


  nohup python3 ../main.py --src_file_path=../data/jepg_dct.txt --actor_lr=0.00003 --gcn_dims=256 \
  --max_iteration=200000 --warmup_iteration=100 --batch_size=4 --temperatre=10 --pea_width=4 --mii=6\
  --beta=0 --layer_nums=5 --max_LRF=4 --max_GRF=4 --max_memory=0 --memory_mode=False --reward_mode=6 >../saving_log/saving_log_re_6/jepg_dct_4_4_GRF_4_LRF_4_ii_6_xyx2_re_6.log 2>&1 & \
wait
echo "finish mesa......"
echo "finish jepg_dct......"


echo "finish all......"