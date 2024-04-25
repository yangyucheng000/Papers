num_path=50
CUDA_VISIBLE_DEVICES=3 nohup python pretrain_rgcn_reuters.py --data_root data/data/reuters/ --num_path ${num_path} > log_rgcn_pretrain_reuters_1_hop_path${num_path}.txt 2>&1 &

# num_path=100
# CUDA_VISIBLE_DEVICES=3 nohup python pretrain_rgcn_reuters.py --data_root data/data/reuters/ --num_path ${num_path} > log_rgcn_pretrain_reuters_1_hop_path${num_path}.txt 2>&1 &

