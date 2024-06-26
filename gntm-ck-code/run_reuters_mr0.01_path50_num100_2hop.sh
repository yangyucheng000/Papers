topics=(20 30 50)
hop_num=2
edge_threshold=0
gcn_epoch=2000
num_path=50
mr_ratio=0.01
num_neigh=100

for K in ${topics[@]}
do
    for round_index in `seq 0 4`
    do
        python main.py \
        --device cuda:2 \
        --dataset Reuters \
        --model GDGNNMODEL \
        --num_topic ${K} \
        --num_epoch 400 \
        --ni 300  \
        --nw 300 \
        --hop_num ${hop_num} \
        --edge_threshold ${edge_threshold} \
        --word \
        --taskid ${round_index} \
        --gcn_epoch ${gcn_epoch} \
        --use_mr \
        --mr_ratio ${mr_ratio} \
        --num_neigh ${num_neigh} \
        --num_path ${num_path} \
        --nwindow 5
    done
done
