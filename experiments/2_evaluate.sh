cd ../src || exit

python3 evaluate.py casia-b \
                 ../save/supcon_casia-b_models/gaitgraph_resgcn-n39-r8_coco_seq_60.pth \
                 ../data/casia-b_pose_test.csv \
                 --network_name resgcn-n39-r8
