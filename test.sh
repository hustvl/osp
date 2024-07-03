export CUDA_VISIBLE_DEVICES=0
export PORT=22228

bash ./tools/dist_test.sh projects/configs/osp/osp.py \
    your_work_dir/epoch_24.pth \
    1 ${PORT} 
