export CUDA_VISIBLE_DEVICES=0,1,2
export WORKDIR="./work_dirs"
export PORT=29990

bash ./tools/dist_train.sh projects/configs/osp/osp_minibatch.py 3 ${PORT} \
    --work-dir ${WORKDIR}