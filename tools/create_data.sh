export nuScene_root="/horizon-bucket/aidi_public_data/nuScenes/origin"
# export dataroot="/horizon-bucket/SD_Algorithm/12_perception_bev_hde/02_user/yiang.shi/dataset/Occpancy3D-nuScenes-V1.0"
export dataroot="/horizon-bucket/SD_Algorithm/12_perception_bev_hde/02_user/yiang.shi/dataset/occ3d_nus_v1.0"
export canbus="/horizon-bucket/SD_Algorithm/12_perception_bev_hde/02_user/yiang.shi/dataset/Occpancy3D-nuScenes-V1.0"
export version="v1.0-trainval"
# export version="v1.0-mini"

# export datamini=${dataroot}/${version}


python tools/create_data.py occ \
    --root-path ${nuScene_root} \
    --out-dir  ${dataroot} --extra-tag occtrainval \
    --canbus ${canbus} \
    --version ${version} \
    --occ-path ${dataroot}  
