import torch
file_path = './ckpts/bevformer_v4.pth'
file_path = '/horizon-bucket/SD_Algorithm/12_perception_bev_hde/02_user/yiang.shi/flexible_occ_log/v1_298/cluster/cluster_v1_298_baselinedebug_addstage/epoch_24.pth'
model = torch.load(file_path, map_location='cpu')
all = 0
for key in list(model['state_dict'].keys()):
    all += model['state_dict'][key].nelement()
print(all)

# smaller 63374123
# v4 69140395
# bev_baseline : 58926586
# v1_298: 51027024
