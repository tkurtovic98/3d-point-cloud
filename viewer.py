import os
import numpy as np

# import pickle
import open3d as o3d

from geometric_registration.common import loadlog

# with open('/home/tomislav/Tomo/Faks/5.god/diplomski_seminar/D3Feat.pytorch/3DMatch/3DMatch_train_0.030_keypts.pkl', 'rb') as f:
#     res = pickle.load(f)
# print(len(res['analysis-by-synthesis-office2-5b/seq-01/cloud_bin_35']))

# if not os.path.exists("./clouds"):
#     os.mkdir("./clouds")

# key = "40_42"

# pcd1 = o3d.io.read_point_cloud(
#     f'/home/tomislav/Tomo/Faks/5.god/diplomski_seminar/D3Feat.pytorch/3DMatch/fragments/7-scenes-redkitchen/cloud_bin_{key.split("_")[0]}.ply')
# pcd2 = o3d.io.read_point_cloud(
#     f'/home/tomislav/Tomo/Faks/5.god/diplomski_seminar/D3Feat.pytorch/3DMatch/fragments/7-scenes-redkitchen/cloud_bin_{key.split("_")[1]}.ply')
# # res = np.load("/home/tomislav/Tomo/Faks/5.god/diplomski_seminar/D3Feat.pytorch/3DMatch/fragments/7-scenes-redkitchen/cloud_bin_0.ply", allow_pickle=True)
# # print((np.array(res)[0:10]))

# logs = loadlog("/home/tomislav/Tomo/Faks/5.god/diplomski_seminar/D3Feat.pytorch/geometric_registration/gt_result/7-scenes-redkitchen-evaluation/")

# gt_pred = np.array([[0.9959768077621137, -0.08303905127768021, 0.03368552156182647, 0.12119991314150608], [0.08857959038738494, 0.9691820723278108, -
#                    0.22986902106454757, 0.269710198615651], [-0.013559298167404976, 0.2319280635052021, 0.9726384316856569, 0.8690483909610458], [0.0, 0.0, 0.0, 1.0]])

# print(gt_pred)

# gt_trans = logs[key]

# print(gt_trans)

# pcd2_copy = o3d.io.read_point_cloud(
#     f'/home/tomislav/Tomo/Faks/5.god/diplomski_seminar/D3Feat.pytorch/3DMatch/fragments/7-scenes-redkitchen/cloud_bin_{key.split("_")[1]}.ply')

# pcd2.transform(gt_trans)

# pcd2_copy.transform(gt_pred)

# o3d.io.write_point_cloud(f'./clouds/test2_real.ply', pcd2)
# o3d.io.write_point_cloud(f'./clouds/test2_pred.ply', pcd2_copy)

pcd1 = o3d.io.read_point_cloud(
    f'/home/tomislav/Downloads/Garfield/pts3D_0.ply')

print(np.array(pcd1.points)[1:10])
