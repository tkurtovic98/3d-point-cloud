import numpy as np
import torch.utils.data as data
import os
import open3d as o3d
from os.path import join

class Testset(data.Dataset):
    __type__ = 'descriptor'
    def __init__(self,
                downsample=0.03,
                config=None,
                ):
        self.root = config.root
        self.downsample = downsample
        self.config = config

        # contrainer
        self.points = []

        self.scene_list = [filename for filename in os.listdir(self.root) if filename.endswith(".ply")]
        self.scene_list = sorted(self.scene_list, key=lambda x: int(x[:-4].split("_")[1]))

        self.num_test = len(self.scene_list)

        print(self.scene_list)

        for point_cloud in self.scene_list:
            pcd = o3d.io.read_point_cloud(join(self.root, point_cloud))
            # pcd = o3d.geometry.PointCloud.voxel_down_sample(pcd, voxel_size=downsample)

            # Load points and labels
            points = np.array(pcd.points)

            self.points += [points]

    def __getitem__(self, index):
        pts = self.points[index].astype(np.float32)
        feat = np.ones_like(pts[:, :1]).astype(np.float32)
        return pts, pts, feat, feat, np.array([]), np.array([])

    def __len__(self):
        return self.num_test
