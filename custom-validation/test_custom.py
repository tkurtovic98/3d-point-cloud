
import argparse
import json
import os
import numpy as np
import torch
from Testset import Testset
from datasets.dataloader import get_dataloader
from models.architectures import KPFCNN
from easydict import EasyDict as edict


def generate_features(model, dloader, config, chosen_snapshot):
    dataloader_iter = dloader.__iter__()

    descriptor_path = f'{save_path}/descriptors'
    keypoint_path = f'{save_path}/keypoints'
    score_path = f'{save_path}/scores'
    if not os.path.exists(descriptor_path):
        os.mkdir(descriptor_path)
    if not os.path.exists(keypoint_path):
        os.mkdir(keypoint_path)
    if not os.path.exists(score_path):
        os.mkdir(score_path)

    # generate descriptors
    for ids in range(len(dset)):
        inputs = dataloader_iter.next()
        for k, v in inputs.items():  # load inputs to device.
            if type(v) == list:
                inputs[k] = [item.cpu() for item in v]
            else:
                inputs[k] = v.cpu()
        features, scores = model(inputs)
        pcd_size = inputs['stack_lengths'][0][0]
        pts = inputs['points'][0][:int(pcd_size)]
        features, scores = features[:int(pcd_size)], scores[:int(pcd_size)]
        np.save(f'{descriptor_path}/{ids}', features.detach().cpu().numpy().astype(np.float32))
        np.save(f'{keypoint_path}/{ids}', pts.detach().cpu().numpy().astype(np.float32))
        np.save(f'{score_path}/{ids}', scores.detach().cpu().numpy().astype(np.float32))
        print(f"Generate pts_{ids}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chosen_snapshot', default='', type=str, help='snapshot dir')
    parser.add_argument('--generate_features', default=False, action='store_true')
    args = parser.parse_args()

    config_path = f'../3DMatch_output/D3Feat/snapshot/{args.chosen_snapshot}/config.json'
    config = json.load(open(config_path, 'r'))
    config = edict(config)

    # create model
    config.architecture = [
        'simple',
        'resnetb',
    ]
    for i in range(config.num_layers-1):
        config.architecture.append('resnetb_strided')
        config.architecture.append('resnetb')
        config.architecture.append('resnetb')
    for i in range(config.num_layers-2):
        config.architecture.append('nearest_upsample')
        config.architecture.append('unary')
    config.architecture.append('nearest_upsample')
    config.architecture.append('last_unary')

    model = KPFCNN(config)
    model.load_state_dict(torch.load(f'../3DMatch_output/D3Feat/snapshot/{args.chosen_snapshot}/models/model_best_acc.pth', map_location=torch.device('cpu'))['state_dict'])
    print(f"Load weight from snapshot/{args.chosen_snapshot}/models/model_best_acc.pth")
    model.eval()

    save_path = f'./geometric_registration/{args.chosen_snapshot}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    config.root = "/home/tomislav/Downloads/Garfield"

    if args.generate_features:
        dset = Testset(downsample=config.downsample, config=config )
        dloader, _ = get_dataloader(dataset=dset,
                                    batch_size=config.batch_size,
                                    shuffle=False,
                                    num_workers=config.num_workers,
                                    )
        generate_features(model.cpu(), dloader, config, args.chosen_snapshot)
