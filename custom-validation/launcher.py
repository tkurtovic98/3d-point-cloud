import argparse
import json
import os
import random
import numpy as np
from common import build_correspondence, get_desc, get_keypts, loadinfo, loadlog
from trans_matrix import trans_matrix
from score import validate_score

scores_path = "./scores"
matrix_path = "./matrix"


def write_to_file(filename: str, T: np.ndarray, score: float) -> None:
    with open(matrix_path + "/"+filename, "w") as f:
        f.write(f'Score: {score}\n')
        f.write(f'T pred \n')
        json.dump(T.tolist(), f)


def launch(dataset_root, save_path, fragment_dir):
    scenes = [filename for filename in os.listdir(fragment_dir)]
    scenes.sort()

    keypts_map = {}
    desc_map = {}

    num_points_list = [5000, 2000, 1000, 750, 500, 250]
    # num_points_list = [500, 250]
    score_thres = args.scores_thres

    if not os.path.exists(scores_path):
        os.mkdir(scores_path)

    if not os.path.exists(matrix_path):
        os.mkdir(matrix_path)

    for scene in scenes:
        scoresDict = dict.fromkeys(num_points_list, [])
        keyptspath = f"{save_path}/keypoints/{scene}"
        descpath = f"{save_path}/descriptors/{scene}"
        scorepath = f"{save_path}/scores/{scene}"
        gtpath = f'{args.project_root}/geometric_registration/gt_result/{scene}-evaluation/'
        gtLog = loadlog(gtpath)
        gtInfo = loadinfo(gtpath)
        pcdpath = f"{dataset_root}/fragments/{scene}/"
        num_frag = len([filename for filename in os.listdir(
            pcdpath) if filename.endswith('ply')])

        pair_count = 0

        for id1 in range(num_frag):
            for id2 in range(id1 + 1, num_frag):
                cloud_bin_s = f'cloud_bin_{id1}'
                cloud_bin_t = f'cloud_bin_{id2}'
                key = f"{id1}_{id2}"
                if key in gtLog.keys():
                    pair_count += 1
                    print("KEY ", key)

                    cached_keypts = keypts_map.get(cloud_bin_s)
                    source_keypts = cached_keypts if cached_keypts is not None else get_keypts(
                        keyptspath, cloud_bin_s)

                    cached_keypts = keypts_map.get(cloud_bin_t)
                    target_keypts = cached_keypts if cached_keypts is not None else get_keypts(
                        keyptspath, cloud_bin_t)

                    cached_desc = desc_map.get(cloud_bin_s)
                    source_desc = cached_desc if cached_desc is not None else get_desc(
                        descpath, cloud_bin_s, 'D3Feat')
                    source_desc = source_desc if cached_desc is not None else np.nan_to_num(
                        source_desc)

                    cached_desc = desc_map.get(cloud_bin_t)
                    target_desc = cached_desc if cached_desc is not None else get_desc(
                        descpath, cloud_bin_t, 'D3Feat')
                    target_desc = target_desc if cached_desc is not None else np.nan_to_num(
                        target_desc)

                    if(keypts_map.get(cloud_bin_s) is None):
                        keypts_map[cloud_bin_s] = source_keypts
                    if(keypts_map.get(cloud_bin_t) is None):
                        keypts_map[cloud_bin_t] = target_keypts

                    if(desc_map.get(cloud_bin_s) is None):
                        desc_map[cloud_bin_s] = source_desc
                    if(desc_map.get(cloud_bin_t) is None):
                        desc_map[cloud_bin_t] = target_desc

                    for num_points in num_points_list:
                        print("Num points: ", num_points)
                        source_indices = np.random.choice(
                            range(source_keypts.shape[0]), num_points)
                        target_indices = np.random.choice(
                            range(target_keypts.shape[0]), num_points)

                        source_keypts = source_keypts[source_indices, :]
                        source_desc = source_desc[source_indices, :]
                        target_keypts = target_keypts[target_indices, :]
                        target_desc = target_desc[target_indices, :]

                        corr = build_correspondence(source_desc, target_desc)

                        T = trans_matrix(source_keypts, target_keypts, corr)

                        score = validate_score(T, gtLog, gtInfo, key)

                        if(score < score_thres):
                            newPoints = [
                                score_item for score_item in scoresDict[num_points]]
                            newPoints.append(score)
                            scoresDict[num_points] = newPoints
                            write_to_file(
                                f'{scene}_{key}_{num_points}_correct', T, score)
                        elif random.randint(a=0, b=100) > 95:
                            write_to_file(
                                f'{scene}_{key}_{num_points}_wrong', T, score)

                    if pair_count % 10 == 0:
                        with open(f'{scores_path}/{scene}-result.json', 'w') as f:
                            json.dump(scoresDict, f, indent=4)
                            print("Saved result")
        with open(f'{scene}-recall', "w+") as f:
            f.write(f'Scene {scene} \n')
            for num_points, scores in scoresDict.items():
                recall = len(scores) / len(gtLog.keys())
                f.write("\n")
                f.write(f'Num points: {num_points}, recall {recall}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--project_root', default='/home/tomislav/Tomo/Faks/5.god/diplomski_seminar/D3Feat.pytorch/')
    parser.add_argument(
        '--dataset_root', default='/home/tomislav/Tomo/Faks/5.god/diplomski_seminar/D3Feat.pytorch/3DMatch', type=str)
    parser.add_argument('--feature_save_path',
                        default='/home/tomislav/Tomo/Faks/5.god/diplomski_seminar/D3Feat.pytorch/geometric_registration/D3Feat05121833', type=str)
    args = parser.parse_args()
    args.scores_thres = 0.2 * 0.2

    launch(args.dataset_root, args.feature_save_path,
           args.dataset_root + "/fragments")
