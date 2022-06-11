import argparse
import json
import matplotlib.pyplot as plt


def draw(filename):
    with open(filename, 'r') as f:
        res = json.load(f)

    labels = [num_points for num_points in res.keys()]

    print(labels)

    for label in labels:
        plt.plot(labels, res.values())


    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_root', default='/home/tomislav/Tomo/Faks/5.god/diplomski_seminar/D3Feat.pytorch/custom-validation/scores')
    parser.add_argument(
        '--data_file', default='7-scenes-redkitchen-result-rand.json', type=str)
    args = parser.parse_args()

    draw(args.data_root + "/" + args.data_file)
