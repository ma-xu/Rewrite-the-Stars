""""
Example to visualize the decision boundary of 2d points.


Use cases:
python svc.py --kernel poly --coef0 4 --degree 4 --gamma scale --dataset_noise 0.4
python svc.py --kernel rbf --coef0 4 --degree 4 --gamma 4 --dataset_noise 0.4

Please feel free to adjust the hyper-parameters to check different results.

Created by: Xu Ma (Email: ma.xu1@northeastern.edu)
Modified Date: Mar/29/2024
"""

import argparse
from sklearn import svm
from create_dataset import MoonDataset, ValDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--dataset_noise', default=0.4, type=float)
    parser.add_argument('--dataset_samples', default=100, type=int)
    parser.add_argument('--dataset_random_state', default=0, type=int)
    parser.add_argument('--kernel', default="rbf")
    parser.add_argument('--coef0', default=4, type=int)
    parser.add_argument('--degree', default=4, type=int)
    parser.add_argument('--gamma', default='scale')  # default='scale'
    return parser.parse_args()

def main():
    args = parse_args()
    train_dataset = MoonDataset(n_samples=args.dataset_samples, noise=args.dataset_noise, random_state = args.dataset_random_state )
    data = train_dataset.data
    label =train_dataset.labels
    clf = svm.SVC(kernel=args.kernel, degree=args.degree, coef0=args.coef0, gamma=args.gamma)
    clf.fit(data, label)
    predicts = clf.predict(data)
    correct = (label ==predicts).sum()
    acc = 100*correct/len(predicts)
    print(acc)

    val_data = ValDataset().data
    val_data = val_data.numpy()
    predicts = clf.predict(val_data)
    print(predicts.shape)

    color0 = (0.9999, 0.91, 0.99999)
    color1 = (0.8, 0.9999, 0.8)
    color2 = (0 / 255.0, 204 / 255.0, 204 / 255.0)
    color_list = []
    for i in predicts:
        if i == 0:
            color_list.append(color0)
        elif i == 1:
            color_list.append(color1)
        else:
            color_list.append(color2)
    plt.scatter(val_data[:, 0], val_data[:, 1], label='Scatter Plot', c=color_list, marker='s', s=7)
    train_data = train_dataset.data
    train_labels = train_dataset.labels
    colors = ["orange", "green", "pink"]
    train_colors = []
    for i in train_labels:
        train_colors.append(colors[i])
    plt.scatter(train_data[:, 0], train_data[:, 1], label='Scatter Plot', c=train_colors, marker='o')
    # Remove axis lines and ticks
    # plt.subplots_adjust(left=0, right=0, top=0, bottom=0)
    plt.axis('off')

    # Make the plot area tight
    plt.margins(0, 0)
    plt.savefig(f'{args.kernel}SVM-G{args.gamma}D{args.degree}C{args.coef0}-acc{str(acc)}-Dnoise{args.dataset_noise}-Dsamples{args.dataset_samples}.png', bbox_inches='tight')
    plt.close()



if __name__ == '__main__':
    main()
