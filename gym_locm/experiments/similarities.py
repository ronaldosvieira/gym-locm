import argparse
import math
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prince import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def get_arg_parser() -> argparse.ArgumentParser:
    """
    Set up the argument parser.
    :return: a ready-to-use argument parser object
    """
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('files', nargs='+',
                   help='pkl choices files generated by tournament.py')
    p.add_argument('--path', '-p', '-o', default='.',
                   help='path to save result files')
    p.add_argument('--dimensions', '-d', default=3, type=int, choices=[2, 3],
                   help='amount of dimensions to reduce to with PCA')

    return p


def run():
    # read command line arguments
    arg_parser = get_arg_parser()
    args = arg_parser.parse_args()

    # create output folder if it doesn't exist
    os.makedirs(args.path, exist_ok=True)

    print("Reading data...")

    # read all csv choices files
    dfs = [pd.read_csv(file, header=[0, 1]) for file in args.files]

    # if more than one file was read, concat them
    if len(dfs) > 1:
        # concat their columns
        choices = pd.concat(dfs, axis=1, join='inner')

        # remove duplicate columns
        choices = choices.loc[:, ~choices.columns.duplicated()]
    else:
        choices = dfs[0]

    # remove columns that will not be used
    choices.drop(columns=['drafter', 'entropy'], level=0, inplace=True)

    # get list of drafters from remaining columns
    drafters = list(choices.columns.get_level_values(0).unique())

    print("Processing data...")

    # concat 1st and 2nd players' choices into a new dataframe
    temp = pd.DataFrame(index=range(len(choices.index) * 2), columns=drafters)
    for drafter in drafters:
        drafter_columns = [choices[(drafter, '1st')], choices[(drafter, '2nd')]]

        temp[drafter] = pd.concat(drafter_columns, ignore_index=True)

    # discard original dataframe in favor of the new one
    choices = temp

    print("Calculating similarities...")

    # initialize the similarities dataframe
    similarities = pd.DataFrame(index=drafters, columns=drafters)

    # populate the similarities dataframe
    total_rows = len(choices.index)
    for drafter1 in drafters:
        # the similarity of a drafter and itself is of 100%
        similarities[drafter1][drafter1] = 1.0

        for drafter2 in drafters:
            if drafter1 == drafter2:
                continue

            # calculate amount of equal choices
            equal_rows = (choices[drafter1] == choices[drafter2]).sum()

            # calculate similarity
            similarity = equal_rows / total_rows

            # update appropriate cells in the dataframe
            similarities[drafter2][drafter1] = similarity
            similarities[drafter1][drafter2] = similarity

    # save similarities dataframe to files
    similarities.to_pickle(args.path + '/similarities.pkl')
    similarities.to_csv(args.path + '/similarities.csv')

    print("Applying PCA...")

    # create mapping between choices and equidistant points in a circumference
    choices_to_points = {0: math.sin(30), 1: math.sin(120), 2: math.sin(210),
                         3: math.cos(30), 4: math.cos(120), 5: math.cos(210)}

    # double the amount of rows to store the points' x and y
    choices = pd.concat([choices, choices+3])

    # map choices to points
    choices = choices.applymap(choices_to_points.__getitem__)

    # apply PCA down to 3 or 3 dimensions
    pca = PCA(n_components=args.dimensions, random_state=824)
    coords = pca.fit_transform(choices.T)
    coords.columns = ['x', 'y', 'z'] if args.dimensions == 3 else ['x', 'y']

    print("Applying k-means...")

    # apply K-Means to original choices data, finding the optimal value of k
    # with the average silhouette method
    silhouettes, clusterings = [], []

    for k in range(2, len(drafters)):
        print(f"Trying k={k}", end="")

        kmeans = KMeans(n_clusters=k, random_state=824).fit(coords)
        silhouette = silhouette_score(coords, kmeans.labels_, random_state=824)

        silhouettes.append(silhouette)
        clusterings.append(kmeans.labels_)
        print(f", silhouette={silhouette}, labels={kmeans.labels_}")

    best_k = np.argmax(silhouettes) + 2

    labels = clusterings[best_k - 2]
    print(f"Best k: {best_k}. Labels: {labels}")

    # color the drafters according to their cluster
    all_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                  'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    coords['color'] = [all_colors[cluster_id] for cluster_id in labels]

    print("All done.")

    # rename agents
    for i in range(len(drafters)):
        tokens = drafters[i].split('/')

        if len(tokens) > 1:
            battler = {'max-attack': 'MA', 'greedy': 'GR'}[tokens[-3]]
            drafter = tokens[-2]

            drafters[i] = f"{drafter}/{battler}"

    print("columns")
    print(coords.columns)

    # normalize the axes
    for axis in coords.columns[:-1]:
        coords[axis] -= coords[axis].min()
        coords[axis] /= coords[axis].max()

    print(coords)

    if args.dimensions == 3:
        # plot the PCA coordinates in 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        objs = []

        for name, x, y, z, color in coords.itertuples():
            objs.append(ax.scatter(x, y, z, marker='o', c=color, label=name))

        plt.legend(objs, drafters, ncol=3, fontsize=8, loc='upper left')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
    else:
        # plot the PCA coordinates in 2D
        plt.subplots_adjust(bottom=0.1)

        for name, x, y, color in coords.itertuples():
            plt.scatter(x, y, marker='o', label=name, c=color)

        for label, x, y in zip(drafters, coords['x'], coords['y']):
            plt.annotate(
                label,
                xy=(x, y), xytext=(-20, 20),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.savefig(args.path + f'/similarities{"3D" if args.dimensions == 3 else ""}.png')

    plt.show()

    print("✅")


if __name__ == '__main__':
    run()
