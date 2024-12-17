import os
import pandas as pd
import ast
import matplotlib.pyplot as plt

# Construct full paths
dataset_dir = "../Dataset"
test_dataset_file = "13x13-Complete.csv"
dataset = os.path.join(dataset_dir, test_dataset_file)

def plotdistrubtion(dataset):
    df = pd.read_csv(dataset)

    df['MoveListCount'] = df['MoveList'].apply(lambda x: len(ast.literal_eval(x)))

    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(df['MoveListCount'], bins=20, color='gray', edgecolor='k', alpha=0.7)
    plt.title('Distribution of moves in games')
    plt.xlabel('Number of Moves')
    plt.ylabel('Amount')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    max_frequency = n.max()
    max_bin_index = n.argmax()
    highest_frequency_range = (bins[max_bin_index], bins[max_bin_index + 1])

    print(f"The distribution with the highest frequency is in the range {highest_frequency_range} with a frequency of {max_frequency}.")
