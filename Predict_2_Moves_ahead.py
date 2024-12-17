import os
from Components.Config import default_args
from Main import Main
from Components.ReadCSV_2ahead import read_from_csv_with_two_moves_left

# Define dataset directory
dataset_dir = "dataset"

# Define file names
train_dataset_file = "13x13-Training.csv"
test_dataset_file = "13x13-Testing.csv"

# Construct full paths
train_dataset_path = os.path.join(dataset_dir, train_dataset_file)
test_dataset_path = os.path.join(dataset_dir, test_dataset_file)

args = default_args()
Boardsize = 13


if __name__ == "__main__":
    # Instantiate and run the main class
    main_instance = Main(args, Boardsize, test_dataset_path, train_dataset_path,read_from_csv_with_two_moves_left)