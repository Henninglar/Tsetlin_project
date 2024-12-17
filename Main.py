from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
import os
from Components.Config import default_args
from Components.ReadCSV import ReadCSV
from Components.ReadCSV_5ahead import read_from_csv_with_five_moves_left
from Components.ReadCSV_2ahead import read_from_csv_with_two_moves_left
from Components.plotdistrubtion import plotdistrubtion
from Components.Evaluation import plot_accuracies
import numpy as np
from Components.FindEdges import FindEdges
import csv

# Define dataset directory
dataset_dir = "dataset"

# Define file names
train_dataset_file = "13x13-Training.csv"
test_dataset_file = "13x13-Testing.csv"

# Construct full paths
train_dataset_path = os.path.join(dataset_dir, train_dataset_file)
test_dataset_path = os.path.join(dataset_dir, test_dataset_file)

## plotdistrubtion(test_dataset_path) - NOT USED, ONLY FOR REPORT PLOTTING

## plotdistrubtion(train_dataset_path) - NOT USED, ONLY FOR REPORT PLOTTING
args = default_args()
Boardsize = 13


class Main:
    def __init__(self, Args, Boardsize, Test_dataset, Train_dataset, DataLoader):
        symbols = ["O", "X", "*"]
        self.args = Args
        self.DataLoader = DataLoader
        self.Edges = FindEdges(Boardsize)
        print("Initialization done")

        if self.DataLoader == ReadCSV:
            print("Normal READCSV Loaded")
        elif (self.DataLoader == read_from_csv_with_five_moves_left):
            print("5 moves ahead loaded.")
        elif (self.DataLoader == read_from_csv_with_two_moves_left):
            print("2 moves ahead loaded.")

        # Read training and testing data from CSV files
        print("Loading data from CSV")
        Testing_data = self.DataLoader(Test_dataset)
        Training_data = self.DataLoader(Train_dataset)

        totaldata = len(Training_data) + len(Testing_data)
        print("Total amount of data", totaldata)


        #GRAPH FOR TRAINING

        graphs_train = Graphs(
            len(Training_data), # Specifies the number of individual graphs to create, one for each game in the training dataset.
            symbols=symbols, #Provides a list of symbols (["O", "X", "*"]) to represent the properties of the nodes in the graph.
            hypervector_size=self.args.hypervector_size, #defines the size of the hypervectors used for encoding the graph.
            hypervector_bits=self.args.hypervector_bits, # Specifies the number of bits in each hypervector.
            double_hashing=self.args.double_hashing #if doublehasing is used (true or false)
        )

        # ensures each graph in graphs_train is initialized with the correct number of nodes.
        for graph_id, game in enumerate(Training_data):
            winner, moves = game
            graphs_train.set_number_of_graph_nodes(graph_id, len(moves))

        # finalizes the node configuration for all graphs in graphs_train.
        graphs_train.prepare_node_configuration()

        #Adds nodes to each graph, specifying their connections (
        for graph_id, game in enumerate(Training_data):
            for node_id in range(len(self.Edges)):
                if self.Edges[node_id]:
                    graphs_train.add_graph_node(graph_id, node_id, len(self.Edges[node_id]))

        #Prepares the graphs to define edges between nodes.
        graphs_train.prepare_edge_configuration()

        # Adds edges to connect nodes based on their relationships.
        for graph_id, game in enumerate(Training_data):
            winner, moves = game
            for node_id in range(len(self.Edges)):
                if self.Edges[node_id]:
                    for edge in self.Edges[node_id]:
                        if moves[node_id] == moves[edge]:
                            graphs_train.add_graph_node_edge(graph_id=graph_id, source_node_name=node_id, destination_node_name=edge, edge_type_name="Connected")
                        else:
                            graphs_train.add_graph_node_edge(graph_id=graph_id, source_node_name=node_id, destination_node_name=edge, edge_type_name="Not Connected")

                feature = moves[node_id]
                if feature == 'Black':
                    graphs_train.add_graph_node_property(graph_id, node_id, symbols[1])
                elif feature == 'White':
                    graphs_train.add_graph_node_property(graph_id, node_id, symbols[0])
                elif feature == 'Empty':
                    graphs_train.add_graph_node_property(graph_id, node_id, symbols[2])

        graphs_train.encode()

        #GRAPH FOR TESTING

        graphs_test = Graphs(
            len(Testing_data),
            symbols=symbols,
            hypervector_size=self.args.hypervector_size,
            hypervector_bits=self.args.hypervector_bits,
            double_hashing=self.args.double_hashing
        )
        # Ensure graph_test config is as graph_train
        graphs_test.symbol_id = graphs_train.symbol_id
        graphs_test.edge_type_id = graphs_train.edge_type_id
        graphs_test.hypervectors = graphs_train.hypervectors
        graphs_test.number_of_hypervector_chunks = graphs_train.number_of_hypervector_chunks

        for graph_id, game in enumerate(Testing_data):
            winner, moves = game
            graphs_test.set_number_of_graph_nodes(graph_id, len(moves))

        graphs_test.prepare_node_configuration()

        for graph_id, game in enumerate(Testing_data):
            for node_id in range(len(self.Edges)):
                if self.Edges[node_id]:
                    graphs_test.add_graph_node(graph_id, node_id, len(self.Edges[node_id]))

        graphs_test.prepare_edge_configuration()

        for graph_id, game in enumerate(Testing_data):
            _, moves = game
            for node_id in range(len(self.Edges)):
                if self.Edges[node_id]:
                    for edge in self.Edges[node_id]:
                        if moves[node_id] == moves[edge]:
                            graphs_test.add_graph_node_edge(graph_id=graph_id, source_node_name=node_id, destination_node_name=edge, edge_type_name="Connected")
                        else:
                            graphs_test.add_graph_node_edge(graph_id=graph_id, source_node_name=node_id, destination_node_name=edge, edge_type_name="Not Connected")

                feature = moves[node_id]
                if feature == 'Black':
                    graphs_test.add_graph_node_property(graph_id, node_id, symbols[1])
                elif feature == 'White':
                    graphs_test.add_graph_node_property(graph_id, node_id, symbols[0])
                else:
                    graphs_test.add_graph_node_property(graph_id, node_id, symbols[2])

        graphs_test.encode()


        # Initilaing the MultiClassGraphTsetlinMachine
        tm = MultiClassGraphTsetlinMachine(
            self.args.number_of_clauses,
            self.args.T,
            self.args.s,
            depth=self.args.depth,
            message_size=self.args.message_size,
            message_bits=self.args.message_bits,
            max_included_literals=self.args.max_included_literals,
            grid=(16 * 13, 1, 1), ## This is for the Nvidia gpu to work - DO NOT REMOVE!
            block=(128, 1, 1)  ## This is for the Nvidia gpu to work - DO NOT REMOVE!
        )

        accuracy_train_epochs = []
        accuracy_test_epochs = []
        Y_train = np.array([simulation[0] for simulation in Training_data])
        Y_test = np.array([simulation[0] for simulation in Testing_data])

        for i in range(self.args.epochs):
                tm.fit(graphs_train, Y_train, epochs=1, incremental=True)
                train_prediction = tm.predict(graphs_train)

                accuracy_train = np.mean(Y_train == train_prediction)
                accuracy_train = round(accuracy_train, 3)
                accuracy_train_epochs.append(accuracy_train)

                pred = tm.predict(graphs_test)

                accuracy_test = np.mean(Y_test == pred)
                accuracy_test = round(accuracy_test, 3)
                accuracy_test_epochs.append(accuracy_test)

                print(f"Epoch iteration:{1+ i}")
                print(f"Training Accuracy: {accuracy_train}")
                print(f"Testing Accuracy: {accuracy_test}")


                avg_accuracy_train = np.mean(accuracy_train_epochs)
                avg_accuracy_test = np.mean(accuracy_test_epochs)

        plot_accuracies(accuracy_train_epochs,accuracy_test_epochs)
        print("avg_accuracy_train", avg_accuracy_train)
        print("avg_accuracy_test", avg_accuracy_test)

        weights = tm.get_state()[1].reshape(2, -1)

        positive_count = 0
        negative_count = 0
        neutral_count = 0

        # List to store clause details
        clause_details = []

        with open("clauselog.txt", 'w') as clauses_log_file:
            for i in range(tm.number_of_clauses):
                pos_weight = weights[0, i]
                neg_weight = weights[1, i]

                # Write clause weights
                clauses_log_file.write("Clause - %d (%d %d) " % (i, pos_weight, neg_weight))

                # Determine clause polarity and update counts
                if pos_weight > neg_weight:
                    polarity = "Positive Polarity"
                    positive_count += 1
                elif neg_weight > pos_weight:
                    polarity = "Negative Polarity"
                    negative_count += 1
                else:
                    polarity = "Neutral"
                    neutral_count += 1

                clauses_log_file.write(f"Polarity: {polarity}\n")

                # Append clause details for sorting later
                clause_details.append((i, pos_weight, neg_weight, polarity))

                # Build clause literals
                l = []
                for k in range(self.args.hypervector_size * 2):
                    if tm.ta_action(0, i, k):
                        if k < self.args.hypervector_size:
                            l.append("x%d" % (k))
                        else:
                            l.append("NOT x%d" % (k - self.args.hypervector_size))

                # Write clause literals and count
                clauses_log_file.write(" AND ".join(l))
                clauses_log_file.write(f"\nNumber of literals: {len(l)}\n\n")

        # Sort clauses to find the top 2 positive and top 2 negative
        sorted_positive = sorted(clause_details, key=lambda x: x[1], reverse=True)  # Sort by positive weight
        sorted_negative = sorted(clause_details, key=lambda x: x[2], reverse=True)  # Sort by negative weight

        top_2_positive = sorted_positive[:2]
        top_2_negative = sorted_negative[:2]

        # Print the polarity summary at the end
        print("Clause Polarity Summary:")
        print(f"Positive Clauses: {positive_count}")
        print(f"Negative Clauses: {negative_count}")
        print(f"Neutral Clauses: {neutral_count}")

        # Print the top 2 positive and negative clauses
        print("\nTop 2 Positive Clauses:")
        for clause in top_2_positive:
            print(f"Clause {clause[0]} ({clause[1]} {clause[2]}) - Polarity: {clause[3]}")

        print("\nTop 2 Negative Clauses:")
        for clause in top_2_negative:
            print(f"Clause {clause[0]} ({clause[1]} {clause[2]}) - Polarity: {clause[3]}")


if __name__ == "__main__":
    # Instantiate and run the main class
    main_instance = Main(args, Boardsize, test_dataset_path, train_dataset_path,ReadCSV)
