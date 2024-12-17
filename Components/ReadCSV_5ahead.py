import csv
# This CSV loader loads the entire dataset but sets the last 5 moves to empty/none, for evaluation 5 moves before the game finishes.

def read_from_csv_with_five_moves_left(filename):
    simulations = []
    with open(filename, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            winner = int(row['Winner'])
            features = eval(row['Board'])
            moves_list = eval(row['MoveList'])  # Parse the movesList column

            if len(moves_list) >= 5:
                # Get the last 5 moves from the movesList
                last_five_moves = moves_list[-5:]

                # Set the corresponding feature indices to 'None'
                for move in last_five_moves:
                    features[move] = 'Empty'

                simulations.append([winner, features])
            else:
                # Skip games with fewer than 5 moves
                print(f"Skipping simulation with insufficient moves: {moves_list}")
    return simulations