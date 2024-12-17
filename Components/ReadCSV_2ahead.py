import csv
# This CSV loader loads the entire dataset but sets the last 2 moves to empty/none, for evaluation 2 moves before the game finishes.

def read_from_csv_with_two_moves_left(filename):
    simulations = []
    with open(filename, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            winner = int(row['Winner'])
            features = eval(row['Board'])
            moves_list = eval(row['MoveList'])  # Parse the movesList column

            if len(moves_list) >= 2:
                # Get the last two moves from the movesList
                last_move = moves_list[-1]
                second_last_move = moves_list[-2]

                # Set the corresponding feature indices to 'Empty'
                features[last_move] = 'Empty'
                features[second_last_move] = 'Empty'

                simulations.append([winner, features])
            else:
                # Skip games with fewer than 2 moves
                print(f"Skipping simulation with insufficient moves: {moves_list}")
    return simulations