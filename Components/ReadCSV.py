import csv
# This CSV loader loads the entire dataset, for evaluation AFTER the game has ended.

def ReadCSV(filename):
    Games = []
    with open(filename, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            winner = int(row['Winner'])
            features = eval(row['Board'])
            Games.append([winner, features])
    return Games
