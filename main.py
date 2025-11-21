import mysklearn.mypytable

from mysklearn.mypytable import MyPyTable 
from mysklearn import myutils

from pathlib import Path

# Link that may or may not have to use not too sure
# Provided by <a href="https://www.sports-reference.com
# /sharing.html?utm_sour
# ce=direct&utm_medium=Share&ut
# m_campaign=ShareTool">Basketball-Reference.com</a>: <a h
# ref="https://www.basketball-reference.com/teams/ATL/2025.html
# ?sr&utm_source=direct&utm_medium=Share&utm_campaign=ShareTool#team
# _misc">View Original Table</a><br>Generated 11/19/2025.


# All teams with stats and their final records to use for testing and training
teams = MyPyTable().load_from_file("data/team-stats-24-25.txt")
teams.save_to_file("data/hawksOut.txt")



# Example of just the warriors that we can use for training and testing
warriors = MyPyTable().load_from_file("data/firstGames/warriors.txt")
warriors.drop_column([""])
# Get just the regular season games
print("table shape:", warriors.get_shape())
myutils.get_reg_season_games(warriors)
print("new table shape:", warriors.get_shape())

# Get the first 20 games, gonna make a separate table for the example but may not want to in final
warriors_f_20 = MyPyTable().load_from_file("data/firstGames/warriors.txt")
warriors_f_20.drop_column([""])
print(warriors_f_20.get_shape())
myutils.get_first_n_games(warriors_f_20, 20)
print(warriors_f_20.get_shape())
warriors_f_20.save_to_file("data/firstGames/fixed/warriOut.txt")

# Trying to look at each file using pathlib
team_headers = []
team_tables = []
folder = Path("data/firstGames")

for file in folder.glob("*.txt"):
    table = MyPyTable().load_from_file(str(file))
    team_tables.append(table)

print(team_tables[0].get_shape())