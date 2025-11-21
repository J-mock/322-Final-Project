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


'''
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
'''

# Now trying with the game log
cols_to_drop = ["Rk", "Gtm", "Date", "OT", "FG", "FGA", "FG%", "STL", "BLK", "3P%", "2P%", "eFG%", "FT%", "", "Opp", "ORB", "DRB"]
cols_to_total = []
alphabet_teams = ['Atlanta Hawks', 'Boston Celtics', 'Brooklyn Nets', 'Charlotte Hornets', 'Chicago Bulls',
        'Cleveland Cavaliers', 'Dallas Mavericks', 'Denver Nuggets', 'Detroit Pistons', 'Golden State Warriors', 'Houston Rockets', 'Indiana Pacers',
        'Los Angeles Clippers', 'Los Angeles Lakers', 'Memphis Grizzlies', 'Miami Heat', 'Milwaukee Bucks', 'Minnesota Timberwolves',
        'New Orleans Pelicans', 'New York Knicks', 'Oklahoma City Thunder', 'Orlando Magic', 'Philadelphia 76ers',
        'Phoenix Suns', 'Portland Trailblazers', 'Sacramento Kings', 'San Antonio Spurs', 'Toronto Raptors', 'Utah Jazz', 'Washington Wizards']


hawks = MyPyTable().load_from_file("data/otherFirst/Atl_Hawks.txt")
hawks.drop_column(cols_to_drop)
myutils.get_first_n_games(hawks, 20)
hawks.convert_to_numeric()
hawks.get_totals("hawks")
hawks.pretty_print()


celtics = MyPyTable().load_from_file("data/otherFirst/Bost_Celt.txt")
celtics.drop_column(cols_to_drop)
myutils.get_first_n_games(celtics, 20)
celtics.convert_to_numeric()
celtics.get_totals("celtics")
celtics.pretty_print()

hawks.save_to_file("data/output/trial1.txt")
celtics.save_DATA_to_file("data/output/trial1.txt")
print()
print()
print()
folder = Path("data/otherFirst/")
files = sorted(folder.glob("*.txt"))  # alphabetical order

team_tables = [MyPyTable().load_from_file(str(f)) for f in files]

for i, team in enumerate(team_tables):
    team.drop_column(cols_to_drop)
    myutils.get_first_n_games(team, 20)
    team.convert_to_numeric()
    print(team.column_names)
    team.get_totals(alphabet_teams[i])
    if i == 0:
        team.save_to_file("data/output/trial1.txt")
    else:
        team.save_DATA_to_file("data/output/trial1.txt")