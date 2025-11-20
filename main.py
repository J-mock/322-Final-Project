import mysklearn.mypytable

from mysklearn.mypytable import MyPyTable 

# Provided by <a href="https://www.sports-reference.com/sharing.html?utm_source=direct&utm_medium=Share&utm_campaign=ShareTool">Basketball-Reference.com</a>: <a href="https://www.basketball-reference.com/teams/ATL/2025.html?sr&utm_source=direct&utm_medium=Share&utm_campaign=ShareTool#team_misc">View Original Table</a><br>Generated 11/19/2025.

teams = MyPyTable().load_from_file("data/team-stats-24-25.txt")
teams.save_to_file("data/hawksOut.txt")

spurs = MyPyTable().load_from_file("data/firstGames/spurs.txt")
spursWL = [spurs.get_column("W"), spurs.get_column("L")]
print(spursWL[1][20])
# Main scope, will use the team-stats-24-24 as the y-train (wins and losses)
# X train will be the first 10-20 games for each team, will probably need another dataset
# for that
# How well can we predict the outcome of a teams season based off their first games
# Can base it just off record or include stats as well to help with classifier if we really want to
