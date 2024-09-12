import pandas as pd
import numpy as np
import time
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn import model_selection, svm, neighbors, tree, ensemble


#Turns MM:SS into total seconds for the ToP column
def min_to_sec(time):
    m,s = time.split(':')
    return (60 * int(m)) + int(s)


## Takes in the week as input for file name below
week_number = input('Week Number: ')


## Pulls finished CSV of all games to build the model on, drops weird unnamed column
final_table_2010 = '/Users/serdarevichar/Library/CloudStorage/GoogleDrive-serdarevichar@gmail.com/.shortcut-targets-by-id/1-15YoYpgC0Rx2M_K-tUhgT52UeAZD80z/Football Model/finaltable-2010-2022.csv'
df = pd.read_csv(final_table_2010)
df = df.drop('Unnamed: 0', axis = 1)


## Creates a dataframe consisting of just the features for the model
X_columns = ['X1stD','TotYd','TOLost','TOGained','Cmp','SkYds','QBR','RA','Pnt','X3DRate','ToP']
df_X = df[X_columns]



## Creates train and test sets to run the regression on
X_train, X_test, y_train, y_test = model_selection.train_test_split(df_X,df['Pts'], test_size = 0.2, random_state = 8)


#Runs 4 different linear regression models
LR = linear_model.LinearRegression()
LR.fit(X_train,y_train)
LR_score = LR.score(X_test,y_test)

RR = linear_model.Ridge(alpha=1)
RR.fit(X_train,y_train)
RR_score = RR.score(X_test,y_test)

lasso = linear_model.Lasso(alpha=1)
lasso.fit(X_train,y_train)
lasso_score = lasso.score(X_test,y_test)

lars = linear_model.Lars()
lars.fit(X_train,y_train)
lars_score = lars.score(X_test,y_test)




# =============================================================================
# ##### Perform a full hyperparameter tuning for a random forest regressor, from randomized search to grid search
# 
# 
# ## Create random forest with a random state for replicability
# forest = ensemble.RandomForestRegressor(random_state = 8)
# 
# ## Create the grid of parameters to randomly search through
# param_grid = {
#     'n_estimators': [100,500,1000,1500,2000],
#     'bootstrap': [True],
#     'max_depth': [10,30,50,70,90,100,None],
#     'max_features': [None,'sqrt'],
#     'min_samples_leaf': [1,2,4,6,8,10],
#     'min_samples_split': [2,4,6,8,10],
# }
# 
# 
# ## Run a randomized search on the first parameter grid
# forest_random = model_selection.RandomizedSearchCV(estimator = forest, param_distributions = param_grid, n_iter = 300, cv = 3, verbose = 3)
# forest_random.fit(X_train,y_train)
# 
# 
# ## Displays the best parameters found, which we use to narrow our grid search on
# forest_random.best_params_
# random_model = forest_random.best_estimator_
# random_model.score(X_test,y_test)
# 
# ## Create a new grid to then do a full grid search
# param_grid = {
#     'n_estimators': [900,1000,1100,1200],
#     'bootstrap': [True],
#     'max_depth': [80,90,100,110,None],
#     'max_features': ['sqrt'],
#     'min_samples_leaf': [1,2,3],
#     'min_samples_split': [7,8,9],
# }
# 
# ## Run a full grid search
# forest_grid = model_selection.GridSearchCV(estimator = forest, param_grid = param_grid, cv = 3, verbose = 4)
# forest_grid.fit(X_train,y_train)
# 
# 
# ## Find the best parameters when running the entire grid search
# best_params = forest_grid.best_params_
# 
# 
# ## Grabs the best model and finds its error value
# best_forest = forest_grid.best_estimator_
# best_forest.score(X_test,y_test)        ## Scored at 0.69
# 
# 
# ## To save the best parameters, they will be written out here. Everything above can be commented out
# best_params = {
#     'bootstrap': True,
#     'max_depth': 80,
#     'max_features': 'sqrt',
#     'min_samples_leaf': 1,
#     'min_samples_split': 5,
#     'n_estimators': 1800
#     }
# =============================================================================

model = ensemble.RandomForestRegressor(random_state = 8,
                                       bootstrap = True,
                                       max_depth = 80,
                                       max_features = 'sqrt',
                                       min_samples_leaf = 1,
                                       min_samples_split = 5,
                                       n_estimators = 1800)
model.fit(X_train,y_train)
model_score = model.score(X_test,y_test)




## Runs an SVM regressor and Gradient boosted regressor as well
support = svm.SVR(kernel = 'linear', C = 0.1, epsilon = 1.5)
support.fit(X_train,y_train)
support_score = support.score(X_test,y_test)



# =============================================================================
# ################# MAKE A BETTER MODEL WITH GRADIENT BOOST TREE -- THIS WORKED!!!
# ##### Run a cross validation search to find the right parameters for a GBDT
# 
# ## Create the beginning model
# boost = ensemble.GradientBoostingRegressor(random_state = 8)
# 
# 
# ## Parameter grid to start with randomly
# param_grid = {
#     "n_estimators": [100,200,300,500,1000],
#     "max_depth": [3,5,10,None],
#     "min_samples_split": [2,5,10],
#     "learning_rate": [0.1,0.5,1,2],
#     "loss": ["squared_error"]
# }
# 
# 
# ## Run a randomized search on the first parameter grid
# boost_random = model_selection.RandomizedSearchCV(estimator = boost, param_distributions = param_grid, n_iter = 300, cv = 3, verbose = 5)
# boost_random.fit(X_train,y_train)
# 
# 
# ## Create a refined grid to do a full search on
# param_grid = {
#     "n_estimators": [100,200,300,500,1000],
#     "max_depth": [3,4,5],
#     "min_samples_split": [2,3,4,5],
#     "learning_rate": [0.1],
#     "max_leaf_nodes": [2,5,8,12,None],
#     "loss": ["squared_error"]
#     }
# 
# 
# ## Run a full grid search this time
# boost_grid = model_selection.GridSearchCV(estimator = boost, param_grid = param_grid, cv = 5, verbose = 5)
# boost_grid.fit(X_train,y_train)
# 
# 
# ## Find the best parameters from the search
# best_params = boost_grid.best_params_
# 
# 
# ## Pull the best model and score it
# boost_best = boost_grid.best_estimator_
# boost_best.score(X_test,y_test)
# 
# 
# ## Manually write out the best parameters found to save the model
# best_params = {
#     'learning_rate': 0.1,
#      'loss': 'squared_error',
#      'max_depth': 3,
#      'max_leaf_nodes': 2,
#      'min_samples_split': 2,
#      'n_estimators': 1000
#     }
# =============================================================================


model_boost = ensemble.GradientBoostingRegressor(random_state=8,
                                                 learning_rate = 0.1,
                                                 loss = 'squared_error',
                                                 max_depth = 3,
                                                 max_leaf_nodes = 2,
                                                 min_samples_split = 2,
                                                 n_estimators = 1000)
model_boost.fit(X_train,y_train)
model_boost_score = model_boost.score(X_test,y_test)



## Puts all models and their error scores into a dataframe
models_scores = [LR_score,RR_score,lasso_score,lars_score,model_score,support_score,model_boost_score]
df_models = pd.DataFrame(models_scores,columns = ['Score'], index = ['Least Squares','Ridge','Lasso','LARS','Random Forest','Support Vector Machine','Gradient Boost Decision Tree']).sort_values(by = ['Score'], ascending = False, axis = 0)





#### Visit draftkings and pull data

## Pulls all the tables appearing on the draftkings site
url = 'https://sportsbook.draftkings.com/leagues/football/nfl'
tables = pd.read_html(url)

## Reads through the first 4 tables (Thu,Sun,SunNight,Mon) and makes new column of just the team names
#### if running after Thursday, only pull first 3 tables
for i in range(0,4):
    table = tables[i]
    name = table.columns[0]
    table['Team Names'] = table[name]

## Combines all tables to make one which contains all the games for that weekend, keeps only team names and O/U and spread
df_dk = pd.concat([tables[0],tables[1],tables[2],tables[3]],axis=0,ignore_index=True)
df_dk = df_dk[['Team Names','Total','Spread']]

## Converts teams names into just the team name, and converts O/U, spread into just the number
split_names = df_dk['Team Names'].str.split(expand = True)
df_dk = pd.concat([split_names[1],df_dk[['Total','Spread']]],axis = 1)
df_dk['Total'] = df_dk['Total'].str.slice(start = 2, stop = -4)
df_dk['Spread'] = df_dk['Spread'].str.slice(start = 0, stop = -4)

## Splits the table into 2 tables, home and away to then merge them together
df_away = df_dk.iloc[0::2,:].reset_index(drop = True)
df_home = df_dk.iloc[1::2,0].reset_index(drop = True)

## Combines the home and away tables so that each row has home team, away team, O/U, and spread for that game
df_dk = pd.concat([df_away,df_home], axis = 1, ignore_index = True)
df_dk = df_dk.rename(columns = {0:'Away Team', 1:'O/U', 2:'Away Spread', 3:'Home Team'})
df_dk = df_dk[['Home Team','Away Team','O/U','Away Spread']]

## Convert O/U and spread columns to float
df_dk['O/U'] = df_dk['O/U'].astype(float)
df_dk['Away Spread'] = df_dk['Away Spread'].astype(float)





########## Pull offensive and defensive stats from last year and this year

## Define the dictionary necessary for converting between full team names and three letter abv
teams_full = ['Rams','Falcons','Panthers','Bears','Bengals','Cardinals','Cowboys','Lions','Texans','Dolphins','Buccaneers','Jets','Titans','Chargers','Commanders','Seahawks','Chiefs','Browns','Jaguars','Saints','Giants','Steelers','Ravens','49ers','Broncos','Raiders','Packers','Bills','Eagles','Vikings','Colts','Patriots']
teams = ["ram","atl","car","chi","cin","crd","dal","det","htx","mia","tam","nyj","oti","sdg","was","sea","kan","cle","jax","nor","nyg","pit","rav","sfo","den","rai","gnb","buf","phi","min","clt","nwe"]
team_dictionary = {teams_full[i]:teams[i] for i in range(0,len(teams))}

## Make a list of just the teams pulled in the DK table (in case running for only Thurs game or Mon game)
teams_playing = list(df_dk['Home Team']) + list(df_dk['Away Team'])
teams_playing_abv = [team_dictionary[i] for i in teams_playing]

## Make empty dataframes to then append to in the loop
offensive_stats = pd.DataFrame(columns = ['1stD', 'TotYd', 'TO', 'TO.1', 'Cmp', 'Yds.1', 'Rate', 'Att.1', 'Pnt','ToP','3DRate'])
defensive_stats = pd.DataFrame(columns = ['1stD.1', 'TotYd.1', 'TO.1', 'TO', 'Cmp', 'Yds.1', 'Rate', 'Att.1', 'Pnt','ToP','3DRate'])

## For loop of pulling each of the stats
for team in teams_playing_abv:
    
    ## Pull in this year's standard table (includes both offense and defense)
    url = f"https://www.pro-football-reference.com/teams/{team}/2023.htm"
    tables = pd.read_html(url, header = 1)
    
    game_off_2023 = tables[1][['1stD','TotYd','TO','TO.1']]
    game_off_2023 = game_off_2023.fillna({'TO':0, 'TO.1':0})
    game_off_2023.dropna(inplace=True)
    
    game_def_2023 = tables[1][['1stD.1','TotYd.1','TO.1','TO']]
    game_def_2023 = game_def_2023.fillna({'TO':0,'TO.1':0})
    game_def_2023.dropna(inplace=True)

    time.sleep(3)
    
    
    ## Pull in last year's standard table
    url = f"https://www.pro-football-reference.com/teams/{team}/2022.htm"
    tables = pd.read_html(url, header = 1)
    
    game_off_2022 = tables[1][['1stD','TotYd','TO','TO.1']]
    game_off_2022 = game_off_2022.fillna({'TO':0, 'TO.1':0})
    game_off_2022.dropna(inplace=True)
    
    game_def_2022 = tables[1][['1stD.1','TotYd.1','TO.1','TO']]
    game_def_2022 = game_def_2022.fillna({'TO':0, 'TO.1':0})
    game_def_2022.dropna(inplace=True)
    
    time.sleep(3)
    
    
    ## Pull in this year's gamelog table
    url = f'https://www.pro-football-reference.com/teams/{team}/2023/gamelog/'
    tables = pd.read_html(url, header = 1)
    
    gamelog_off_2023 = tables[0][['Cmp','Yds.1','Rate','Att.1','Pnt','ToP']]
    gamelog_off_2023 = gamelog_off_2023.dropna()
    gamelog_off_2023['3DRate'] = tables[0]['3DConv'] / tables[0]['3DAtt']    
    gamelog_off_2023['ToP'] = gamelog_off_2023['ToP'].apply(min_to_sec)
        
    gamelog_def_2023 = tables[1][['Cmp','Yds.1','Rate','Att.1','Pnt','ToP']]
    gamelog_def_2023 = gamelog_def_2023.dropna()
    gamelog_def_2023['3DRate'] = tables[1]['3DConv'] / tables[1]['3DAtt']     
    gamelog_def_2023['ToP'] = gamelog_def_2023['ToP'].apply(min_to_sec)
        
    time.sleep(3)
    
    
    ## Pull in last year's gamelog table
    url = f'https://www.pro-football-reference.com/teams/{team}/2022/gamelog/'
    tables = pd.read_html(url, header = 1)

    gamelog_off_2022 = tables[0][['Cmp','Yds.1','Rate','Att.1','Pnt','ToP']]
    gamelog_off_2022 = gamelog_off_2022.dropna()
    gamelog_off_2022['3DRate'] = tables[0]['3DConv'] / tables[0]['3DAtt']     
    gamelog_off_2022['ToP'] = gamelog_off_2022['ToP'].apply(min_to_sec)

    gamelog_def_2022 = tables[1][['Cmp','Yds.1','Rate','Att.1','Pnt','ToP']]
    gamelog_def_2022 = gamelog_def_2022.dropna()
    gamelog_def_2022['3DRate'] = tables[1]['3DConv'] / tables[1]['3DAtt']     
    gamelog_def_2022['ToP'] = gamelog_def_2022['ToP'].apply(min_to_sec)
    
    
    ## Take the averages of both last year tables by column
    game_off_2022 = game_off_2022.mean(numeric_only=True).to_frame().T
    game_def_2022 = game_def_2022.mean(numeric_only=True).to_frame().T
    gamelog_off_2022 = gamelog_off_2022.mean(numeric_only=True).to_frame().T
    gamelog_def_2022 = gamelog_def_2022.mean(numeric_only=True).to_frame().T
    
    
    ## Connect the game and gamelog tables
    off_stats_2023 = pd.concat([game_off_2023,gamelog_off_2023], axis = 1)
    def_stats_2023 = pd.concat([game_def_2023,gamelog_def_2023], axis = 1)
    off_stats_2022 = pd.concat([game_off_2022,gamelog_off_2022], axis = 1)
    def_stats_2022 = pd.concat([game_def_2022,gamelog_def_2022], axis = 1)
    
    
    ## Stack offense and defensive tables from this and last year
    off_stats = pd.concat([off_stats_2023,off_stats_2022], axis = 0)
    def_stats = pd.concat([def_stats_2023,def_stats_2022], axis = 0)
    
    
    ## Again average out the columns
    off_stats = off_stats.mean(numeric_only=True).to_frame().T
    def_stats = def_stats.mean(numeric_only=True).to_frame().T
    off_stats.rename(index = {0:team}, inplace = True)
    def_stats.rename(index = {0:team}, inplace = True)
    
    
    ## Save this as a row in the main offensive stats and defensive stats table
    offensive_stats = pd.concat([offensive_stats,off_stats], axis = 0)
    defensive_stats = pd.concat([defensive_stats,def_stats], axis = 0)
    
    time.sleep(3)


## Change the column names to format in model, as well as order of columns
column_names = ['X1stD','TotYd','TOLost','TOGained','Cmp','SkYds','QBR','RA','Pnt','X3DRate','ToP']
offensive_stats.rename(columns = {'1stD':'X1stD','TO':'TOLost','TO.1':'TOGained','Yds.1':'SkYds','Rate':'QBR','Att.1':'RA','3DRate':'X3DRate'}, inplace = True)
defensive_stats.rename(columns = {'TotYd.1':'TotYd','1stD.1':'X1stD','TO':'TOGained','TO.1':'TOLost','Yds.1':'SkYds','Rate':'QBR','Att.1':'RA','3DRate':'X3DRate'}, inplace = True)
offensive_stats = offensive_stats[column_names]
defensive_stats = defensive_stats[column_names]




## Converts all teams in games table to 3 letter abreviations in lists of home and away
home_teams = list(df_dk['Home Team'])
home_teams = [team_dictionary[home_teams[i]] for i in range(len(home_teams))]
away_teams = list(df_dk['Away Team'])   
away_teams = [team_dictionary[away_teams[i]] for i in range(len(away_teams))]


## Splits the offensive and defensive stats tables into the home and away subtables
home_offensive_stats = offensive_stats.loc[home_teams].reset_index(drop=True)
away_defensive_stats = defensive_stats.loc[away_teams].reset_index(drop=True)
away_offensive_stats = offensive_stats.loc[away_teams].reset_index(drop=True)
home_defensive_stats = defensive_stats.loc[home_teams].reset_index(drop=True)


## Combine home off with away def and away off with home def to get inputs for the model
home_inputs = home_offensive_stats.add(away_defensive_stats).div(2)
away_inputs = away_offensive_stats.add(home_defensive_stats).div(2)


## Use the model to predict values
home_proj_score = pd.DataFrame(model_boost.predict(home_inputs), columns = ['Home Score'])
away_proj_score = pd.DataFrame(model_boost.predict(away_inputs), columns = ['Away Score'])


## Add the projected scores into the table of odds
df_dk = pd.concat([df_dk,home_proj_score,away_proj_score], axis = 1)


## Add in a column of the projected spread result and difference from the spread line
df_dk['Proj Away Spread'] = df_dk['Home Score'] - df_dk['Away Score']
df_dk['Spread Difference'] = df_dk['Proj Away Spread'] - df_dk['Away Spread']


## Make a column in df_dk of the picks (here it is empty first)
df_dk['Picks'] = np.zeros(len(df_dk))


## Fill in the picks column with our picks
homes = (df_dk['Spread Difference'] > 2.5)
df_dk.loc[homes,'Spread Picks'] = df_dk['Home Team']
aways = (df_dk['Spread Difference'] < -2.5)
df_dk.loc[aways,'Spread Picks'] = df_dk['Away Team']


## Make a table of only spreads we picked
df_spread = df_dk.loc[(df_dk['Spread Picks'] != 0), ['Home Team','Away Team','Away Spread','Spread Picks']]



##### Does the same for O/Us but not sure want to continue in that direction
## Add in columns of the projected O/U as well as difference from the line. If negative, we picked under
df_dk['Proj O/U'] = df_dk['Home Score'] + df_dk['Away Score']
df_dk['O/U Difference'] = df_dk['Proj O/U'] - df_dk['O/U']


## Make a column in df_dk of the picks (here it is empty first)
df_dk['Picks'] = np.zeros(len(df_dk))


## Fill the picks column with our pick (or no pick)
overs = (df_dk['O/U Difference'] > 2.5)
df_dk.loc[overs,'O/U Picks'] = 'Over'
unders = (df_dk['O/U Difference'] < -2.5)
df_dk.loc[unders,'O/U Picks'] = 'Under'


## Make a table of only the O/U we made a pick for
df_ou = df_dk.loc[(df_dk['O/U Picks'] != 0), ['Home Team','Away Team','O/U','O/U Picks']]


## Output a CSV of the entire table that can then be pasted into the main google sheet
df_dk.to_excel(f'/Users/serdarevichar/Documents/Python/Football Model/week{week_number}.xlsx', index = False)








































