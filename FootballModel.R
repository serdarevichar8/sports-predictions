library("rvest")
library("stringr")
#library("lubridate")
library("leaps")
library("ISLR2")
#library("caret")
library("MASS")
library("dplyr")

rm(list = ls())

# Kevin working directory
setwd("G:/My Drive/Football Model")

ptm = proc.time()

team = c("ram","atl","car","chi","cin","crd","dal","det",
         "htx","mia","tam","nyj","oti","sdg","was","sea",
         "kan","cle","jax","nor","nyg","pit","rav","sfo",
         "den","rai","gnb","buf","phi","min","clt","nwe")

year = c(2010:2022)

teamtable = list()

##################################### Pulling data - Loop ##################################### 

for (i in 1:length(team)) {
  
  yeartable = list()
  
  for (j in 1:length(year)) {
    
    URL = paste0("https://www.pro-football-reference.com/teams/",team[i],"/",year[j],".htm")
    URL2 = paste0("https://www.pro-football-reference.com/teams/",team[i],"/",year[j],"/gamelog/")
    
    game = URL %>%
      read_html() %>%
      html_nodes(css = "#games") %>%
      html_table() %>% as.data.frame()
    
    colnames(game) = game[1,]
    game = game[-1,]
    game = game[!game[,which(colnames(game) == "Day")] == "",]
    
    # Creating home/away column
    colnames(game)[9] = c("home")
    game$home[!game$home == "@"] = 1
    game$home[game$home == "@"] = 0
    
    #fix game if playoffs
    game = game[!is.na(as.integer(game$Week)),]
    game = game[!game[,5] == "canceled",]
    
    game_advanced_css = paste0("#gamelog",year[j])
    
    game_advanced = URL2 %>%
      read_html() %>%
      html_nodes(css = game_advanced_css) %>%
      html_table() %>% as.data.frame()
    
    colnames(game_advanced) = game_advanced[1,]
    game_advanced = game_advanced[-1,]
    
    game = cbind.data.frame(game,game_advanced)
    game = game[,-c(1:8,10,12,18:21,23:35,38:40,43:45,48:54)]
    colnames(game) = c("home","Pts","1stD","TotYd","PassY","RushY","TOLost","TOGained","Cmp","PA","Sk","SkYds",
                       "QBR","RA","Pnt","PntYds","3DConv","3DAtt","4DConv","4DAtt","ToP")
  
    Sys.sleep(5)
    
    yeartable[[j]] = game
    
  }
  
  teamtable[[i]] = do.call("rbind",yeartable)

}

finaltable = do.call("rbind",teamtable)

View(finaltable)

write.csv(finaltable, '/Users/serdarevichar/Documents/Python/Football Model/finaltable-2010-2022.csv')

offensive_table = finaltable

##################################### END Pulling data - Loop ###########################

##################################### Cleaning data ##################################### 

        ## CLEANING
        
        #Reading in file from above (to avoid running loop again)
        offensive_table = read.csv("finaltable.csv")
        defensive_table = read.csv("DefensiveTable.csv")
        
        ######### Offensive Cleaning #############
        offensive_table$TO = as.integer(offensive_table$TO)
        offensive_table$TO[is.na(offensive_table$TO)] = 0
        offensive_table = offensive_table[,-1]
        offensive_table[is.na(offensive_table)] = 0
        
        ## removing highly correlated models
        
        offensive_table$ToP = as.numeric(ms(offensive_table$ToP))
        offensive_table = offensive_table %>% mutate_at(1:dim(offensive_table)[2],as.numeric) 
            
            # Changing 3rd / 4th down conversions to rate 
            offensive_table$`X3DRate` = offensive_table$`X3DConv`/offensive_table$`X3DAtt`
            offensive_table$X3DRate[is.na(offensive_table$X3DRate)] = 0
        
            # removing attempts (3rd & 4th) & 4th down rate (very low amount of occurences that would be hard to predict)
            offensive_table = offensive_table %>% select(-c(`X3DAtt`,`X4DAtt`,`X4DConv`))
            
            # removing pass and rush yards, pass attempts, punt yds
            offensive_table = offensive_table %>% select(-c(PassY, RushY,PA, PntYds,`X3DConv`,Sk))
            
            write.csv(finaltable,"finaltable-2010-2022.csv")
            
            
##################################### Cleaning data #####################################             
            
            
        
##################################### Model Creation #####################################    
            
# Create Correlation Matrix - Offensive Model
cor_matrix = cor(offensive_table)
cor_matrix_rm = cor_matrix
cor_matrix_rm[upper.tri(cor_matrix_rm)] <- 0
diag(cor_matrix_rm) <- 0
cor_matrix_rm

offensive_table_updated <- offensive_table[ , !apply(cor_matrix_rm,    # Remove highly correlated variables
                           2,
                           function(x) any(x > 0.75))]

# Testing Predictions
Xh = data.frame(X1stD=18,TO=0,Cmp=13, SkYds = 0, QBR = 69.2, RA = 30, ToP = 1592, X3DRate = .46)
predict(ModelOff, Xh)

set.seed(123)
trainIndex <- createDataPartition(offensive_table_updated$Pts, p = 0.8, list = FALSE)
trainData <- offensive_table_updated[trainIndex, ]
testData <- offensive_table_updated[-trainIndex, ]

#Creating Offensive Model - .68 r^2
FitallOff = lm(Pts~.,data = trainData)
ModelOff = step(FitallOff, direction = "backward")

predictedValues <- predict(ModelOff, newdata = testData, type = "response")

SS.total      <- sum((testData$Pts - mean(testData$Pts))^2)
SS.residual   <- sum((testData$Pts - predictedValues)^2)
SS.regression <- sum((predictedValues - mean(testData$Pts))^2)

R2 = SS.regression/SS.total

##################################### Model Creation ##################################### 


###################### Pulling in Draftkings Data ############################

            ############### Reading in Games
            game_table_url = "https://sportsbook.draftkings.com/leagues/football/nfl"
            
            css_games = ".sportsbook-table"
            
            page = read_html(game_table_url)
            games = page %>% html_element(css = css_games) %>% html_table()
            games = as.data.frame(games)
            
            for (i in 1: nrow(games)) {
              
              x = strsplit(games[i,1]," ")
              games[i,1] = x[[1]][2]
              
              games[i,2] = substr(games[i,2],1,nchar(games[i,2])-4)
              games[i,3] = substr(games[i,3],1,nchar(games[i,3])-4)
            }
            
            games = games[,-c(4)]
            
            games$hTEAM = rep(NA,nrow(games))
            colnames(games) = c("aTEAM","aSPREAD","O/U","hTEAM")
            
            games = games %>% relocate(`O/U`, .after = hTEAM)
            
            for (i in 1: (nrow(games)/2)) {
              
              games[(2*i)-1,3] = games[2*i,1]
              #games[(2*i)-1,5] = games[2*i,2]
              #games[(2*i)-1,6] = games[2*i,3]
              
            }
            
            games = games[-c(seq(2,nrow(games),by=2)),]
            rownames(games) = 1:nrow(games)
            
            # Creating Team Dictionary
            DkTeam = c("Rams","Falcons","Panthers","Bears","Bengals","Cardinals","Cowboys","Lions",
                       "Texans","Dolphins","Buccaneers","Jets","Titans","Chargers","Commanders","Seahawks",
                       "Chiefs","Browns","Jaguars","Saints","Giants","Steelers","Ravens","49ers",
                       "Broncos","Raiders","Packers","Bills","Eagles","Vikings","Colts","Patriots")
            
            team = c("ram","atl","car","chi","cin","crd","dal","det",
                     "htx","mia","tam","nyj","oti","sdg","was","sea",
                     "kan","cle","jax","nor","nyg","pit","rav","sfo",
                     "den","rai","gnb","buf","phi","min","clt","nwe")
            
            Dict = cbind.data.frame(DkTeam, team)
            
            games = left_join(games,Dict,by=c("aTEAM" = "DkTeam"))
            games = left_join(games,Dict,by=c("hTEAM" = "DkTeam"))
            colnames(games)[c(5,6)] = c("aABV","hABV")
            
            games$aTEAM = games$aABV
            games$hTEAM = games$hABV
            games = games[,-c(5,6)]
            
            games$`O/U` = as.numeric(substr(games$`O/U`,3,nchar(games$`O/U`)))
            games[which(games$aSPREAD == "pk"),2] = 0
            
            View(games)
            
###################### Pulling in Draftkings Data ############################
            
            
            
            
            
            
## Predict values for input into model - want to talk with Har
## Pull in exp 1stD, PassY, PA, SkYds, QBR, RA, PntYds, ToP, 3DRate to input into pts model

            predictedValues <- predict(ModelOff, newdata = testData, type = "response")
            
            # Create loop to collect season stats to fill inputs
            
            for (a in 1:length(teams)) {
            
            URL = paste0("https://www.pro-football-reference.com/teams/",team[a],"/",year(Sys.Date()),".htm")
            URL2 = paste0("https://www.pro-football-reference.com/teams/",team[a],"/",year(Sys.Date()),"/gamelog/")
            
            game = URL %>%
              read_html() %>%
              html_nodes(css = "#games") %>%
              html_table() %>% as.data.frame()
            
            colnames(game) = game[1,]
            game = game[-1,]
            game = game[!game[,which(colnames(game) == "Day")] == "",]
            
            #fix game if playoffs
            game = game[!is.na(as.integer(game$Week)),]
            game = game[!game[,5] == "canceled",]
            
            game_advanced_css = paste0("#gamelog",year(Sys.Date()))
            
            game_advanced = URL2 %>%
              read_html() %>%
              html_nodes(css = game_advanced_css) %>%
              html_table() %>% as.data.frame()
            
            colnames(game_advanced) = game_advanced[1,]
            game_advanced = game_advanced[-1,]
            
            game = cbind.data.frame(game,game_advanced)
            game = game[,-c(1:10,12,18:35,38:40,43:45,48:54)]
            colnames(game) = c("Pts","1stD","TotYd","PassY","RushY","TO","Cmp","PA","Sk","SkYds",
                               "QBR","RA","Pnt","PntYds","3DConv","3DAtt","4DConv","4DAtt","ToP")
            
            game[game == ''] <- NA
            game = game %>% filter(!`1stD` %in% c(NA))
            
            #Cleaning
            game$TO = as.integer(game$TO)
            game$TO[is.na(game$TO)] = 0
            game = game[,-1]
            game[is.na(game)] = 0
            
            # ToP to seconds
            game$ToP = as.numeric(ms(game$ToP))
            game = game %>% mutate_at(1:dim(game)[2],as.numeric) 
            
            game$X3DRate = game$`3DConv`/game$`3DAtt`

            
            # Reduce to model variables and take average
            #########################
            
            Sys.sleep(5)
            
            
            }
            
          
            games
            
            # home team wins 53% of the time in the NFL on average
          
            
            
          x =  game %>% subset(colnames(game) %in% c(`1stD`,PassY,PA,SkYds,QBR,RA,PntYds,ToP,X3DRate))
            
            

  