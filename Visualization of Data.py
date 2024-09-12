import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


## Pull in master file of all data
final_table_2010 = '/Users/serdarevichar/Library/CloudStorage/GoogleDrive-serdarevichar@gmail.com/.shortcut-targets-by-id/1-15YoYpgC0Rx2M_K-tUhgT52UeAZD80z/Football Model/finaltable-2010-2022.csv'
df = pd.read_csv(final_table_2010)
df = df.drop('Unnamed: 0', axis = 1)


## Histogram of QBR
plt.figure(dpi = 300, figsize = (5,3))
plt.hist(df['QBR'],
         bins = list(np.linspace(0,160, num = 20, endpoint = True)),
         color = 'cornflowerblue',
         density = False,
         edgecolor = 'black')
plt.title('Frequency of QBR')
plt.xlabel('QBR')
plt.ylabel('Frequency')


## Histogram of First Downs
plt.figure(dpi = 300, figsize = (5,3))
plt.hist(df['X1stD'],
         bins = list(np.linspace(0,40, num = 20, endpoint = True)),
         color = 'cornflowerblue',
         density = False,
         edgecolor = 'black')
plt.title('Frequency of First Downs')
plt.xlabel('First Downs')
plt.ylabel('Frequency')


## Histogram of Total Yards
plt.figure(dpi = 300, figsize = (5,3))
plt.hist(df['TotYd'],
         bins = list(np.linspace(50,650, num = 20, endpoint = True)),
         color = 'cornflowerblue',
         density = False,
         edgecolor = 'black')
plt.title('Frequency of Yards')
plt.xlabel('Yards')
plt.ylabel('Frequency')


## Scatter plot of QBR vs Pts
plt.figure(dpi=300,figsize = (5,3))
plt.scatter(df['QBR'],df['Pts'],
            color = 'cornflowerblue',
            alpha = 0.3)
plt.xlabel('QBR')
plt.ylabel('Points')
plt.title('Scatter plot of QBR vs Points')


## 2 Dimensional Histogram of QBR and PTS
plt.figure(dpi=300, figsize = (5,3))
plt.hist2d(df['QBR'],df['Pts'],
           bins = 15,
           cmap = 'Blues')
plt.xlabel('QBR')
plt.ylabel('Points')
plt.title('Histogram of QBR vs Points')
























