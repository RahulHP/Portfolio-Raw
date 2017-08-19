
# Steam Game Recommender
#### Capstone Project, Machine Learning Engineer Nanodegree

## Project Overview
Steam is a gaming platform where multiple games on different platforms (including PC, Mac, and Linux) are sold to players. It also has a social component including streaming and community forums. Many video game providers also hold frequent discounts on the offered games. Users can also see recommendations based on the games they have played already.

Improving the recommendations shown to a user can increase user engagement and increase revenue of the game producer (and Steam too). Given the huge variety of games available across various platforms, targeting the correct user who is likely to buy the game, conduct micro-transactions, and/or recommend the game to his friends is a very lucrative problem for game producers. Since Steam receives a cut on games sold through Steam, Valve also has an interest in properly recommending games.

## Problem Statement

Create a recommendation system which can predict user engagement with games (the scoring function which is defined below) given playtime data about the games and the user. Top-5 games will be recommended to the user along with the predicted score (defined above). This score can be used by companies to decide revenues. Eg. If a user plays a game (eg Candy Crush) very much as compared to other players, similar long-term games with micro-transactions can be suggested to him. On the other hand, if a user plays a lot of different games but doesn’t spend a lot of time on any of them, another variety of games with one-time prices can be shown to him. 


## Scoring Function

A scoring function will be defined to incorporate the wide range in hours played across different users and different games. This will be an attempt to reduce skew and normalise the given data.

$ score = personal \ ratio * normalised \ hours $

Example
If user XYZ has played game A for 20 hours and he has totally played 100 hours on Steam:

$ personal \ ratio = \frac{20}{100} = 0.2 $

But what if game A is a singleplayer campaign which only lasts around 10 hours? Then user XYZ seems to like game A a lot compared to other players. It is assumed that the players playing game A are normally distributed.
If game A is distributed with a mean of 10 hours and a standard deviation of 5 hours, 
$ normalised \ hours = \frac{20-10}{5} = 2 $

$ score = 0.2 * 2 = 0.4 $

## Datasets
- [Steam Video Games dataset](https://www.kaggle.com/tamber/steam-video-games)

This dataset provided by [Tamber](https://tamber.com/) contains user playtime data for different Steam games, ie the hours a particular user has played a particular game.

To quote the content description from the Kaggle page - “This dataset is a list of user behaviors, with columns: user-id, game-title, behavior-name, value. The behaviors included are 'purchase' and 'play'. The value indicates the degree to which the behavior was performed - in the case of 'purchase' the value is always 1, and in the case of 'play' the value represents the number of hours the user has played the game.”

- [Internet Game Database API](https://www.igdb.com/api)

Instead of using DBPedia as planned in the original capstone proposal, the Internet Game Database API was used since it contained detailed game information and had a good API as compared to DBPedia which would have heavy SPARQL queries.


## Evaluation Metrics
Instead of using “Precision at N” as used by [Kevin Wong](http://www.nextvideogame.com/methodology.html#evaluation), an RMSE(Root Mean Squared Error) score will be used since the predicted score is more important in this case than just predicting whether the game is played or not. Eg. Suggesting “Candy Crush” to a player who plays such micro-transaction games for a long time (and pays for the in game boosts) will generate more revenue than suggesting a game which has a small one time price. 


## Proposed Solution

Collaborative filtering (explained below) will be used to fill in the missing user-game score matrix. The top 5 games (arranged in order of descending scores) will then be chosen and shown to the user as recommendations. Matrix factorization will be used to characterise the latent features between users and games. Since game data is also available (from IGDB) factorisation machine models will also be used to improve performance.

Since explicit data is available (in the form of the defined scores), a [factorisation Recommender](https://turi.com/products/create/docs/generated/graphlab.recommender.factorization_recommender.create.html) will be used as suggested [here](https://timchen1.gitbooks.io/graphlab/content/recommender/choosing-a-model.html). Note : An item similarity recommender will not be used since it can not incorporate user and game specific information (as noted in the relevant documentation Notes section). Parameter tuning will then be done to improve performance. 

### Collaborative Filtering
| Player | Civ 5 | Empire TW | EU IV | CS GO | CoD MW2 | Candy Crush | 
| -- | -- | -- | -- | -- | -- | -- |
| A | 5 | 4 | ?? | ?? | 1 | ?? |
| B | 4 | 5 | 5 | 1 | 1 | ?? |
| C | 2 | 1 | 3 | 5 | 4 | ?? |
| D | ?? | ?? | ?? | ?? | ?? | 5 |

| Genre | Game Code | Full Form | 
| -- | -- | -- |
| Strategy | Civ 5 | Sid Meier’s Civilisation 5 |
| Strategy | Empire TW | Empire Total War |
| Strategy | EU IV | Europa Universalis IV |
| First-player Shooter | CS GO | Counter Strike GO |
| First-player Shooter | CoD MW2 |  Call of Duty : Modern Warfare 2 |
| Mobile | Candy Crush | Candy Crush |
Suppose we want to predict games for player A based on the above table. The scores range from 1-5 (with 5 being the highest) and a '??' denotes that the player hasn't played the game.

Player A can have roughly 3 kinds of correlations with other players:
1. Positive Correlation - Player A and player B have high scores for strategy games and low scores for first player shooters. Therefore based on player B, we’ll suggest EU IV to A but we won’t suggest CS Go.

2. Negative Correlation - Player A and player C have the opposite tastes. C likes first player shooters but doesn’t like strategy games. It is the opposite with player A. Therefore based on player C, we won’t suggest the games C likes to player A.

3. Zero Correlation - Player D only likes mobile games and has no ratings about other games. Therefore, player D’s ratings won’t be used to suggest any games to A.

This is called collaborative filtering where the system finds users with similar tastes as the target user and then recommends items which those users have used (but the target user hasn’t). The correlation coefficients for different players can be input to learning algorithms to create predictions for player A.

#### Adding genre information
It is likely that a player who plays strategy games like Sid Meier's Civilization V will also be interested in playing Empire Total War. Similarly, a player who plays multiplayer first-person shooters like Counter-Strike Global Offensive a lot may also want to play Call of Duty Modern Warfare 2. So the playtime data available is supplemented by adding genre information from IGDB. 


## Benchmark
There are 40 million player-game interactions possible in this dataset. But only 70,000 interactions are present in the entire dataset (even before splitting into train-CV-test). Due to this sparsity (0.17% filled entries), it is likely that a simpler model may perform better than an advanced model. As a benchmark, a simple popularity recommender will be used which will just use the average of the scores as the output. It is not tailored to individuals and won’t use the genre data.


$ 3,500 \ games * 11,350 \ players = 40,860,000 \ interactions $

## Data Preprocessing

The dataset provided by Tamber had 200,000 interactions out of which only 70,489 interactions contained playtime data. Out of these 70,489 interactions, there were duplicate entries for 12 player-games. The duplicate entries were removed and only the last occurrences were kept.



```python
import pandas as pd
```


```python
rawdata = pd.read_csv('data/steam-200k.csv',header=None,
                      names=['user-id','game-title','behavior-name','hours','null'])
rawdata[:5]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user-id</th>
      <th>game-title</th>
      <th>behavior-name</th>
      <th>hours</th>
      <th>null</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>151603712</td>
      <td>The Elder Scrolls V Skyrim</td>
      <td>purchase</td>
      <td>1.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>151603712</td>
      <td>The Elder Scrolls V Skyrim</td>
      <td>play</td>
      <td>273.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>151603712</td>
      <td>Fallout 4</td>
      <td>purchase</td>
      <td>1.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>151603712</td>
      <td>Fallout 4</td>
      <td>play</td>
      <td>87.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>151603712</td>
      <td>Spore</td>
      <td>purchase</td>
      <td>1.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



This dataset is a list of user behaviors, with columns: user-id, game-title, behavior-name, hours and null. The behaviors included are 'purchase' and 'play'. The value in the 'hours' column indicates the degree to which the behavior was performed - in the case of 'purchase' the value is always 1, and in the case of 'play' the value represents the number of hours the user has played the game.

Since only playtime data is required, the rows which have the 'purchase' behavior are removed.


```python
playtimedata=rawdata[rawdata['behavior-name']=='play']
playtimedata.drop(['behavior-name','null'],axis=1,inplace=True)
playtimedata[:5]
```

    /home/rahulhp/anaconda2/envs/gl-env/lib/python2.7/site-packages/ipykernel/__main__.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      from ipykernel import kernelapp as app





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user-id</th>
      <th>game-title</th>
      <th>hours</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>151603712</td>
      <td>The Elder Scrolls V Skyrim</td>
      <td>273.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>151603712</td>
      <td>Fallout 4</td>
      <td>87.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>151603712</td>
      <td>Spore</td>
      <td>14.9</td>
    </tr>
    <tr>
      <th>7</th>
      <td>151603712</td>
      <td>Fallout New Vegas</td>
      <td>12.1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>151603712</td>
      <td>Left 4 Dead 2</td>
      <td>8.9</td>
    </tr>
  </tbody>
</table>
</div>



### Removing duplicate rows
In the dataset, 12 user-game interactions have 2 values.


```python
duplicates = playtimedata.groupby(['user-id','game-title']).count().reset_index()
duplicates[duplicates['hours']>1]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user-id</th>
      <th>game-title</th>
      <th>hours</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8827</th>
      <td>28472068</td>
      <td>Grand Theft Auto III</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8829</th>
      <td>28472068</td>
      <td>Grand Theft Auto San Andreas</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8830</th>
      <td>28472068</td>
      <td>Grand Theft Auto Vice City</td>
      <td>2</td>
    </tr>
    <tr>
      <th>11066</th>
      <td>33865373</td>
      <td>Sid Meier's Civilization IV</td>
      <td>2</td>
    </tr>
    <tr>
      <th>18414</th>
      <td>50769696</td>
      <td>Grand Theft Auto San Andreas</td>
      <td>2</td>
    </tr>
    <tr>
      <th>22653</th>
      <td>59925638</td>
      <td>Tom Clancy's H.A.W.X. 2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>27873</th>
      <td>71411882</td>
      <td>Grand Theft Auto III</td>
      <td>2</td>
    </tr>
    <tr>
      <th>27944</th>
      <td>71510748</td>
      <td>Grand Theft Auto San Andreas</td>
      <td>2</td>
    </tr>
    <tr>
      <th>43939</th>
      <td>118664413</td>
      <td>Grand Theft Auto San Andreas</td>
      <td>2</td>
    </tr>
    <tr>
      <th>51079</th>
      <td>148362155</td>
      <td>Grand Theft Auto San Andreas</td>
      <td>2</td>
    </tr>
    <tr>
      <th>58054</th>
      <td>176261926</td>
      <td>Sid Meier's Civilization IV</td>
      <td>2</td>
    </tr>
    <tr>
      <th>58055</th>
      <td>176261926</td>
      <td>Sid Meier's Civilization IV Beyond the Sword</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



We are keeping only the last value in this case.


```python
playtimedata.drop_duplicates(subset=['user-id','game-title'],keep='last',inplace=True)
```

    /home/rahulhp/anaconda2/envs/gl-env/lib/python2.7/site-packages/pandas/util/decorators.py:91: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      return func(*args, **kwargs)


## Adding genre and developer information

Since there are 3600 games and the API had a rate limitation of 7000 queries per day, the following code which queries the API for genre + developer information has been commented out. The results have been stored in a pickle file for later usage. The API key has also been removed for privacy reasons. The API key can be obtained from the [Mashape website](https://market.mashape.com/igdbcom/internet-game-database)


```python
import requests
import sys
import pickle
```

This helper function was used to get the developer and genre information


```python
def get_game_data(search_string):
    API_URL = "https://igdbcom-internet-game-database-v1.p.mashape.com/games/?"
    FIELDS = "fields=name%2Cslug%2Cdevelopers%2Cgenres&limit=1&offset=0&search="
    
    search_url = API_URL + FIELDS + search_string
    headers={
        "X-Mashape-Key": "INSERT-KEY-HERE",
        "Accept": "application/json"
      }
    try:
        r = requests.get(search_url,headers=headers)
        d = dict()
        d['name'] = search_string
        d['developers'] = r.json()[0].get(['developers'][0],0)
        d['genres'] = r.json()[0].get(['genres'][0],0)
        return d
    except:
        print "\tUnexpected error: "+str(sys.exc_info()[0])
        return None
```

The below code has which called the API for the games been disabled.
games = list(gamedata['game-title'])
dictlist = []
count=0
for i in games:
    print count,'\t',i
    d = get_game_data(i)
    if d is not None:
        dictlist.append(d)
    count+=1
pickle.dump(dictlist,open("gameinfo.p","wb"))

```python
gameinfo = pickle.load(open("gameinfo.p","rb"))
```

Example output from the API:


```python
print gameinfo[12]
```

    {'genres': [12, 31], 'developers': [1243], 'name': 'The Incredible Adventures of Van Helsing III'}



```python
genres = set()
for game in gameinfo:
    if type(game['genres'])==list:
        for i in game['genres']:
            genres.add(i)
print len(genres)
```

    20


There are 20 distinct genres present in the dataset.


```python
developers = set()
for game in gameinfo:
    if type(game['developers'])==list:
        for i in game['developers']:
            developers.add(i)
print len(developers)
```

    1858


There are 1858 distinct developers. Since there are 3600 games in the dataset, on average each developer has made 2 games. Due to the very sparse data in this field, this field has been removed from future calculations. Only the genre information will be used.

### Removing extra games from playtimedata


```python
print len(gameinfo)
```

    3581


Information for only 3581 games was obtained from IGDB. We'll be removing the data for the other 19 games before proceeding.


```python
igdbgames = set([i['name'] for i in gameinfo])
tambergames = set(playtimedata['game-title'].unique())

print 'Data from these 19 games was unavailable: ' + ' '.join(x for x in list(tambergames.difference(igdbgames)))
```

    Data from these 19 games was unavailable: Batla Dethroned! Voxelized Aberoth Jumpdrive Deepworld Squirreltopia Breezeblox Cosmophony Rexaura Immune KWAAN CRYENGINE Burstfire Unium ROCKETSROCKETSROCKETS Rotieer Alganon Motorbike


A right join is used to keep only those games which occur in the IGDB dataset.


```python
igdbgamesDF = pd.DataFrame({'game-title':[i['name'] for i in gameinfo]})
finalplaytimedata = pd.merge(playtimedata,igdbgamesDF,how='right')
```

### Creating a game information table
Since only the genre information is being used, basic one-hot encoding is used to create the item data table. This dataframe is used in the factorisation recommender.


```python
genrelist = list()
for i in gameinfo:
    genrelist.append({'genres':i['genres'],'game-title':i['name']})
genreDF = pd.DataFrame(genrelist)

genre_encoded = pd.get_dummies(genreDF['genres'].apply(pd.Series).stack(),prefix='genre').sum(level=0)

genre_dummies = pd.concat([genreDF,genre_encoded],axis=1).drop('genres',axis=1)
genre_dummies[:3]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>game-title</th>
      <th>genre_0.0</th>
      <th>genre_2.0</th>
      <th>genre_4.0</th>
      <th>genre_5.0</th>
      <th>genre_7.0</th>
      <th>genre_8.0</th>
      <th>genre_9.0</th>
      <th>genre_10.0</th>
      <th>genre_11.0</th>
      <th>...</th>
      <th>genre_14.0</th>
      <th>genre_15.0</th>
      <th>genre_16.0</th>
      <th>genre_24.0</th>
      <th>genre_25.0</th>
      <th>genre_26.0</th>
      <th>genre_30.0</th>
      <th>genre_31.0</th>
      <th>genre_32.0</th>
      <th>genre_33.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10 Second Ninja</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Toy Soldiers War Chest</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Infestation Survivor Stories</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 22 columns</p>
</div>



## Exploratory Data Analysis

### Player Data


```python
userdata=finalplaytimedata[['user-id','game-title']].groupby('user-id').count().reset_index() \
.rename(columns={'game-title':'games-played'}).sort_values(by='games-played',ascending=False)

print len(userdata)
userdata[:5]

```

    11350





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user-id</th>
      <th>games-played</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1478</th>
      <td>62990992</td>
      <td>496</td>
    </tr>
    <tr>
      <th>175</th>
      <td>11403772</td>
      <td>314</td>
    </tr>
    <tr>
      <th>4506</th>
      <td>138941587</td>
      <td>298</td>
    </tr>
    <tr>
      <th>989</th>
      <td>47457723</td>
      <td>297</td>
    </tr>
    <tr>
      <th>1070</th>
      <td>49893565</td>
      <td>297</td>
    </tr>
  </tbody>
</table>
</div>




```python
%matplotlib inline
import seaborn as sns
sns.set(color_codes=True)
```


```python
sns.boxplot(userdata['games-played'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7ff3bc4ac590>




![png](images/output_43_1.png)


There are 11350 unique players in this dataset. But as seen in the boxplot, there are a lot of extreme players in this dataset but a majority of the players (~80%) have played less than 5 games.


```python
userdata['games-played'].mean()
```




    6.209427312775331




```python
userdata['games-played'].std()
```




    17.75490607377872




```python
userdata['games-played'].median()
```




    1.0




```python
for threshold in [1,5,10,25,50,100,250,500]:
    playercount = len(userdata[userdata['games-played']<=threshold])
    playerpercent = 100.0*playercount/len(userdata)
    print 'Number of players with <= %4d games: %5d (%3.0f%%)' % (threshold, playercount, playerpercent)
```

    Number of players with <=    1 games:  6559 ( 58%)
    Number of players with <=    5 games:  9221 ( 81%)
    Number of players with <=   10 games:  9978 ( 88%)
    Number of players with <=   25 games: 10753 ( 95%)
    Number of players with <=   50 games: 11069 ( 98%)
    Number of players with <=  100 games: 11267 ( 99%)
    Number of players with <=  250 games: 11343 (100%)
    Number of players with <=  500 games: 11350 (100%)


### Game Data


```python
gamedata=finalplaytimedata[['user-id','game-title']].groupby('game-title').count().reset_index()\
.rename(columns={'user-id':'players'}).sort_values('players',ascending=False)

print len(gamedata)
```

    3581



```python
gamedata['percent'] = 100*gamedata['players']/sum(gamedata['players'])
top10=gamedata[:10]
top10
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>game-title</th>
      <th>players</th>
      <th>percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>913</th>
      <td>Dota 2</td>
      <td>4841</td>
      <td>6.873394</td>
    </tr>
    <tr>
      <th>2977</th>
      <td>Team Fortress 2</td>
      <td>2323</td>
      <td>3.298264</td>
    </tr>
    <tr>
      <th>666</th>
      <td>Counter-Strike Global Offensive</td>
      <td>1377</td>
      <td>1.955105</td>
    </tr>
    <tr>
      <th>3332</th>
      <td>Unturned</td>
      <td>1069</td>
      <td>1.517798</td>
    </tr>
    <tr>
      <th>1721</th>
      <td>Left 4 Dead 2</td>
      <td>801</td>
      <td>1.137283</td>
    </tr>
    <tr>
      <th>668</th>
      <td>Counter-Strike Source</td>
      <td>715</td>
      <td>1.015178</td>
    </tr>
    <tr>
      <th>3050</th>
      <td>The Elder Scrolls V Skyrim</td>
      <td>677</td>
      <td>0.961224</td>
    </tr>
    <tr>
      <th>1304</th>
      <td>Garry's Mod</td>
      <td>666</td>
      <td>0.945606</td>
    </tr>
    <tr>
      <th>663</th>
      <td>Counter-Strike</td>
      <td>568</td>
      <td>0.806463</td>
    </tr>
    <tr>
      <th>2675</th>
      <td>Sid Meier's Civilization V</td>
      <td>554</td>
      <td>0.786585</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.barplot(x="game-title",y="percent",data=top10)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7ff3ad2ccd10>




![png](images/output_52_1.png)



```python
for threshold in [10,50,100,200,500,5000]:
    playercount = len(gamedata[gamedata['players']<=threshold])
    playerpercent = 100.0*playercount/len(gamedata)
    print 'Number of games with <= %4d players: %5d (%3.2f%%)' % (threshold, playercount, playerpercent) 
```

    Number of games with <=   10 players:  2593 (72.41%)
    Number of games with <=   50 players:  3297 (92.07%)
    Number of games with <=  100 players:  3448 (96.29%)
    Number of games with <=  200 players:  3541 (98.88%)
    Number of games with <=  500 players:  3571 (99.72%)
    Number of games with <= 5000 players:  3581 (100.00%)


The distribution of players is very uneven. DOTA 2 has double the players as the next game TF2. Only 6 games have more than 1% of the players in this dataset.

## Genre data


```python
combine=pd.merge(gamedata,genre_dummies,on='game-title')
```


```python
genregamecount=genre_dummies.filter(regex='genre').sum(axis=0)
```


```python
genreplayercount=combine.filter(regex='genre').mul(combine['players'],axis=0).sum(axis=0)
```


```python
genrecombined=pd.DataFrame({'games':genregamecount,'players':genreplayercount})
```


```python
genrecombined['avg-players']=genrecombined['players']/genrecombined['games']
```


```python
genrecombined.sort('games')
```

    /home/rahulhp/anaconda2/envs/gl-env/lib/python2.7/site-packages/ipykernel/__main__.py:1: FutureWarning: sort(columns=....) is deprecated, use sort_values(by=.....)
      if __name__ == '__main__':





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>games</th>
      <th>players</th>
      <th>avg-players</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>genre_30.0</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>genre_26.0</th>
      <td>6.0</td>
      <td>14.0</td>
      <td>2.333333</td>
    </tr>
    <tr>
      <th>genre_7.0</th>
      <td>28.0</td>
      <td>316.0</td>
      <td>11.285714</td>
    </tr>
    <tr>
      <th>genre_4.0</th>
      <td>62.0</td>
      <td>1908.0</td>
      <td>30.774194</td>
    </tr>
    <tr>
      <th>genre_14.0</th>
      <td>101.0</td>
      <td>1281.0</td>
      <td>12.683168</td>
    </tr>
    <tr>
      <th>genre_25.0</th>
      <td>124.0</td>
      <td>2964.0</td>
      <td>23.903226</td>
    </tr>
    <tr>
      <th>genre_16.0</th>
      <td>130.0</td>
      <td>2463.0</td>
      <td>18.946154</td>
    </tr>
    <tr>
      <th>genre_10.0</th>
      <td>140.0</td>
      <td>2381.0</td>
      <td>17.007143</td>
    </tr>
    <tr>
      <th>genre_24.0</th>
      <td>147.0</td>
      <td>11998.0</td>
      <td>81.619048</td>
    </tr>
    <tr>
      <th>genre_33.0</th>
      <td>157.0</td>
      <td>1667.0</td>
      <td>10.617834</td>
    </tr>
    <tr>
      <th>genre_2.0</th>
      <td>186.0</td>
      <td>1269.0</td>
      <td>6.822581</td>
    </tr>
    <tr>
      <th>genre_11.0</th>
      <td>219.0</td>
      <td>8257.0</td>
      <td>37.703196</td>
    </tr>
    <tr>
      <th>genre_8.0</th>
      <td>248.0</td>
      <td>4672.0</td>
      <td>18.838710</td>
    </tr>
    <tr>
      <th>genre_9.0</th>
      <td>286.0</td>
      <td>4131.0</td>
      <td>14.444056</td>
    </tr>
    <tr>
      <th>genre_13.0</th>
      <td>333.0</td>
      <td>5989.0</td>
      <td>17.984985</td>
    </tr>
    <tr>
      <th>genre_0.0</th>
      <td>388.0</td>
      <td>2999.0</td>
      <td>7.729381</td>
    </tr>
    <tr>
      <th>genre_15.0</th>
      <td>534.0</td>
      <td>11676.0</td>
      <td>21.865169</td>
    </tr>
    <tr>
      <th>genre_12.0</th>
      <td>535.0</td>
      <td>16227.0</td>
      <td>30.330841</td>
    </tr>
    <tr>
      <th>genre_5.0</th>
      <td>569.0</td>
      <td>24645.0</td>
      <td>43.312830</td>
    </tr>
    <tr>
      <th>genre_31.0</th>
      <td>1065.0</td>
      <td>19653.0</td>
      <td>18.453521</td>
    </tr>
    <tr>
      <th>genre_32.0</th>
      <td>1073.0</td>
      <td>15798.0</td>
      <td>14.723206</td>
    </tr>
  </tbody>
</table>
</div>



Genre 24 (Tactical games) has only 147 games but it has the highest average players per game. This is because DOTA 2 (the most popular game) falls in this genre.

## Scoring Function

Since a score was the target variable, it has to be created for the whole dataset together (since we needed scores for the test interactions too). But to avoid this test data from leaking into the training data, the scores for the training data were also calculated separately. 

A separate create_score() function was created for this function. Splitting the dataset properly so that no leakage occurred was a major issue during this step. Sanity checks and debugging statements were used at each step to ensure that the correct number of rows existed at each step.

![][scoring]

[scoring]: images/Scoring.jpg


```python
import graphlab as gl
```


```python
playtimedataSF = gl.SFrame(finalplaytimedata)
genreSF = gl.SFrame(genre_dummies)
```


```python
def create_score(dataset):
    #print 'Creating scores.'
    #print "There are " + str(dataset.shape[0]) + " interactions in this dataset."
    totalhours=dataset.groupby(key_columns='user-id',operations={'total_hours':gl.aggregate.SUM('hours')})
    #print "There are " + str(totalhours.shape[0]) + " unique users in this dataset."
    
    hoursjoined = dataset.join(totalhours,on='user-id')
    hoursjoined['personal-ratio']=hoursjoined['hours']/hoursjoined['total_hours']
    hoursjoined.remove_columns(['hours','total_hours'])
    #print "There are " + str(hoursjoined.shape[0]) + " interactions in the personal-ratio dataset."
    
    gamedist=dataset.groupby(key_columns='game-title',operations={'hours-avg':gl.aggregate.MEAN('hours'),
                                                           'hours-std':gl.aggregate.STD('hours')})
    #print "There are " + str(gamedist.shape[0]) + " unique games in this dataset."
    
    playerdist = dataset.join(gamedist,on='game-title')
    playerdist['normalised-ratio']=(playerdist['hours']-playerdist['hours-avg'])/playerdist['hours-std']
    playerdist = playerdist.fillna('normalised-ratio',0)
    playerdist.remove_columns(['hours','hours-std','hours-avg'])
    
    #print "There are " + str(playerdist.shape[0]) + " interactions in the player-distribution dataset."
    
    joined = playerdist.join(hoursjoined)
    joined['score']=joined['normalised-ratio']*joined['personal-ratio']
    joined.remove_columns(['normalised-ratio','personal-ratio'])
    #print "There are " + str(joined.shape[0]) + " scores in the final scoring dataset."
    return joined
```


```python
train, other = gl.recommender.util.random_split_by_user(playtimedataSF,user_id='user-id',item_id='game-title',random_seed=12)
print train.shape
print other.shape
```

    (69232, 3)
    (1199, 3)



```python
train_score = create_score(train)

whole_score = create_score(playtimedataSF)

other_score = other.join(whole_score,how='left')
other_score = other_score.remove_column('hours')
cv_score,test_score = gl.recommender.util.random_split_by_user(other_score,user_id='user-id',item_id='game-title',random_seed=12)

print train_score.shape
print cv_score.shape
print test_score.shape
```

    (69232, 3)
    (970, 3)
    (229, 3)


## Benchmark model

A basic popularity recommender is created. This will be used as the benchmark model and the factorisation recommender performance will be compared to this model.


```python
basemodel = gl.recommender.popularity_recommender.create(train_score,user_id='user-id'
                                                         ,item_id='game-title',target='score')
```


<pre>Recsys training: model = popularity</pre>



<pre>Preparing data set.</pre>



<pre>    Data has 69232 observations with 11221 users and 3558 items.</pre>



<pre>    Data prepared in: 0.316876s</pre>



<pre>69232 observations to process; with 3558 unique items.</pre>



```python
gl.evaluation.rmse(basemodel.predict(cv_score), cv_score['score'])
```




    0.4796248793464026



The popularity recommender has an RMSE of 0.479 on the cross validation data.

## Factorisation Recommender


```python
simplemodel = gl.recommender.factorization_recommender.create(train_score,user_id='user-id',item_id='game-title',target='score',item_data=genreSF)
```


<pre>Recsys training: model = factorization_recommender</pre>



<pre>Preparing data set.</pre>



<pre>    Data has 69232 observations with 11221 users and 3581 items.</pre>



<pre>    Data prepared in: 0.250445s</pre>



<pre>Training factorization_recommender for recommendations.</pre>



<pre>+--------------------------------+--------------------------------------------------+----------+</pre>



<pre>| Parameter                      | Description                                      | Value    |</pre>



<pre>+--------------------------------+--------------------------------------------------+----------+</pre>



<pre>| num_factors                    | Factor Dimension                                 | 8        |</pre>



<pre>| regularization                 | L2 Regularization on Factors                     | 1e-08    |</pre>



<pre>| solver                         | Solver used for training                         | adagrad  |</pre>



<pre>| linear_regularization          | L2 Regularization on Linear Coefficients         | 1e-10    |</pre>



<pre>| side_data_factorization        | Assign Factors for Side Data                     | True     |</pre>



<pre>| max_iterations                 | Maximum Number of Iterations                     | 50       |</pre>



<pre>+--------------------------------+--------------------------------------------------+----------+</pre>



<pre>  Optimizing model using SGD; tuning step size.</pre>



<pre>  Using 10000 / 69232 points for tuning the step size.</pre>



<pre>+---------+-------------------+------------------------------------------+</pre>



<pre>| Attempt | Initial Step Size | Estimated Objective Value                |</pre>



<pre>+---------+-------------------+------------------------------------------+</pre>



<pre>| 0       | 2.17391           | Not Viable                               |</pre>



<pre>| 1       | 0.543478          | Not Viable                               |</pre>



<pre>| 2       | 0.13587           | Not Viable                               |</pre>



<pre>| 3       | 0.0339674         | 0.15806                                  |</pre>



<pre>| 4       | 0.0169837         | 0.174171                                 |</pre>



<pre>| 5       | 0.00849185        | 0.193037                                 |</pre>



<pre>| 6       | 0.00424592        | 0.199002                                 |</pre>



<pre>+---------+-------------------+------------------------------------------+</pre>



<pre>| Final   | 0.0339674         | 0.15806                                  |</pre>



<pre>+---------+-------------------+------------------------------------------+</pre>



<pre>Starting Optimization.</pre>



<pre>+---------+--------------+-------------------+-----------------------+-------------+</pre>



<pre>| Iter.   | Elapsed Time | Approx. Objective | Approx. Training RMSE | Step Size   |</pre>



<pre>+---------+--------------+-------------------+-----------------------+-------------+</pre>



<pre>| Initial | 195us        | 0.232142          | 0.481811              |             |</pre>



<pre>+---------+--------------+-------------------+-----------------------+-------------+</pre>



<pre>| 1       | 333.179ms    | 0.457293          | 0.676228              | 0.0339674   |</pre>



<pre>| 2       | 688.785ms    | DIVERGED          | DIVERGED              | 0.0339674   |</pre>



<pre>| RESET   | 781.723ms    | 0.232142          | 0.481811              |             |</pre>



<pre>| 1       | 1.14s        | 0.233856          | 0.483586              | 0.0169837   |</pre>



<pre>| 2       | 1.47s        | 0.227827          | 0.477311              | 0.0169837   |</pre>



<pre>| 3       | 1.91s        | 0.230452          | 0.480053              | 0.0169837   |</pre>



<pre>| 4       | 2.19s        | 0.434907          | 0.659473              | 0.0169837   |</pre>



<pre>| 5       | 2.53s        | DIVERGED          | DIVERGED              | 0.0169837   |</pre>



<pre>| RESET   | 2.60s        | 0.232143          | 0.481812              |             |</pre>



<pre>| 1       | 2.90s        | 0.232859          | 0.482555              | 0.00849185  |</pre>



<pre>| 3       | 3.46s        | 0.226978          | 0.476422              | 0.00849185  |</pre>



<pre>| 6       | 4.47s        | 0.22236           | 0.47155               | 0.00849185  |</pre>



<pre>| 8       | 5.06s        | 0.220761          | 0.469852              | 0.00849185  |</pre>



<pre>| 13      | 6.38s        | 0.235932          | 0.485727              | 0.00849185  |</pre>



<pre>| 16      | 7.39s        | DIVERGED          | DIVERGED              | 0.00849185  |</pre>



<pre>| RESET   | 7.46s        | 0.23214           | 0.481809              |             |</pre>



<pre>| 2       | 8.08s        | 0.230712          | 0.480324              | 0.00424592  |</pre>



<pre>| 7       | 9.38s        | 0.226936          | 0.476378              | 0.00424592  |</pre>



<pre>| 12      | 10.78s       | 0.224473          | 0.473786              | 0.00424592  |</pre>



<pre>| 17      | 12.18s       | 0.222414          | 0.471607              | 0.00424592  |</pre>



<pre>| 22      | 13.59s       | 0.22067           | 0.469755              | 0.00424592  |</pre>



<pre>| 27      | 15.14s       | 0.218975          | 0.467948              | 0.00424592  |</pre>



<pre>| 31      | 16.40s       | 0.217855          | 0.466749              | 0.00424592  |</pre>



<pre>| 32      | 16.70s       | 0.21758           | 0.466454              | 0.00424592  |</pre>



<pre>+---------+--------------+-------------------+-----------------------+-------------+</pre>



<pre>Optimization Complete: Maximum number of passes through the data reached (hard limit).</pre>



<pre>Computing final objective value and training RMSE.</pre>



<pre>       Final objective value: 0.217244</pre>



<pre>       Final training RMSE: 0.466094</pre>



```python
gl.evaluation.rmse(simplemodel.predict(cv_score), cv_score['score'])
```




    0.48101388952900964



The factorisation recommender has an RMSE of 0.481 which is comparable to that of the popularity recommender. Hyperparameter tuning will now be done to see if the performance improves.

## Hyperparameter tuning


```python
from graphlab import model_parameter_search
```


```python
def rmse_evaluator(model,train,test):
    rmse = gl.evaluation.rmse(model.predict(test),test['score'])
    return {'rmse':rmse}
```


```python
params = {'user_id':'user-id', 'item_id':'game-title','target':'score','item_data': [genreSF],
         'linear_regularization':[1e-05,1e-07,1e-09],
         'max_iterations':50,
         'num_factors':[8,16,32],
         'regularization':[1e-06,1e-07,1e-08]}
job = model_parameter_search.create((train_score,cv_score),
                                   gl.recommender.factorization_recommender.create,
                                   params,
                                   evaluator = rmse_evaluator,
                                   perform_trial_run=True,
                                   return_model=True)
```

    [INFO] graphlab.deploy.job: Validating job.
    [INFO] graphlab.deploy.job: Creating a LocalAsync environment called 'async'.
    [INFO] graphlab.deploy.map_job: Validation complete. Job: 'Model-Parameter-Search-Aug-14-2017-20-44-4100000' ready for execution
    [INFO] graphlab.deploy.map_job: Job: 'Model-Parameter-Search-Aug-14-2017-20-44-4100000' scheduled.
    [INFO] graphlab.deploy.job: Validating job.
    [INFO] graphlab.deploy.map_job: A job with name 'Model-Parameter-Search-Aug-14-2017-20-44-4100000' already exists. Renaming the job to 'Model-Parameter-Search-Aug-14-2017-20-44-4100000-97966'.
    [INFO] graphlab.deploy.map_job: Validation complete. Job: 'Model-Parameter-Search-Aug-14-2017-20-44-4100000-97966' ready for execution
    [INFO] graphlab.deploy.map_job: Job: 'Model-Parameter-Search-Aug-14-2017-20-44-4100000-97966' scheduled.



```python
results = job.get_results()
```


```python
job.get_best_params(metric='rmse',ascending=True)
```




    {'item_data': Columns:
     	game-title	str
     	genre_0.0	float
     	genre_2.0	float
     	genre_4.0	float
     	genre_5.0	float
     	genre_7.0	float
     	genre_8.0	float
     	genre_9.0	float
     	genre_10.0	float
     	genre_11.0	float
     	genre_12.0	float
     	genre_13.0	float
     	genre_14.0	float
     	genre_15.0	float
     	genre_16.0	float
     	genre_24.0	float
     	genre_25.0	float
     	genre_26.0	float
     	genre_30.0	float
     	genre_31.0	float
     	genre_32.0	float
     	genre_33.0	float
     
     Rows: 3581
     
     Data:
     +-------------------------------+-----------+-----------+-----------+-----------+
     |           game-title          | genre_0.0 | genre_2.0 | genre_4.0 | genre_5.0 |
     +-------------------------------+-----------+-----------+-----------+-----------+
     |        10 Second Ninja        |    0.0    |    0.0    |    0.0    |    0.0    |
     |     Toy Soldiers War Chest    |    0.0    |    0.0    |    0.0    |    1.0    |
     |  Infestation Survivor Stories |    0.0    |    0.0    |    0.0    |    0.0    |
     | Nightmares from the Deep 2... |    0.0    |    0.0    |    0.0    |    0.0    |
     |  Ultimate General Gettysburg  |    0.0    |    0.0    |    0.0    |    0.0    |
     |     Unreal Tournament 2004    |    0.0    |    0.0    |    0.0    |    1.0    |
     |          Toki Tori 2+         |    0.0    |    0.0    |    0.0    |    0.0    |
     |          Life is Hard         |    0.0    |    0.0    |    0.0    |    0.0    |
     |          World of Goo         |    0.0    |    0.0    |    0.0    |    0.0    |
     |      Hollywood Visionary      |    0.0    |    0.0    |    0.0    |    0.0    |
     +-------------------------------+-----------+-----------+-----------+-----------+
     +-----------+-----------+-----------+------------+------------+------------+------------+
     | genre_7.0 | genre_8.0 | genre_9.0 | genre_10.0 | genre_11.0 | genre_12.0 | genre_13.0 |
     +-----------+-----------+-----------+------------+------------+------------+------------+
     |    0.0    |    0.0    |    0.0    |    0.0     |    0.0     |    0.0     |    0.0     |
     |    0.0    |    0.0    |    0.0    |    0.0     |    0.0     |    0.0     |    0.0     |
     |    0.0    |    0.0    |    0.0    |    0.0     |    0.0     |    0.0     |    0.0     |
     |    0.0    |    0.0    |    0.0    |    0.0     |    0.0     |    0.0     |    0.0     |
     |    0.0    |    0.0    |    0.0    |    0.0     |    1.0     |    0.0     |    1.0     |
     |    0.0    |    0.0    |    0.0    |    0.0     |    0.0     |    0.0     |    0.0     |
     |    0.0    |    1.0    |    1.0    |    0.0     |    0.0     |    0.0     |    0.0     |
     |    0.0    |    0.0    |    0.0    |    0.0     |    0.0     |    1.0     |    1.0     |
     |    0.0    |    0.0    |    1.0    |    0.0     |    0.0     |    0.0     |    0.0     |
     |    0.0    |    0.0    |    0.0    |    0.0     |    0.0     |    0.0     |    0.0     |
     +-----------+-----------+-----------+------------+------------+------------+------------+
     +------------+------------+------------+------------+------------+------------+
     | genre_14.0 | genre_15.0 | genre_16.0 | genre_24.0 | genre_25.0 | genre_26.0 |
     +------------+------------+------------+------------+------------+------------+
     |    0.0     |    0.0     |    0.0     |    0.0     |    0.0     |    0.0     |
     |    0.0     |    0.0     |    0.0     |    0.0     |    0.0     |    0.0     |
     |    0.0     |    0.0     |    0.0     |    0.0     |    0.0     |    0.0     |
     |    0.0     |    0.0     |    0.0     |    0.0     |    0.0     |    0.0     |
     |    0.0     |    1.0     |    0.0     |    0.0     |    0.0     |    0.0     |
     |    0.0     |    0.0     |    0.0     |    0.0     |    0.0     |    0.0     |
     |    0.0     |    0.0     |    0.0     |    0.0     |    0.0     |    0.0     |
     |    0.0     |    0.0     |    0.0     |    0.0     |    0.0     |    0.0     |
     |    0.0     |    0.0     |    0.0     |    0.0     |    0.0     |    0.0     |
     |    0.0     |    0.0     |    0.0     |    0.0     |    0.0     |    0.0     |
     +------------+------------+------------+------------+------------+------------+
     +------------+------------+-----+
     | genre_30.0 | genre_31.0 | ... |
     +------------+------------+-----+
     |    0.0     |    0.0     | ... |
     |    0.0     |    0.0     | ... |
     |    0.0     |    1.0     | ... |
     |    0.0     |    1.0     | ... |
     |    0.0     |    0.0     | ... |
     |    0.0     |    0.0     | ... |
     |    0.0     |    1.0     | ... |
     |    0.0     |    0.0     | ... |
     |    0.0     |    0.0     | ... |
     |    0.0     |    1.0     | ... |
     +------------+------------+-----+
     [3581 rows x 22 columns]
     Note: Only the head of the SFrame is printed.
     You can use print_rows(num_rows=m, num_columns=n) to print more rows and columns.,
     'item_id': 'game-title',
     'linear_regularization': 1e-05,
     'max_iterations': 50,
     'num_factors': 16,
     'regularization': 1e-07,
     'target': 'score',
     'user_id': 'user-id'}



The best parameters obtained by hyperparameter tunings are :
- **linear_regularization** : 1e-05
- **num_factors** : 16
- **regularization** : 1e-07


```python
tunedmodel = gl.recommender.factorization_recommender.create(train_score, user_id = 'user-id', item_id = 'game-title',
                                                            target = 'score', item_data = genreSF,
                                                           linear_regularization = 1e-05, max_iterations = 50,
                                                           num_factors = 16, regularization = 1e-07,
                                                           random_seed=123,verbose=False)
```


<pre>Recsys training: model = factorization_recommender</pre>



```python
gl.evaluation.rmse(tunedmodel.predict(cv_score), cv_score['score'])
```




    0.47952367621155995



The tuned factorisation recommender has an RMSE of 0.479 on the cross validation data.

## Model Evaluation

We'll now compare the 3 models on the test dataset


```python
print gl.evaluation.rmse(basemodel.predict(test_score), test_score['score'])
print gl.evaluation.rmse(simplemodel.predict(test_score), test_score['score'])
print gl.evaluation.rmse(tunedmodel.predict(test_score), test_score['score'])
```

    0.256368898906
    0.246270273001
    0.243559527707


| | Popularity Recommender | Factorisation Recommender | Tuned Factorisation Recommender |
| -- | -- | -- | -- |
|CV RMSE | 0.479 | 0.481 | 0.479 |
| Test RMSE | 0.256 | 0.246 | 0.243 |

## Robustness
To test the robustness of the final model, we'll try different splits of the data and see how the CV RMSE changes.


```python
import time
```


```python
def evaluate_models(seed):
    print seed
    train, other = gl.recommender.util.random_split_by_user(playtimedataSF,user_id='user-id',item_id='game-title'
                                                            ,random_seed=seed)
    
    train_score = create_score(train)
    whole_score = create_score(playtimedataSF)
    other_score=other.join(whole_score,how='left')
    other_score.remove_column('hours')
    
    cv_score,test_score = gl.recommender.util.random_split_by_user(other_score,user_id='user-id',item_id='game-title'
                                                                   ,random_seed=seed)
    
    basestart = time.clock()
    basemodel = gl.recommender.popularity_recommender.create(train_score,user_id='user-id',item_id='game-title'
                                                             ,target='score',random_seed=seed,verbose=False)
    baseend = time.clock()
    
    
    tunedstart = time.clock()
    tunedmodel = gl.recommender.factorization_recommender.create(train_score, user_id = 'user-id', item_id = 'game-title',
                                                            target = 'score', item_data = genreSF,
                                                           linear_regularization = 1e-05, max_iterations = 50,
                                                           num_factors = 16, regularization = 1e-07,
                                                                 random_seed=seed, verbose=False)
    tunedend = time.clock()
    
    
    return [baseend-basestart,gl.evaluation.rmse(basemodel.predict(cv_score), cv_score['score'])
            ,tunedend-tunedstart,gl.evaluation.rmse(tunedmodel.predict(cv_score), cv_score['score'])]
    
```


```python
def score_pipeline():

    basetimes = list()
    basescores = list()

    tunedtimes = list()
    tunedscores = list()
    
    for i in xrange(1,21):
        [basetime,basescore,tunedtime,tunedscore] = evaluate_models(i)
        basetimes.append(basetime)
        basescores.append(basescore)
        
        tunedtimes.append(tunedtime)
        tunedscores.append(tunedscore)
        
        
    timesDF = pd.DataFrame({'Base':basetimes,'Tuned':tunedtimes})
    scoresDF = pd.DataFrame({'Base':basescores,'Tuned':tunedscores})
    
    return timesDF, scoresDF
```


```python
timesDF, scoresDF = score_pipeline()
```

    1



<pre>Recsys training: model = popularity</pre>



<pre>Recsys training: model = factorization_recommender</pre>


    2



<pre>Recsys training: model = popularity</pre>



<pre>Recsys training: model = factorization_recommender</pre>


    3



<pre>Recsys training: model = popularity</pre>



<pre>Recsys training: model = factorization_recommender</pre>


    4



<pre>Recsys training: model = popularity</pre>



<pre>Recsys training: model = factorization_recommender</pre>


    5



<pre>Recsys training: model = popularity</pre>



<pre>Recsys training: model = factorization_recommender</pre>


    6



<pre>Recsys training: model = popularity</pre>



<pre>Recsys training: model = factorization_recommender</pre>


    7



<pre>Recsys training: model = popularity</pre>



<pre>Recsys training: model = factorization_recommender</pre>


    8



<pre>Recsys training: model = popularity</pre>



<pre>Recsys training: model = factorization_recommender</pre>


    9



<pre>Recsys training: model = popularity</pre>



<pre>Recsys training: model = factorization_recommender</pre>


    10



<pre>Recsys training: model = popularity</pre>



<pre>Recsys training: model = factorization_recommender</pre>


    11



<pre>Recsys training: model = popularity</pre>



<pre>Recsys training: model = factorization_recommender</pre>


    12



<pre>Recsys training: model = popularity</pre>



<pre>Recsys training: model = factorization_recommender</pre>


    13



<pre>Recsys training: model = popularity</pre>



<pre>Recsys training: model = factorization_recommender</pre>


    14



<pre>Recsys training: model = popularity</pre>



<pre>Recsys training: model = factorization_recommender</pre>


    15



<pre>Recsys training: model = popularity</pre>



<pre>Recsys training: model = factorization_recommender</pre>


    16



<pre>Recsys training: model = popularity</pre>



<pre>Recsys training: model = factorization_recommender</pre>


    17



<pre>Recsys training: model = popularity</pre>



<pre>Recsys training: model = factorization_recommender</pre>


    18



<pre>Recsys training: model = popularity</pre>



<pre>Recsys training: model = factorization_recommender</pre>


    19



<pre>Recsys training: model = popularity</pre>



<pre>Recsys training: model = factorization_recommender</pre>


    20



<pre>Recsys training: model = popularity</pre>



<pre>Recsys training: model = factorization_recommender</pre>



```python
sns.boxplot(scoresDF)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7ff37d5c7a50>




![png](images/output_98_1.png)



```python
sns.boxplot(timesDF)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7ff3980594d0>




![png](images/output_99_1.png)



```python
sns.boxplot(timesDF.Base)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7ff386677610>




![png](images/output_100_1.png)



```python
sns.boxplot(timesDF.Tuned)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7ff38655b290>




![png](images/output_101_1.png)



```python
timesDF.Base.mean()
```




    0.6497844499999246




```python
timesDF.Tuned.mean()
```




    84.56693535000002




```python
pickle.dump(timesDF,open("timesDF.p","wb"))
pickle.dump(timesDF,open("scoresDF.p","wb"))
```

## Conclusion
While both models have comparable RMSE values, there is a huge difference between the times taken for the 2 models.
Therefore, the basic popularity recommender should be used instead of the tuned factorisation recommender.

## Reflection

To recap the process:
1. Obtain data from Kaggle
2. Process the data
    1. Remove duplicate rows
    2. Add genre information (Create the genre information table)
    3. Remove the 19 games which weren’t in the IGDB database
3. Create scoring function
4. Create benchmark popularity recommender model
5. Create suggested factorisation recommender model
6. Perform hyper-parameter tuning to get tuned factorisation recommender.
7. Evaluate the 3 models on the test dataset
8. Check the robustness of the models and suggest the final model

Defining the scoring function (and implementing it correctly) was the main issue due to the various joins and merges of the dataset. The 12 duplicate rows were one of the biggest roadblocks in this project. I only realised this issue when I was implementing the scoring function and the rows weren’t matching (as detailed in the appendix of the Jupyter notebook).

Checking the robustness of the 2 models via the pipeline was an interesting job. It showed me that while the 2 models had roughly the equal performances, the time taken to train the models made the basic popularity recommender a clear cut winnre

## Improvements
- Define the scoring function to be a weighted product of both the scores. This could be used by companies to decide what factor they want to focus more on. For example, people who have a high normalised score ie. they are heavy players could be targeted with micro-transaction type games.

- More data! This small dataset had limited interactions. Steam could create a better recommendation engine by using their own data.


## Appendix -  Debugging issue with merge and multiple player-game data¶

**Discovery** - While creating the scoring function, it was observed that the dataset created after an inner merge has more rows than the component datasets.
The personal-ratio dataset had 69276 interactions and the playtime-distribution dataset also had 69276 interactions. It was expected that the merged dataset would also have 69276 rows. But it was observed that the merged dataset had 69300 rows. This [SO question](https://stackoverflow.com/questions/41580249/pandas-merged-inner-join-data-frame-has-more-rows-than-the-original-ones) alerted me to the fact that there might be duplicates in the dataset. I observed that certain player-game combinations had 2 playtime entries instead of just 1 (as I had assumed). These duplicate rows were resulting in the issue observed above.

For example, the user ID 71411882 and the game 'Grand Theft Auto III' had 2 entries. This issue was present in the raw data itself.
