# MLND-Capstone

# Steam Game Recommender
## Rahul Phatak

### Project Overview
Steam is a gaming platform where multiple games on different platforms (including PC, Mac, and Linux) are sold to players. It also has a social component including streaming and community forums. Many video game providers also hold frequent discounts on the offered games. Users can also see recommendations based on the games they have played already.

Improving the recommendations shown to a user can increase user engagement and increase revenue of the game producer (and Steam too). Given the huge variety of games available across various platforms, targeting the correct user who is likely to buy the game, conduct micro-transactions, and/or recommend the game to his friends is a very lucrative problem for game producers. Since Steam receives a cut on games sold through Steam, Valve also has an interest in properly recommending games.

### Problem Statement
Create a recommendation system which can predict user engagement with games given playtime data about the games and the user. Top-5 games will be recommended to the user along with the predicted score. This score can be used by companies to decide revenues. Eg. If a user plays a game (eg Candy Crush) very much as compared to other players, similar long-term games with micro-transactions can be suggested to him. On the other hand, if a user plays a lot of different games but doesn’t spend a lot of time on any of them, another variety of games with one-time prices can be shown to him.

### Libraries Required
- pandas
- requests
- sys
- pickle
- seaborn
- graphlab
- time

### Datasets
- [Steam Video Games dataset](https://www.kaggle.com/tamber/steam-video-games)

This dataset provided by [Tamber](https://tamber.com/) contains user playtime data for different Steam games, ie the hours a particular user has played a particular game.

To quote the content description from the Kaggle page - “This dataset is a list of user behaviors, with columns: user-id, game-title, behavior-name, value. The behaviors included are 'purchase' and 'play'. The value indicates the degree to which the behavior was performed - in the case of 'purchase' the value is always 1, and in the case of 'play' the value represents the number of hours the user has played the game.”

- [Internet Game Database API](https://www.igdb.com/api)

Instead of using DBPedia as planned in the original capstone proposal, the Internet Game Database API was used since it contained detailed game information and had a good API as compared to DBPedia which would have heavy SPARQL queries.