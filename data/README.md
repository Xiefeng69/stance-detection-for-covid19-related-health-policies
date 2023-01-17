# Data Summary
In this work, we adopt three health policies: (1) Stay at Home Order, (2) Wear Masks, and (3) Vaccination, and only select labeled tweets posted in the USA. Meanwhile, we also collect unlabeled tweets for these three policies via [Twitter API](https://developer.twitter.com/). In addition, each labeled text is associated with a *GeoId* which indicates user's location. We use the "location" field provided by Twitter API to collect the shared place attachment of tweets. There are 52 *GeoIds* in total, including 50 states, 1 capital city (i.e., Washington, D.C.), and a general identifier: "USA''. In geographical adjacency matrix, the node with "USA'' label is an individual node without connecting with any others. We collect policy descriptions as external knowledge from [World Health Organization (WHO)](https://www.who.int/). The keywords of the crawler are exhibited in the `data/keywords.json` file. The labeled tweets are mainly from [stance-detection-in-covid-19-tweets](https://github.com/kglandt/stance-detection-in-covid-19-tweets), but some of the original tweets are unavailable due to deletion or privacy issues.

## Data Statistics
| Topic   |  #Unlabeled |   #Labeled (Favor/Against/None) |
| :------------- | :----------: | :------------: |
| Stay at Home (SH) |   778   | 420 (194/113/113) |
| Wear Masks (WM)       |    1030     |  756 (173/288/295)  |
| Vaccination (VA)       |    1535     |  526 (106/194/226) |

## Collected Descriptions
| Topic | Policy Description |
| :------------- | :---------- |
| Stay at Home | Under a stay-at-home order, all non-essential workers must stay home. People can leave their homes only for essential needs like grocery stores and medicine, or for solo outdoor exercise.|
| Wear Masks | Masks are a key measure to reduce transmission and save lives. Wearing well-fitted masks should be used as part of a comprehensive Do it all! approach including maintaining physical distancing, avoiding crowded, closed and close-contact setting, ensuring good ventilation of indoor spaces, cleaning hands regularly, and covering sneezes and coughs with a tissue of bent elbow.|
| Vaccination | Getting vaccinated could save your life. COVID-19 vaccines provide strong protection against serious illness, hospitalization and death. There is also some evidence that being vaccinated will make it less likely that you will pass the virus on to others, which means your decision to get the vaccine also protects those around you.|

## File Structure
+ `data/covid19-policies-stance`
    + `xxx_train.csv`: the training samples of a topic xxx.
    + `xxx_train_unseen`: the unseen/unlabled training samples of a topic xxx.
    + `xxx_val.csv`: the validation samples of a topic xxx.
    + `wiki_dict.pkl`: the pkl file which contains the wikipedia information for [WS-BERT](https://github.com/zihaohe123/wiki-enhanced-stance-detection).
    + `wiki.py`: add wiki information to the wiki_dict.pkl.
+ `data/data_unseen`: collected unlabeled tweets.
+ `data/us-adj.txt`: the geographical adjacent matrix of USA.
+ `data/us-state-label.xlsx`: the mapping file between [GeoIDs](https://www.census.gov/programs-surveys/geography/guidance/geo-identifiers.html) and states' name.
+ `data/keywords.json`: the keywords of the crawler.
+ `data/description.txt`: the description information of each topic.
+ `data/location_map_id.py`: a python file which maps the location field to geoid.
+ `data/change_topic.py`: a python file which aims to change the topic words of samples automatically.
+ `data/COVID-19_State_and_County_Policy_Orders.csv`: the covid-19-related policies' timeline
