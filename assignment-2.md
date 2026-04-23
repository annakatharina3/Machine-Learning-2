# The task: 

## Cluster the Clustering Monster (7 points)
Modify the provided clustering code (@ClusterMonster_v3.ipynb) to work on your own dataset. Apply at least two clustering methods (e.g. k-means and one other), adjust their
parameters, visualize the results, and briefly compare which method works
best for your data and why.

We want to use data on character from the computer game League of Legends (LoL), by getting all the info on the champions available in the game and their characteristics. The goal is to use a clustering method to classify a given champion into one or multiple tags (class labels) of champions. We treat this as multi-label classification. Traditionally champions had at maximum two class labels (e.g. Fighter + Tank). 

Please solve the task seamlessly.

## How you could get the data:
### 1. Current patch version
GET https://ddragon.leagueoflegends.com/api/versions.json

### 2. Full champion list (names, tags, info)
GET https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/champion.json

### 3. Detailed stats per champion (loop over ~170 IDs)
GET https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/champion/{champ_id}.json


## Clustering methods that are available to use from lecture material (you may also use a different, but make sure to explain it more detailed in this case): 

- K-Means Clustering — covered in detail, including the algorithm steps (centroid initialization, assignment, update), preprocessing (standardization/min-max scaling), hyperparameters (max_iter, tol), and the SSE (Sum of Squared Errors) objective function.
- Mini Batch K-Means — mentioned as a comparison to K-Means (slide showing output differences).
- DBScan — density-based clustering (credited to K. Ramasamy).
- Mean Shift — mode-seeking clustering (credited to M. Nedrich).