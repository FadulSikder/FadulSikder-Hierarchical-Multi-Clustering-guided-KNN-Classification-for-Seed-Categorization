# FadulSikder-Hierarchical-Multi-Clustering-guided-KNN-Classification-for-Seed-Categorization
**Final Project Report**

**Name: Fadul Sikder**

**UTA ID – 1001965359**

**Project Name:**

Hierarchical Multi Clustering guided KNN Classification for Seed Categorization

**Description:**

This seed categorization project utilizes Hierarchical Clustering and KNN to achieve high accuracy in

predicting the correct seed. First, I applied three hierarchical clustering using different linkage criteria on

the dataset. Then, with that clustering, I built a new feature vector upon which I applied KNN classification

to categorize the seeds.

**Procedures:**

\1. First, I have used Hierarchical Clustering with three different linkage criteria to generate a new

feature vector that adds the cluster labels with the original feature vector. The three linkage criteria

are average (weighted average), single, and complete. Then, I created a new column in the feature

vector for each type of clustering and added the cluster id of each data point in that new column.

Below, we can see the three new columns added to the original dataset.





\2. After this, I calculated two parameters to determine the similarity between clusters and a data point.

The first parameter obtains the similarity measurement between clusters to which the data point

belongs. To measure this, we calculate how many item intersections the clusters have among them

in proportion to their combine item numbers. Given a data point the following equation gives us

the parameters. For a data point D(i) we get the similarity coefficient,

퐶푎푣푒푟푎푔푒 ∩ 퐶푠푖푛푔푙푒 ∩ 퐶퐶표푚푝푙푒푡푒

SCi=

퐶푎푣푒푟푎푔푒 ∪ 퐶푠푖푛푔푙푒 ∪ 퐶퐶표푚푝푙푒푡푒

\3. Additionally, another similarity measurement is calculated by the cluster member voting

mechanism. Each data point gets a vote from each of the clusters they belong. The vote is calculated

by summing up the inverse of the distance between that data point and all other members of a

particular cluster. For a data point D(i) we get the Vote,





1

∑

V(i,j) =

푒푎푐ℎ 푐푙푢ꢁꢃ푒푟 푚푒푚푏푒푟ꢁ 푑ꢀꢁꢃ푎ꢂ푐푒 푓푟표푚 ꢃℎ푎ꢃ 푝표ꢀꢂꢃ

\4. This two-similarity measurement is encoded in the data point in place of their respective cluster id.

I experimented with a couple of formulas to encode those similarities information.

For datapoint di,

1

Di[cluster\_algo] =

Vote from cluster\_algo member+cluster similarity coefficient

Di[cluster\_algo] = Vote from cluster\_algomember × cluster similarity coefficient

1

Di[cluster\_algo] =

× cluster similarity coefficient

Vote from cluster\_algo member

\5. With this we get new feature vector.

\6. Subsequently, this new feature vector has been used to perform KNN classification.





\7. I used several cluster numbers and three KNN value to generate the prediction accuracy for seed

species.

**Result:**

1

Di[cluster\_algo] =

Vote from cluster\_algo member+ cluster similarity coefficient

Using this equation to consider similarity information we get the following result:

**Cluster**

**K = 3**

**K = 5**

**K = 7**

**Number**

**0**

**88.99**

**88.99**

**98.56**

**99.04**

**98.56**

**98.56**

**99.04**

**98.56**

**100.0**

**99.04**

**99.52**

**88.03**

**88.03**

**98.56**

**99.04**

**97.12**

**98.56**

**98.56**

**100.0**

**100.0**

**98.56**

**, 100.0**

**88.03**

**4**

**88.03**

**98.56**

**99.04**

**98.08**

**99.04**

**98.56**

**98.56**

**100.0**

**98.08**

**100.0**

**6**

**8**

**12**

**16**

**20**

**24**

**27**

**30**

**60**





Di[cluster\_algo] = Vote from cluster\_algomember × cluster similarity coefficient

Using this equation to consider similarity information we get the following result:

**Cluster**

**K = 3**

**K = 5**

**K = 7**

**Number**

**0**

**88.99**

**87.55**

**94.73**

**99.04**

**98.08**

**98.08**

**99.04**

**99.04**

**98.65**

**99.04**

**99.52**

**88.03**

**90.43**

**95.21**

**98.56**

**99.52**

**99.52**

**99.52**

**99.04**

**99.04**

**98.56**

**98.56**

**88.03**

**4**

**89.95**

**95.21**

**98.08**

**98.56**

**98.56**

**99.04**

**99.04**

**99.04**

**99.04**

**99.04**

**6**

**8**

**12**

**16**

**20**

**24**

**27**

**30**

**60**

In both table, first row has the KNN classification accuracy for the original data without clustering. The

other rows contain prediction accuracy with different numbers of clusters. Here we can see that we are

getting significantly higher accuracy with the Hierarchical Multi Clustering guided KNN Classification

than only KNN classification without clustering.





**Conclusion:**

From the result we can conclude that guiding KNN Classification with Hierarchical multi clustering we can

improve the prediction accuracy significantly.
