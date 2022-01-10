# BLG 561E - Deep Learning Project
## Delving into Deep Imbalanced Regression

### Introduction

Data imbalance is very common and ubiquitous in the real world. Certain target values have significantly fewer observations. Such imbalanced data is actually very common and natural in the real world regardless of data sets with different attributes. There are many critical applications that are related to people's safety or health such as autonomous driving and medical diagnosis.

![image](https://user-images.githubusercontent.com/70148273/148697875-7ec577c4-767c-4588-8d6a-8400accbdf8c.png)

For instance , it is shown a clarty dataset for place classification located at the upper left. If we observe the frequency of different categories from high to low we will find that the data distribution has a long tail based on the stored class index. Moreover , the majority class such as living room has more than a thousand samples. On the other side , for rare classes such as museum it is only about 10 samples.

![image](https://user-images.githubusercontent.com/70148273/148698090-9acdea28-665d-4dbe-8f92-fcfa3c3b5f04.png)

### Feature Distribution Smoothing (FDS)

Second solution of imbalance data for regression is Feature Distribution Smoothing (FDS)

Firstly, in the graph shown below let's focus on impact of data imbalance on feature statistics. In the paper, there is real-word age dataset to infer a person's age from visual appereances and they focus on the learned feature space and visualize the feature statistics for age "0".

![Screenshot 2022-01-10 075734](https://user-images.githubusercontent.com/79253076/148720244-c869129f-611e-445d-a0be-592df77e053a.png)

In particular, when we talk about visualizing the similarity between feature statistics, we naturally have to look at the mean and the variance of the feature statistics as shown below in the upper figure shows that the cosine similarity of the feature mean and the lower figure shows that the cosine similarity of the feature variance.

![Screenshot 2022-01-10 080325](https://user-images.githubusercontent.com/79253076/148720477-fb7f2177-e28e-40dc-aa03-9d43ffbbfd39.png)

The red bar is the anchor target which is age "0". The blue bar for a specific target refers to the similarity score between that target and the anchor. Also in graph, the regions are marked with different data densities using different colors:

- The purple refers to many-shot region
- The yellow refers to medium-shot region
- The pink refers to the few-shot region

![Screenshot 2022-01-10 081351](https://user-images.githubusercontent.com/79253076/148721067-7f33609c-50d2-46d6-8ec6-1a7dd251c791.png)

However, the graph as shown above shows that there is a problem with regions that have very few data samples like age 0 which is our selected one would expect that the feature of age 0 would only have high similarity with its nearby ages such as age "1" and age "2" and this similarity score should decrease gradually as the age increases. However, the mean and the variance for age 0 have very high similarity with age "40" to age "60" which is NOT expected.

The reason of this unjustified similarity is due to data imbalance. Specifically, there are not enough images for age 0, so this range says inherits its prior from the range with the maximum amount of data and this corresponds to the manage of the region which is the relevent around age 40.

Therefore, FDS aims to smooth the feature space using the same idea as LDS. Basically, they want to transfer the feature statistics between the nearby target. Specifically, they have a model that maps the input data to the continuous predictions.

![Screenshot 2022-01-10 082322](https://user-images.githubusercontent.com/79253076/148721619-aa65da5c-a615-47c7-97a6-34fa0a9097da.png)

Moreover, FDS is performed by first estimating the statistics of each target being without loss of generality they substitute variance with covariance to reflect also the relationship between the feature elements within z which is feature vector. And there is again a submision kernel k to smooth the distribution of the feature mean and covariance over the target bins this results in a smooth version of the statistics.

After that, both estimated and smooth statistics calibrate the visual representation for each input sample and the whole pipeline of FDS can be added into deep networks as a calibration layer.

![Screenshot 2022-01-10 083001](https://user-images.githubusercontent.com/79253076/148722011-1ec1f3e0-c65b-4eca-91c2-3da730729084.png)

Finally, to obtain more stable and accurate estimation of the feature statistics during training there is a momentum update of the running statistics across each epoch.

