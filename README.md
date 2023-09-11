# CS301-Project-G5
Group Members- Vincent Cinardo, Aron Shrestha, Dhyeykumar Kansagara
section 104

Colab Notebook: [https://colab.research.google.com/drive/1ff83iK-ik1Y1XPqS5xpmHmcBVpZgSA-I?authuser=1#scrollTo=iSNgeg7odDZM](https://colab.research.google.com/drive/1ff83iK-ik1Y1XPqS5xpmHmcBVpZgSA-I?usp=sharing)

## Kaggle Project URL
https://www.kaggle.com/c/ubiquant-market-prediction

## Proposal

The name of our company will be Market Genie. The project we have chosen from Kaggle is titled “Ubiquant Market Prediction”. We will create a model which will give us an estimate of how much a given investment’s return rate will be. It is interesting because we recognize the utility in being able to forecast investment returns. We are only human, our predictions cannot compare to what a computer’s would be. Our final product could yield great profits. For our initial understanding, we will use an up-to-date blog post by Katherine Li from neptune.ai. This is a reputable website for storage of metadata in machine learning operations. We will also read scholarly articles that compare different machine learning models’ ability to predict the trend by feeding 10 years of historical data. This will help us to observe how different data models perform and help us propose our own data model. For this project, the data that we will be using will be real historical data from thousands of investments that are available for the competition. The data we will use will be split into a training set and a testing set. 10% of the data will be used as the test set and 90% of the data would be used to train our model. The learning technique we will be using to train our dataset is linear regression. We will train our model by minimizing our loss function using methods such as mean squared error.  There are some existing learning methods that use neural networks and LSTMs to train datasets to predict the market. Since we have not covered such topics in class, we will start training our dataset with linear regression and pivot accordingly when we learn more about these new learning models. Qualitatively, we expect to have non-linear plots. The goal will be to get the best line of fit for future data. We expect to get a line of fit with a similar pattern but not perfectly accurate; slightly lower or higher than the actual data. This is a common trend we observed among implementations. Quantitatively, we will calculate the accuracy and the runtime of our model by using our test dataset. We will then compare those results with the other teams in the competition to evaluate how good our results are. 
