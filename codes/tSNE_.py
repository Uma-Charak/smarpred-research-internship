import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns

#read csv file
df=pd.read_csv('Randomized_bacterial_dataset.csv', sep=',',header=None)

#converts datapoints in csv to float16 
data = np.array(df.values[1:,1:-1], dtype='float16')

#extracts labels from csv
labels = np.array(df.values[1:,-1], dtype='str')

#sets up and runs tSNE
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300, random_state=0)
tsne_results = tsne.fit_transform(data)

d = { 'labels' : labels}

#creates a new dataframe to accomodate tsne results
tsne_data=pd.DataFrame(d,columns=['labels'])

#adds first tSNE dimention to the dataframe 
tsne_data['tsne_dim1']=tsne_results[:,0]

#adds second tSNE dimention to the dataframe
tsne_data['tsne_dim2'] = tsne_results[:,1]

#sets up plotting area and plots it
plt.figure(figsize=(10,10))
sns.scatterplot(
    x="tsne_dim1", y="tsne_dim2",
    hue="labels",
    palette=sns.color_palette("hls", 2),
    data=tsne_data,
    legend="full"
)
plt.xlim(-15,20)
plt.ylim(-15,20)
plt.show()
