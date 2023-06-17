import pandas as pd
df = pd.read_csv("/Users/prasonagnihottri/Desktop/train.csv")
df.head()df.head()
target = df.Survived
inputs = df.drop('Survived',axis='columns')
dummies = pd.get_dummies(inputs.Sex)
    #as machine learning models cannot handle text we are changing sex to numbers
dummies.head(3)
#now we have to add dummies database to our input database using pandas concat function
inputs = pd.concat([inputs,dummies],axis='columns')
inputs.head(3)
#now we don't want the sex column so we are dropping. it 
inputs.drop('Sex',axis='columns',inplace=True)
inputs.head(3)
#now we are checking if our datasets has any nan value and what are those values
inputs.columns[inputs.isna().any()]
#now we rae handling these nan values
inputs.Age = inputs.Age.fillna(inputs.Age.mean())
inputs.head(6)
#this is what we usually do while tranning a mode so that its not biassed
#trainig of model 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(inputs,target,test_size=0.2)
#naive model 
#here we are using gaussiannb as our data distribution is normal or gaussian distribution
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
 #now we are training the model
model.fit(X_train, y_train)
model.score(X_test,y_test)
