import pickle
import numpy as np
import pickle

loaded_model = pickle.load(open("D:/ML PROJECTS/fake-news/fake_news.model.pkl",'rb'))

x_test =
x_new = x_test[3]

prediction = loaded_model.predict(x_new)

print(prediction)

if (prediction[0]==0):
    print('The news is Real')
else:
    print('The news is fake')