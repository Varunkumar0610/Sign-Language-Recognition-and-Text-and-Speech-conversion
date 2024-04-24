import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
#use of data.pickle file created by create_dataset.py
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = KNeighborsClassifier()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
score = accuracy_score(y_test, y_predict)
#use of different classifier model and print the accuracy of the trained model
print('Accuracy of the K-Nearest Neighbors (KNN) Classifier model: {}'.format(score*100))
#dump the trained model as model.p file or model.h5 file
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
