import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
# Correct import
from sklearn.svm import SVC


# load the data which i collected
data=pd.read_csv("train_data_driver.csv")

# seprate that data with their feature
X=data[['EAR','MAR','Pitch','Yaw']]
y=data['Label']

# now perform train test
X_train , X_test , y_train , y_test =train_test_split(X,y,test_size=0.2,random_state=42)

# intializing svm model
print("training SVM model")
svm_model = make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True, C=10))
svm_model.fit(X_train , y_train)

# 4. Check the Accuracy on the 20% test data
predictions = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)


print(f" MODEL ACCURACY: {accuracy * 100:.2f}%")
print("----------------------------------------\n")

# save that trained data
joblib.dump(svm_model , "driver_model.pkl")
print("model saved success fully!")
