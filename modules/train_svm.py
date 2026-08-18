import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import joblib
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
svm_model=SVC(kernel='linear', probability=True)
svm_model.fit(X_train , y_train)

# save that trained data
joblib.dump(svm_model , "driver_model.pkl")
print("model saved success fully!")
