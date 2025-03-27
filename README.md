# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.

5.Display the results.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by:Abisheik Raj.J 
RegisterNumber:212224230006  
*/
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x
y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
### PLACEMENT DATA
![Screenshot 2025-03-27 203110](https://github.com/user-attachments/assets/3ddf8e96-53c7-4e74-9bf3-7a7f0f695cee)

### CHECKING THE NULL()FUNCTION
![Screenshot 2025-03-27 203123](https://github.com/user-attachments/assets/0f83111f-8e17-4975-ad82-560b3e425fb4)

### PRINT DATA
![Screenshot 2025-03-27 203137](https://github.com/user-attachments/assets/cabfce49-461d-4c5f-887b-b59e40898a23)

### Y_PREDICTION ARRAY
![Screenshot 2025-03-27 203146](https://github.com/user-attachments/assets/af1e36f9-0bc6-4361-a0ad-1e0feea18868)

### ACCURACY VALUE
 ![Screenshot 2025-03-27 203154](https://github.com/user-attachments/assets/a430d545-756a-4fa0-8e1d-75b52f923f62)

### CONFUSION ARRAY
![Screenshot 2025-03-27 203200](https://github.com/user-attachments/assets/e6bf7de7-d511-4171-a7c4-065de1462d5b)

### CLASSIFICATION REPORT
![Screenshot 2025-03-27 203208](https://github.com/user-attachments/assets/260dff5d-f732-476c-83e9-e7b468696623)

### PREDICATION OF LR
![Screenshot 2025-03-27 203216](https://github.com/user-attachments/assets/b68bd9b2-2f3b-46b7-97a6-134d1d8556eb)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
