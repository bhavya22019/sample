import pandas as  pd
import seaborn as sns
#import matplotlib.pyplot as plt
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')
data=pd.read_csv('cardio_train.csv')
data=data.drop_duplicates()

data=data.drop(['id','active'],axis=1)
data['BMI']=data['weight']/((data.height/100)**2)
cm=data.corr()
data=data[data['ap_lo']>0]
data=data[data['ap_hi']<250]
data=data[data['ap_hi']>60]
data=data[data['ap_lo']<150]
data=data[data['ap_lo']>50]
upper_limit= data.weight.mean() + 2*data.weight.std()
print(upper_limit)
lower_limit= data.weight.mean() - 2*data.weight.std()
lower_limit
data=data[data['weight']<upper_limit]
data=data[data['weight']>lower_limit]

upper_limit= data.height.mean() + 2*data.height.std()
print('upper limit: ',upper_limit)
lower_limit= data.height.mean() - 2*data.height.std()
print('Lower limit: ',lower_limit)

data=data[data['height']<upper_limit]
data=data[data['height']>lower_limit]
upper_limit= data.ap_hi.mean() + 3*data.ap_hi.std()
print('upper limit: ',upper_limit)
lower_limit= data.ap_hi.mean() - 3*data.ap_hi.std()
print('Lower limit: ',lower_limit)

data=data[data['ap_hi']<upper_limit]
data=data[data['ap_hi']>lower_limit]
upper_limit= data.ap_lo.mean() + 3*data.ap_lo.std()
print('upper limit: ',upper_limit)
lower_limit= data.ap_lo.mean() - 3*data.ap_lo.std()
print('Lower limit: ',lower_limit)

data=data[data['ap_lo']<150]
data=data[data['ap_lo']>50]
y=data.cardio
x=data[['age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco']]

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.linear_model import LogisticRegression
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=True)

lgr=LogisticRegression(solver='newton-cg').fit(x_train,y_train)
print('Accuracy of Logistic Regression:',accuracy_score(y_test,lgr.predict(x_test)))
print('Classification Report:\n',classification_report(y_test,lgr.predict(x_test)))

model=GradientBoostingClassifier().fit(x_train,y_train)
pred=model.predict(x_test)
print('Accuracy of GradientBoostingClassifier:',accuracy_score(y_test,pred))
print('Classification Report:\n',classification_report(y_test,pred))
rfc = RandomForestClassifier(random_state=True)
rfc.fit(x_train, y_train)
print('Accuracy of random forrest classifier:',accuracy_score(y_test,rfc.predict(x_test)))
print('Classification Report:\n',classification_report(y_test,rfc.predict(x_test)))
from sklearn.metrics import plot_confusion_matrix,plot_roc_curve,plot_precision_recall_curve
pickle.dump(model,open('Healtcare.pkl','wb'))
pic=pickle.load(open('Healtcare.pkl','rb'))
print(pic.predict([[52.0,1,165,64.0,130,70,3,1,0,0]]))