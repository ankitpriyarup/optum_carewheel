import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


ip= pd.read_csv("../data/observations.csv")
ip.head()
ip = ip.drop(['PID','death_status','futime'], axis=1)
byethnic= ip.groupby('ethnicity_fac')['Status'].mean().plot(kind= "bar", title= "Death Rate by Ethnic Group")

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

scaler= StandardScaler()
enc = LabelEncoder()

# Apply the encoding 
ip['ethnicity_cat'] = enc.fit_transform(ip['ethnicity_fac'])
ip.groupby(['ethnicity_cat','ethnicity_fac'])['age'].count()
ip['over_60'] = enc.fit_transform(ip['over_60'])
ethnic = pd.get_dummies(ip['ethnicity_cat'], drop_first= True, prefix="ethnic")

ip = pd.concat([ip, ethnic], axis= 1)
ip[['age']] = scaler.fit_transform(ip[['age']])
ip = ip.drop(['ethnicity_fac','ethnicity_cat'], axis= 1)

sns.heatmap(ip.corr())

corvec= ip.corr().Status
corvec.shape
print(corvec)
corvec.plot(kind="bar", figsize= [10,4])


# Training based on features selected
from sklearn.model_selection import train_test_split
X = ip.drop('Status', axis= 1)
y= ip.Status
X_train, X_test, y_train, y_test = train_test_split(X, y,   stratify=y)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Create a Random Forest Classifier with specified criterion
rf_class = RandomForestClassifier(criterion='entropy')

# Create the parameter grid
param_grid = {'max_depth': [2, 4, 8, 15], 'max_features': ['auto','sqrt'],
             'n_estimators' : [100, 500, 1000]} 

# Create a GridSearchCV object
grid_rf_class = GridSearchCV(
    estimator=rf_class,
    param_grid=param_grid,
    scoring= 'roc_auc',
    n_jobs=4,
    cv=5,
    refit=True, return_train_score=True)
print(grid_rf_class)

grid_rf_class.fit(X_train, y_train)
# Read the cv_results property into a dataframe & print it out
cv_results_df = pd.DataFrame(grid_rf_class.cv_results_)

# Extract and print the column with a dictionary of hyperparameters used
column = cv_results_df.loc[:, ['params']]

# Extract and print the row that had the best mean test score
best_row = cv_results_df[cv_results_df['rank_test_score'] == 1 ]
print(best_row)


from sklearn.metrics import classification_report, roc_auc_score

# Create an array of predictions directly using the best_estimator_ property
predictions = grid_rf_class.best_estimator_.predict(X_test)

# Take a look to confirm it worked, this should be an array of 1's and 0's
print(predictions[0:5])

# Now create a confusion matrix 
print("Confusion Matrix \n", classification_report(y_test, predictions))

# Get the ROC-AUC score
predictions_proba = grid_rf_class.best_estimator_.predict_proba(X_test)[:,1]
print("ROC-AUC Score \n", roc_auc_score(y_test, predictions_proba))


featureImp= []
for feat, importance in zip(X_train.columns, grid_rf_class.best_estimator_.feature_importances_):  
    temp = [feat, importance*100]
    featureImp.append(temp)

fT_df = pd.DataFrame(featureImp, columns = ['Feature', 'Importance'])
fT_df=  fT_df.sort_values('Importance', ascending = False)
print(fT_df)
