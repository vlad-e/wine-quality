
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle

wines_red = pd.read_csv('winequality-red.csv',delimiter=';')
wines_white = pd.read_csv('winequality-white.csv',delimiter=';')

wines_red['type'] = 'red'
wines_white['type'] = 'white'

wines = pd.concat([wines_red,wines_white],axis=0)
wines.reset_index(drop=True, inplace=True)
wines['quality'].astype(CategoricalDtype(range(1,10),ordered = True))


train_full, val = train_test_split(wines, test_size=0.2, random_state=0, shuffle = True)

X = train_full.drop(['quality','type'], axis=1)
y = train_full['quality']
X_val = val.drop(['quality','type'], axis=1)
y_val = val['quality']



skf = StratifiedKFold(n_splits=3, shuffle = False)



rf = RandomForestClassifier()



params = {
    'max_features':[0.1,0.3,0.5,0.8,1],
    'n_estimators':[10,30,70,100],
    'max_depth':[3,6,9,12],
    'max_leaf_nodes':[2,4,6,8,10]
}
grd_rf = GridSearchCV(estimator=rf, param_grid = params, cv =skf,scoring = 'balanced_accuracy' )


print('Doing grid search on Random Forest Classifier')

grd_rf.fit(X,y)

print("\n Results from Grid Search: " )
print("\n The best score across ALL searched params:\n",grd_rf.best_score_)
print("\n The best parameters across ALL searched params:\n",grd_rf.best_params_)


best_rf = grd_rf.best_estimator_
y_pred = best_rf.predict(X_val)

print('\n The balanced accuracy score for the validation dataset is: ',balanced_accuracy_score(y_val, y_pred))

print('\n Saving model...')
with open('model_.bin','wb') as f_out:
    pickle.dump(best_rf,f_out)
print('done')


