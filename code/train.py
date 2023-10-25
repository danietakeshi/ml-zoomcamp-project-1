import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
import xgboost as xgb

def rmse(y, y_pred):
    se = (y - y_pred) ** 2
    mse = se.mean()
    return np.sqrt(mse)

eta = 0.1
max_depth = 3

print('Loading the data...')

df = pd.read_csv('../data/CrabAgePrediction.csv')

print('Formatting the data...')

df.columns = df.columns.str.lower().str.replace(' ', '_')

df = df[df.height != 0]

for column in ['length', 'diameter', 'height']:
    df[column] = df[column] * 0.3048

for column in ['weight', 'shucked_weight', 'viscera_weight', 'shell_weight']:
    df[column] = df[column] / 35.274

df['bmi'] = df.weight / df.height ** 2
df['density'] = df.weight / (df.height * df.length * df.diameter)

df['height_above_average'] = (df.height > df.height.mean()).astype(int)
df['weight_above_average'] = (df.weight > df.weight.mean()).astype(int)

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

columns = ['height','sex','weight']

y_full_train = df_full_train.age.values
y_test = df_test.age.values

dv = DictVectorizer(sparse = False)

X_full_train = dv.fit_transform(df_full_train[columns].to_dict(orient='records'))
X_test = dv.transform(df_test[columns].to_dict(orient='records'))

features = dv.get_feature_names_out().tolist()

dtrain = xgb.DMatrix(X_full_train, label=y_full_train, feature_names=features)
dval = xgb.DMatrix(X_test, label=y_test, feature_names=features)

xgb_params = {
    'eta': eta, 
    'max_depth': max_depth,
    'min_child_weight': 1,
    
    'objective': 'reg:squarederror',
    'nthread': 8,
    
    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dtrain, num_boost_round=100,
                  verbose_eval=5)

y_pred = model.predict(dval)
score = rmse(y_test, y_pred)

output_file = f'model_eta={eta}_max_depth={max_depth}_v{score.round(2)}.bin'

print(f'Saving the model on {output_file}')

with open(output_file, 'wb') as f_out: 
    pickle.dump((dv, model, features), f_out)

