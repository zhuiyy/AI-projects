# %% [markdown]
# #### Importing

# %%
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import re
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_percentage_error

from scipy.stats import randint, uniform

import lightgbm as lgb

# %%
train = pd.read_csv("./data/train.csv")
train_y = train[["SALE PRICE"]]
low = train_y['SALE PRICE'].quantile(0.01)
up = train_y['SALE PRICE'].quantile(0.99)
train = train[(train_y['SALE PRICE'] > low) & (train_y['SALE PRICE'] < up)]
train_X = train
train_y = train[["SALE PRICE"]]
del train_X["SALE PRICE"]

train_y_log = np.log1p(train_y)

test_X = pd.read_csv("./data/test.csv")
test_y = pd.read_csv("./data/test_groundtruth.csv")

print("train_X:",train_X.shape)
print("train_y:",train_y.shape)
print("test_X:",test_X.shape)
print("test_y:",test_y.shape)

# %% [markdown]
# #### Original Dataset

# %%
num_train_samples = len(train_X)

data_X = pd.concat([train_X, test_X])
data_X.head(4)

# %% [markdown]
# ### Encoding and Data Pre Processing

# %%
# 请注意 下面删除的特征很可能是有用的，合理的处理能够获得更为准确的预测模型，请探索所删除特征的使用
#del data_X['ADDRESS']
#del data_X['APARTMENT NUMBER']
#del data_X['BUILDING CLASS AT PRESENT']
#del data_X['BUILDING CLASS AT TIME OF SALE']
#del data_X['NEIGHBORHOOD']
del data_X['SALE DATE']
#del data_X['LAND SQUARE FEET']
#del data_X['GROSS SQUARE FEET']

# %%
need_encoding = data_X.columns
need_1hot = list(i for i in need_encoding if i in [
    'BOROUGH',
    'NEIGHBORHOOD',
    'BUILDING CLASS CATEGORY',
    'BUILDING CLASS AT PRESENT',
    'TAX CLASS AT PRESENT',
    'BUILDING CLASS AT TIME OF SALE',
])
need_freq = list(i for i in need_encoding if i in ['BLOCK', 'LOT', 'APARTMENT NUMBER'])
need_fill = list(i for i in need_encoding if i in ['LAND SQUARE FEET', 'GROSS SQUARE FEET', 'YEAR BUILT'])

# %% [markdown]
#     address

# %%
if 'ADDRESS' in need_encoding:
    street_suffix_map = {
        'STREET': ['ST', 'STREET'],
        'AVENUE': ['AVE', 'AVENUE', 'AV', 'AVN'],
        'ROAD': ['RD', 'ROAD'],
        'DRIVE': ['DR', 'DRIVE', 'DRV'],
        'LANE': ['LN', 'LANE'],
        'BOULEVARD': ['BLVD', 'BOUL', 'BOULV'],
        'COURT': ['CT', 'COURT'],
        'PLACE': ['PL', 'PLACE'],
        'SQUARE': ['SQ', 'SQUARE'],
        'TERRACE': ['TER', 'TERRACE'],
        'WAY': ['WAY'],
        'CIRCLE': ['CIR', 'CIRCLE', 'CRCL'],
        'HIGHWAY': ['HWY', 'HIGHWAY'],
        'PARKWAY': ['PKWY', 'PARKWAY', 'PKY'],
        'ALLEY': ['ALY', 'ALLEY'],
        'TRAIL': ['TRL', 'TRAIL', 'TR']
    }

    
    all_suffixes_flat = []
    for standard_form, variants in street_suffix_map.items():
        for variant in variants:
            all_suffixes_flat.append(variant)
    all_suffixes_flat.sort(key=len, reverse=True)

    regex_pattern = r"(?i)\b(" + "|".join(re.escape(s) for s in all_suffixes_flat) + r")\b"

    temp_address_series = data_X['ADDRESS'].astype(str)
    extracted_suffixes = temp_address_series.str.extract(regex_pattern, expand=False)
    data_X['EXTRACTED_SUFFIX_RAW'] = extracted_suffixes.str.upper()

    variant_to_standard_map = {}
    for standard, variants in street_suffix_map.items():
        for variant in variants:
            variant_to_standard_map[variant.upper()] = standard

    data_X['STREET_SUFFIX_STANDARD'] = data_X['EXTRACTED_SUFFIX_RAW'].map(variant_to_standard_map)
    data_X['STREET_SUFFIX_STANDARD'] = data_X['STREET_SUFFIX_STANDARD'].fillna('OTHER_SUFFIX')

    data_X['STREET_SUFFIX_STANDARD'] = data_X['STREET_SUFFIX_STANDARD'].astype('category')
    one_hot_suffixes = pd.get_dummies(data_X['STREET_SUFFIX_STANDARD'], prefix='Suffix', dtype=int)
    data_X = pd.concat([data_X, one_hot_suffixes], axis=1)

    if 'EXTRACTED_SUFFIX_RAW' in data_X.columns:
        del data_X['EXTRACTED_SUFFIX_RAW']
    
    del data_X['STREET_SUFFIX_STANDARD']
    del data_X['ADDRESS']

    

# %% [markdown]
#     timing

# %%
if 'SALE DATE' in need_encoding:
    data_X['SALE YEAR'] = pd.to_datetime(data_X['SALE DATE']).dt.year
    data_X['SALE MONTH'] = pd.to_datetime(data_X['SALE DATE']).dt.month
    data_X['SALE DATE'] = pd.to_datetime(data_X['SALE DATE']).dt.day

# %% [markdown]
#     1-hot

# %%
data_X[need_1hot] = data_X[need_1hot].astype('category')

one_hot_encoded = pd.get_dummies(data_X[need_1hot], dtype=int)

data_X = data_X.drop(need_1hot,axis=1)
data_X = pd.concat([data_X, one_hot_encoded] ,axis=1)

# %% [markdown]
#     filling

# %%
### 注意!编码只能用训练数据算mean!!!!!

data_X[need_fill] = data_X[need_fill].apply(lambda x: pd.to_numeric(x, errors='coerce'))
train[need_fill] = train[need_fill].apply(lambda x: pd.to_numeric(x, errors='coerce'))
data_X[need_fill] = data_X[need_fill].replace([0], np.nan)
mean = train[need_fill].mean(skipna=True)
data_X[need_fill] = data_X[need_fill].fillna(mean)

# %% [markdown]
#     freq encoding

# %%
### 注意!编码只能用训练数据算mean!!!!!

for i in need_freq:
    freq_map = train[i].value_counts(normalize=True)
    data_X[i] = data_X[i].map(freq_map)
    data_X[i] = data_X[i].fillna(0) 

# %% [markdown]
# #### Last Checking

# %%
data_X.head(10)

# %%
data_X.info()

# %%
data_X.columns

# %%
train_X = data_X[:num_train_samples].to_numpy()
test_X = data_X[num_train_samples:].to_numpy()
train_y_log = train_y_log.to_numpy().ravel()

# %%
scaler = StandardScaler()
train_X_scaled = scaler.fit_transform(train_X) 
test_X_scaled = scaler.transform(test_X) 

# %% [markdown]
# # Regression

# %%
M = RandomForestRegressor(
    n_estimators=200,
    criterion='squared_error',
    max_depth=25,
    min_samples_split=2,
    min_samples_leaf=3,
    min_weight_fraction_leaf=0.0,
    max_features=0.3,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    bootstrap=True,
    oob_score=True,
    n_jobs=-1,
    random_state=None,
    verbose=0,
    warm_start=False,
    ccp_alpha=0.0,
    max_samples=None,
    monotonic_cst=None
)

'''M = lgb.LGBMRegressor(
    boosting_type='rf',
    bagging_freq=1,
    bagging_fraction=0.9,
    num_leaves=1000,
    max_depth=-1,
    learning_rate=0.1,
    n_estimators=2000,
    subsample_for_bin=200000,
    objective='mape',
    class_weight=None,
    min_split_gain=0.0,
    min_child_weight=0.001,
    min_child_samples=20,
    subsample=1,
    subsample_freq=1,
    colsample_bytree=1.0,
    reg_alpha=0.0,
    reg_lambda=0.1,
    random_state=None,
    n_jobs=8,
    importance_type='split'
)'''

param = {
    'n_estimators': [100, 200],        
    'max_depth': [15, 25],            
    'min_samples_leaf': [3, 4, 5],         
    'max_features': ['sqrt', 0.3]      
}

grid_search = GridSearchCV(
    estimator=M,           
    param_grid=param,             
    cv=3,                              
    scoring='neg_mean_squared_error',  
    n_jobs=3,                         
    verbose=2                     
)

#print('GridSearchCV working on RF...')
#grid_search.fit(train_X_scaled, train_y_log)
#print('completed')
#print(f"best param: {grid_search.best_params_}")
#M = grid_search.best_estimator_

M.fit(train_X_scaled, train_y_log)
print('M done')

# %%
import gc

del grid_search

gc.collect()

# %%
N = MLPRegressor(
    hidden_layer_sizes=(150, 100, 50),
    activation='relu',               
    solver='adam',                   
    alpha=0.0082, 
    batch_size=64,
    learning_rate='adaptive',     
    learning_rate_init=0.001, 
    max_iter=1000,
    shuffle=True,   
    random_state=42,  
    early_stopping=True,
    validation_fraction=0.1, 
    n_iter_no_change=15,
    verbose=False
)

param = {
    'hidden_layer_sizes': [
        (64, 32), 
        (100, 50, 25), 
        (128, 64), 
        (150, 100, 50),
        (200,)
    ],
    'alpha': uniform(0.0001, 0.01),
    'learning_rate_init': [0.001, 0.0005, 0.0001],
    'batch_size': [32, 64, 128, 256]
}

random_search = RandomizedSearchCV(
    estimator=N,                   
    param_distributions=param,    
    n_iter=20,                                  
    cv=3,                                       
    scoring='neg_mean_squared_error',           
    n_jobs=3,                                  
    verbose=2,                                  
    random_state=74                            
)

#print('RandomizedSearchCV working on MLP...')
#random_search.fit(train_X_scaled, train_y_log)
#print('completed')
#print(f"best param: {random_search.best_params_}")
#N = random_search.best_estimator_
N.fit(train_X_scaled, train_y_log)
print('N done')

# %%
import gc

del random_search

gc.collect()

# %%
u = 0.963

train_Y_pre = u * np.expm1(M.predict(train_X_scaled)) - (1 - u) * np.expm1(N.predict(train_X_scaled))
mean_absolute_percentage_error(train_y,train_Y_pre)

# %%
Y_pred = u * np.expm1(M.predict(test_X_scaled)) - (1 - u) * np.expm1(N.predict(test_X_scaled))
print(mean_absolute_percentage_error(test_y,Y_pred))

# %%
pd.DataFrame({"pred":Y_pred}).to_csv("your.csv")


