#!/usr/bin/env python
# coding: utf-8

# Gender CLassification

# importing necessary libraries

# reading data
import pandas as pd

# data visualization
import plotly.express as px
import plotly.graph_objects as go

# data preprocessing
import math
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# models
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# metrics
from sklearn.metrics import accuracy_score, classification_report, make_scorer, confusion_matrix, log_loss

# saving model
import pickle

# notebook specific options
# print plots in vscode
import plotly.io as pio
pio.renderers.default = 'plotly_mimetype+notebook'


# Data Pre-processing
# reading the dataset
df = pd.read_csv('dataset/merged_audio_files.csv')
df.head()
df.info()

# checkinf for null values
df.isna().sum()

# function to convert label into an str

def classify(df):
    if df['label'] == 0:
        return 'male'
    elif df['label'] == 1:
        return 'female'

# creating a copy of the dataframe
df_copy = df.copy()
df_copy['class'] = df_copy.apply(classify, axis=1)

# histogram to check if there is any class imbalance
hist = px.histogram(df_copy, x = 'class', template = 'plotly_dark', color = 'class')

# Add annotations for count of each class
counts = df_copy['class'].value_counts()
annotations = [dict(x=c, y=0, text=str(counts[c]), showarrow=False) for c in counts.index]
hist.update_layout(annotations=annotations)

# storing the column names in a list
cols = df.columns.tolist()

# training features
features = [c for c in cols if c not in ['class', 'label']]

# label
target = 'label'

# scaling and splitting the data
df_shuffled = df.sample(frac=1, random_state=3).reset_index(drop=True)

# initiating a scaler instance
scaler = StandardScaler()

# fit and transform the columns to be scaled
df[features] = scaler.fit_transform(df[features])

# Save the scaler to a file
with open('Scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# choose what percentage of dataset should be used
percentage = 0.9

# finding the size of 90% from the dataset
n_rows = math.ceil(percentage * (df.shape[0]))

# use remaining rows for validation
rem_rows = df.shape[0] - n_rows

subset = df.copy()
train_subset = subset.head(n_rows)
val_subset = subset.tail(rem_rows)

X = train_subset[features].values
y = train_subset[target]

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

X_val = val_subset[features].values
y_val = val_subset[target]

# Regional voice data
# loading the dataset
mdf = pd.read_csv('dataset/malayalam_data.csv')

# shuffling the dataset
mdf_shuff = mdf.sample(frac=1).reset_index(drop=True)

# loading the scaler
with open('Scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# scaling the dataset
X_kl = scaler.fit_transform(mdf_shuff.iloc[:, :20])
y_kl = mdf_shuff['label']

# Models
# 1 - Logistic Regression

lr = LogisticRegression(penalty='l2', class_weight='balanced', random_state=42, solver='sag', max_iter=1000)
lr.fit(X_train, y_train)

# training accuracy
pred = lr.predict(X_train)
accuracy_score(y_train, pred)

# testing accuracy
lr.score(X_test, y_test)
print(classification_report(y_test, lr.predict(X_test)))

# validation accuracy
pred = lr.predict(X_val)
accuracy_score(y_val, pred)

# regional data accuracy
pred = lr.predict(X_kl)
accuracy_score(y_kl, pred)


# 2 - Decision Trees

dtree = DecisionTreeClassifier(criterion='gini', splitter='random', min_samples_leaf=2, max_features='log2', random_state=42)

# fitting the model
dtree.fit(X_train, y_train)

# training accuracy
pred = dtree.predict(X_train)
accuracy_score(y_train, pred)

# testing accuracy
pred = dtree.predict(X_test) 
accuracy_score(y_test, pred)

print(classification_report(y_test, pred))

# performing grid-search to prevent over-fitting
max_depth = []
acc_gini = []
acc_entropy = []
acc_logloss = []

for i in range(1,30):

    # gini
    dtree = DecisionTreeClassifier(criterion='gini', max_depth=i)
    dtree.fit(X_train, y_train)
    pred = dtree.predict(X_test)
    acc_gini.append(accuracy_score(y_test, pred))

    # entropy
    dtree = DecisionTreeClassifier(criterion='entropy', max_depth=i)
    dtree.fit(X_train, y_train)
    pred = dtree.predict(X_test)
    acc_entropy.append(accuracy_score(y_test, pred))

    # log loss
    dtree = DecisionTreeClassifier(criterion='log_loss', max_depth=i)
    dtree.fit(X_train, y_train)
    pred = dtree.predict(X_test)
    acc_logloss.append(accuracy_score(y_test, pred))

    max_depth.append(i)

fig = go.Figure()
fig.add_trace(go.Scatter(x=max_depth, y=acc_gini, name='gini'))
fig.add_trace(go.Scatter(x=max_depth, y=acc_entropy, name='entropy'))
fig.add_trace(go.Scatter(x=max_depth, y=acc_logloss, name='log loss'))
fig.update_layout(
    title = {
         'text': 'Gini vs Entropy',
         'x':0.5,
         'xanchor': 'center',
    },
    template='plotly_dark'
)
fig.show()


# making best depth tree
max_depth = dtree.get_depth()

max_depth_grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(),
    scoring=make_scorer(accuracy_score),
    param_grid=ParameterGrid(
    {"max_depth": [[max_depth] for max_depth in range(1, max_depth + 1)]}
    ),
)

max_depth_grid_search.fit(X_train, y_train)
max_depth_grid_search.best_params_

best_tree = max_depth_grid_search.best_estimator_
best_depth = best_tree.get_depth()

# validation accuracy after grid-search
pred = best_tree.predict(X_val)
accuracy_score(y_val, pred)

# regional data accuracy
pred = best_tree.predict(X_kl)
accuracy_score(y_kl, pred)


# 3 - Random Forest Classifier

rfc = RandomForestClassifier(n_estimators=15, criterion='gini', max_depth=6, min_samples_split=3, min_weight_fraction_leaf=0.01, max_features='log2')

# fitting the model
rfc.fit(X_train, y_train)

# # Save the trained model to a file using pickle
# with open('RandomForestClassifier.pkl', 'wb') as f:
#     pickle.dump(rfc, f)

# training accuracy
pred = rfc.predict(X_train.values)
accuracy_score(y_train, pred)

# testing accuracy
pred = rfc.predict(X_test.values)
accuracy_score(y_test, pred)

print(classification_report(y_test, pred))

# validation accuracy
prediction = rfc.predict(X_val)
accuracy_score(y_val, prediction)

print(confusion_matrix(y_val, prediction))

# regional data accuracy
pred = rfc.predict(X_kl)
accuracy_score(y_kl, pred)

print(confusion_matrix(y_kl, pred))

# training vs testing accuracy
train_accs = []
test_accs = []
for i in range(1, 21):
    rfc.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, rfc.predict(X_train))
    test_acc = accuracy_score(y_test, rfc.predict(X_test))
    train_accs.append(train_acc)
    test_accs.append(test_acc)


# In[124]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=list(range(1, 21)), y=train_accs, name='Training accuracy'))
fig.add_trace(go.Scatter(x=list(range(1, 21)), y=test_accs, name='Testing accuracy'))
fig.update_layout(
    template='plotly_dark'
)
fig.show()

# training vs testing loss
train_loss = []
test_loss = []
for i in range(1, 21):
    rfc.fit(X_train, y_train)
    train_loss.append(float(log_loss(y_train, rfc.predict(X_train))))
    test_loss.append(float(log_loss(y_test, rfc.predict(X_test))))

fig = go.Figure()
fig.add_trace(go.Scatter(x=list(range(1, 21)), y=train_loss, name='Training loss'))
fig.add_trace(go.Scatter(x=list(range(1, 21)), y=test_loss, name='Testing loss'))
fig.update_layout(
    template='plotly_dark'
)
fig.show()
