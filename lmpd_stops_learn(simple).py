# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 15:16:32 2017

@author: Admin
"""
##### IMPORTS ####
import pandas as pd
import tensorflow as tf
import re
import tempfile
##################

### READ IN DATA SOURCE
### READ DIFFRENT SECTIONS FOR TRAIN, TEST, PREDICT
df_train = pd.read_csv('LMPD_STOPS_DATA_CLEAN_V1_HEADERS.csv', nrows=90000, skipinitialspace=True)
# SAve Headers for the next selections that skip header row.
headers = list(df_train)

df_test = pd.read_csv('LMPD_STOPS_DATA_CLEAN_V1_HEADERS.csv',names=headers,skiprows=93000, nrows=1000, skipinitialspace=True)
df_predict = pd.read_csv('LMPD_STOPS_DATA_CLEAN_V1_HEADERS.csv',names=headers,skiprows=93001, nrows=100, skipinitialspace=True)

df_train =df_train.dropna(axis=0, how='any')
df_test =df_test.dropna(axis=0, how='any')
df_predict =df_predict.dropna(axis=0, how='any')

################ SETUP TENSORFLOW INPUTS ##########################
# Establish Labels to generate with model.
df_train['train_labels'] = df_train["ACTIVITY_RESULTS"] == 'CITATION ISSUED'
df_test['test_labels'] = df_test["ACTIVITY_RESULTS"] == 'CITATION ISSUED'
  
train_input = tf.estimator.inputs.pandas_input_fn(
       x=df_train,
       y=df_train["train_labels"],
       batch_size=100,
       num_epochs=None,
       shuffle=True,
       num_threads=5)
  
test_input = tf.estimator.inputs.pandas_input_fn(
       x=df_test,
       y=df_test["test_labels"],
       batch_size=100,
       num_epochs=None,
       shuffle=True,
       num_threads=5)

#predict is done only once, without shuffle, since we are writing results nex
#next to data inputs.
predict_input = tf.estimator.inputs.pandas_input_fn(
       x=df_predict,
       batch_size=100,
       num_epochs=1,
       shuffle=False,
       num_threads=1)
 
############ DEFINE FEATURE COULMNS ##############
## categorical features
#first get list of distinct itmes in train dataset.
GENDER_LIST = df_train.OFFICER_GENDER.unique().tolist()
#define that list as the categories to group by
OFFICER_GENDER = tf.feature_column.categorical_column_with_vocabulary_list(
     "OFFICER_GENDER", GENDER_LIST)

OF_RACE_LIST = df_train.OFFICER_RACE.unique().tolist()
OFFICER_RACE = tf.feature_column.categorical_column_with_vocabulary_list(
     "OFFICER_RACE", OF_RACE_LIST)
 
OF_AGE_LIST = df_train.OFFICER_AGE_RANGE.unique().tolist()
OFFICER_AGE_RANGE = tf.feature_column.categorical_column_with_vocabulary_list(
     "OFFICER_AGE_RANGE", OF_AGE_LIST)

ACTIVITY_DIVISION_LIST = df_train.ACTIVITY_DIVISION.unique().tolist()
ACTIVITY_DIVISION = tf.feature_column.categorical_column_with_vocabulary_list(
     "ACTIVITY_DIVISION", ACTIVITY_DIVISION_LIST)

ACTIVITY_BEAT_LIST = df_train.ACTIVITY_BEAT.unique().tolist()
ACTIVITY_BEAT = tf.feature_column.categorical_column_with_vocabulary_list(
"ACTIVITY_BEAT", ACTIVITY_BEAT_LIST)

DRIVER_GENDER_LIST = df_train.DRIVER_GENDER.unique().tolist()
DRIVER_GENDER = tf.feature_column.categorical_column_with_vocabulary_list(
     "DRIVER_GENDER", DRIVER_GENDER_LIST)

DRIVER_RACE_LIST = df_train.DRIVER_RACE.unique().tolist()
DRIVER_RACE = tf.feature_column.categorical_column_with_vocabulary_list(
     "DRIVER_RACE", DRIVER_RACE_LIST)

DRIVER_AGE_RANGE_LIST = df_train.DRIVER_AGE_RANGE.unique().tolist()
DRIVER_AGE_RANGE = tf.feature_column.categorical_column_with_vocabulary_list(
     "DRIVER_AGE_RANGE", DRIVER_AGE_RANGE_LIST)
  
## Continuous Features
#Use range of int/floats (that make sense)
NUMBER_OF_PASSENGERS = tf.feature_column.numeric_column('NUMBER_OF_PASSENGERS')

#### Adding features to Model
base_columns = [OFFICER_GENDER, 
     OFFICER_RACE, 
     OFFICER_AGE_RANGE, 
     ACTIVITY_DIVISION, 
     ACTIVITY_BEAT, 
     DRIVER_GENDER,
     DRIVER_RACE,
     DRIVER_AGE_RANGE,
     NUMBER_OF_PASSENGERS
 ]
# Related columns can be crossed in meaningful ways.
crossed_columns = [
     tf.feature_column.crossed_column(
         ["OFFICER_GENDER", "OFFICER_RACE", "OFFICER_AGE_RANGE"], hash_bucket_size=1000),
     tf.feature_column.crossed_column(
         ["DRIVER_GENDER", "DRIVER_RACE","DRIVER_AGE_RANGE"], hash_bucket_size=1000),
 ]

# Define where tensorflow model is saved.  This hardcoded version will break.
# files need to be deleted after run.  Use tempfolder to deal with this.
model_dir = tempfile.mkdtemp()
m = tf.estimator.LinearClassifier(
     model_dir=model_dir, feature_columns=base_columns + crossed_columns)
    
############ Training and Evaluating Our Model #####################
 #train
m.train(input_fn=train_input, steps=1000)
  
 #eval
results = m.evaluate(input_fn=test_input, steps=100)

#Show eval results
print("model directory = %s" % model_dir)
for key in sorted(results):
    tf.summary.scalar(key, results[key])
    print("%s: %s" % (key, results[key]))
print ('######################################')

#predict
predictions = m.predict(input_fn=predict_input)

#create list to save predictions to.
predict_list = []
prob_list = []

# For each input row, add binary prediction to new column.
for i, p in enumerate(predictions):
  #print("Prediction %s: %s" % (i + 1, p['classes']))
  #predict_list.append(p['classes'])
  for x in p['classes']:
     predict_list.append(re.sub("\D", "", str(x)))
  #print("Probabilities %s: %s" % (i + 1, p['probabilities']))
  prob_list.append(str(p['probabilities']))

pred_series = pd.Series(predict_list)
df_predict['Prediction'] = pred_series.values

prob_series = pd.Series(prob_list)
df_predict['Probabilities'] = prob_series.values

# Output prediction results to file.
df_predict.to_csv('prediction_results.csv', sep=',', encoding='utf-8')



