# LMPD Citation Prediction Model: 
## Tensorflow LinearClassifier
A TensorFlow LinearClassifier example using Louisville Police Traffic Citation Data to predict what demographics receives warnings vs citations.

Data is a selection from the Louisville City OpenData "Uniform Citation Data"
http://portal.louisvilleky.gov/dataset/uniformcitation-data

Utilizing a TensorFlow Linear Classifier, this model can predict wether a citation would be issued based on the following data points:

     OFFICER_GENDER,
     OFFICER_RACE, 
     OFFICER_AGE_RANGE, 
     ACTIVITY_DIVISION, 
     ACTIVITY_BEAT, 
     DRIVER_GENDER,
     DRIVER_RACE,
     DRIVER_AGE_RANGE,
     NUMBER_OF_PASSENGERS

This model can predict if a citation would be given at <b>80%-85% accuracy<b>
