Data: Hourly averaged data of responses from 5 metal oxides chemical sensors of Air Quality Chemical Multisensor Device. The device was placed in a significantly polluted area of Italy.


Sensors:
- S1 tin oxide (nominally CO targeted)
- S2 titania (nominally NMHC targeted (benzene))
- S3 tungsten oxide (nominally NOx targeted)
- S4 tungsten oxide (nominally NO2 targeted) 
- S5 indium oxide (nominally O3 targeted)


Predict the target values using the following parameters:
- 5 sensors collecting chemicals data in the air
- Weather condition at the time of collecting the data (humidity, temperature)
  Extracted the following features from date and time column:
  - Time of day 
  - Weekday
  - Season
  - Month


Target values:
- Carbon monoxide 
- Benzene 
- Nitrogen oxides 


Auto-correlation
- Before applying machine learning models to time series data, you have to transform it to an “ingestible” format for your models, and this often involves calculating lagged variables, which can measure auto-correlation i.e. how past values of a variable influence its future values, thus unlocking predictive value.
- In the model I added the impact of the previous hour 
- I used shift method. Shifting column data by 1 row down


Cross-Validation: Stratified K-Folds:
- Splits = 10


Algorithm: 
- CatBoost 
  advantages over other Decision Trees based algorithms:
  - Computational efficiency 

Metrics: RMSLE (root mean squared logarithmic error)
- Not sensitive to outliers




