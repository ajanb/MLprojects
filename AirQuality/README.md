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


```
Training for target_carbon_monoxide...
Fold 0 RMSLE: 0.10258944372602422
Fold 1 RMSLE: 0.09280572216033992
Fold 2 RMSLE: 0.10085057484285562
Fold 3 RMSLE: 0.09589105909574525
Fold 4 RMSLE: 0.09210514288124344
Fold 5 RMSLE: 0.09621907880673664
Fold 6 RMSLE: 0.08723444850420152
Fold 7 RMSLE: 0.09892748647347609
Fold 8 RMSLE: 0.0987373721690222
Fold 9 RMSLE: 0.0941885627184215

Overall target_carbon_monoxide RMSLE: 0.09605308363480305

Training for target_benzene...
Fold 0 RMSLE: 0.07822385298537006
Fold 1 RMSLE: 0.0804583209907528
Fold 2 RMSLE: 0.08237858250659683
Fold 3 RMSLE: 0.08073212561916356
Fold 4 RMSLE: 0.08130173525238096
Fold 5 RMSLE: 0.0805312489213579
Fold 6 RMSLE: 0.07999118620593676
Fold 7 RMSLE: 0.08093512454914382
Fold 8 RMSLE: 0.0806252494347655
Fold 9 RMSLE: 0.08014225320644701

Overall target_benzene RMSLE: 0.08053817083534802

Training for target_nitrogen_oxides...
Fold 0 RMSLE: 0.1895218103641727
Fold 1 RMSLE: 0.1912791115364236
Fold 2 RMSLE: 0.20829691482399315
Fold 3 RMSLE: 0.21156644427777663
Fold 4 RMSLE: 0.20094590222434783
Fold 5 RMSLE: 0.19807655043959943
Fold 6 RMSLE: 0.19296132752075443
Fold 7 RMSLE: 0.1942070288287844
Fold 8 RMSLE: 0.19516145636469723
Fold 9 RMSLE: 0.18680730274809407

Overall target_nitrogen_oxides RMSLE: 0.19702871922988907


Total RMSLE is 0.12453999123334672
```
