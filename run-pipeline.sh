nohup python data-preparation.py 

sleep 10

nohup python resample-log-data.py 

sleep 10

nohup python merge-log-with-survey.py 

sleep 10

nohup python pivot-dataframe.py

sleep 10

nohup python feature-selection.py

sleep 10

nohup python remove-faulty-features.py

sleep 10

nohup python train-nomothetic-rf.py

sleep 10

nohup python train-nomothetic-xgboost.py

sleep 10

nohup python train-nomothetic-svr.py

sleep 10

nohup python train-idiographic-rf.py

sleep 10

nohup python train-idiographic-xgboost.py

sleep 10

nohup python train-idiographic-svr.py

sleep 10

nohup python naive-baseline-features.py

sleep 10

nohup python naive-baseline-merge.py

sleep 10

nohup python correlations-naive-baseline.py

sleep 10

nohup python correlations-idiographic-models.py

sleep 10

nohup python correlations-nomothetic-models.py

sleep 10

nohup python figure-smartphone-use-per-window-per-person.py

sleep 10

nohup python figure-prediction-error-naive-baseline.py

sleep 10

nohup python figure-smartphone-use-across-week.py

sleep 10

figure-nomothetic-error-prediction-plot.py

sleep 10

figure-idiographic-correlations-histogram.py

sleep 10

figure-idiographic-prediction-error.py

sleep 10

figure-nomothetic-correlations-histogram.py

sleep 10

number-of-strong-correlations.py