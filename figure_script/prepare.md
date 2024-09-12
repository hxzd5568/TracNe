# Instrucitons for getting figures in the paper

## Step1. Save the experiment results in CSV files
```
# Run evaluation 1
python test_fuzzer.py 1-60 --method MCMC
python test_fuzzer.py 1-60 --method DEMC
python test_fuzzer.py 1-60 --method MEGA
# Arrage the results in all_error.txt according to the format of synerrors.csv and replace the synerrors.csv with new data

# Run evluation 2
python test_tbatch.py model_dir --method DEMC
# Save the results into dnn.csv according to the format of dnn.csv
# For getting the between-class distance of Image models, Cancel comment at Line 60 in test_fuzztorch.py.
# For getting the between-class distance of NLP models, Cancel comment at Line 59 in test_fuzztorch.py.

# Run ablation experiment
Change the fuzzer.py's global configuration at the beginning
if run DE, line 27 should be "speed1, speed2, speed3, speed4 = 1, 24, 0, 16"
if run MDE, line 27 should be "speed1, speed2, speed3, speed4 = 5, 24, 0, 16"
if run GA, it is "speed1, speed2, speed3, speed4 = 1, 1, 1, 16"
if run DEGA, it is "speed1, speed2, speed3, speed4 = 1, 24, 2, 16"
if run MEGA, it is "speed1, speed2, speed3, speed4 = 5, 24, 2, 16"
Then, save the results into ablationfuzz.csv
```

## Step2. Run the plotting script.
```
python plot.py / # Or execute the plot.ipynb
```
Note that the ipynb file provides not only the scripts for drawing the picture in the paper but also the scripts of some related yet important picture.
