# LSTM-based-Deep-Neural-Network-for-NR-LF-IQA
In this work, we present a NR LF-IQA metric that is a Long Short-Term Memory based Deep Neural Network, named LSTMN-LFIQA. In this metric, we use CNN to extract primary features from LF images. Then we pass these features to LSTM network that learns long-term dependencies among quality-aware features. Furthermore, to overcome the problem of small size of input dataset, we demonstrate the efficiency of bottleneck features extracted from a pre-trained neural network for training process.

## Generation of EPI:
We generated EPIs using the method MultiEPL https://bit.ly/3Da8fB6

## Generation of MLI:
The code to generate MLI is given in direcorty Prep:
1. Setup the paths to input in run.m file. 
2. Run run.m to generate MLIs. 

## Generation of Bottleneck Features of MLI:
The code to generate vgg16 bottlenceck feaures of MLI is given in direcorty Prep:
1. Setup the paths to input in extract_bottleneck_features.py file.
2. Run extract_bottleneck_features.py 

## Train / Test the model:
1. Prepare the EPIs and Bottleneck features of MLIs.
2. To train the model, run train.py file after setting paths to inputs EPI and Bottleneck features.
3. Run test.py to test the trained model.
