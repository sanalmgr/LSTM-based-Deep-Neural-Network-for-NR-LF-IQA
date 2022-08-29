# LSTM-based-Deep-Neural-Network-for-NR-LF-IQA
In this work, we present a No-Reference (NR) LFIIQA method that is based on a Long Short-Term Memory based Deep Neural Network (LSTM-DNN). The method is composed of two streams. The first stream extracts long-term dependent distortion related features from horizontal epipolar plane images, while the second stream processes bottleneck features of micro-lens images. The outputs of both streams are fused, and supplied to a regression operation that generates a scalar value as a predicted quality score.

## Paper:
[No-Reference Light Field Image Quality Assessment Method Based on a Long-Short Term Memory Neural Network](https://www.computer.org/csdl/proceedings-article/icmew/2022/09859419/1G4EXa6qYx2).

## Generation of EPI:
We generated EPIs using the [MultiEPL](https://bit.ly/3Da8fB6) method.

## Generation of MLI:
I am grateful to [Chawin-S](https://github.com/Chawin-S) to achieve this task.
The code to generate MLI is given in direcorty Prep:
1. Setup the paths to input in run.m file. 
2. Run run.m to generate MLIs. 

## Generation of Bottleneck Features of MLI:
The code to generate vgg16 bottlenceck feaures of MLI is given in direcorty Prep:
1. Setup the paths to input in extract_bottleneck_features.py file.
2. Run extract_bottleneck_features.py 

## Training Model:
1. Prepare the EPIs and Bottleneck features of MLIs.
2. To train the model, import functions from training_model.py file, and pass the parameters accordingly.

## Cite this article:
```
@INPROCEEDINGS {9859419,
author = {S. Alamgeer and M. Q. Farias},
booktitle = {2022 IEEE International Conference on Multimedia and Expo Workshops (ICMEW)},
title = {No-Reference Light Field Image Quality Assessment Method Based on a Long-Short Term Memory Neural Network},
year = {2022},
pages = {1-6},
doi = {10.1109/ICMEW56448.2022.9859419},
url = {https://doi.ieeecomputersociety.org/10.1109/ICMEW56448.2022.9859419},
publisher = {IEEE Computer Society},
address = {Los Alamitos, CA, USA},
month = {jul}
}
```
