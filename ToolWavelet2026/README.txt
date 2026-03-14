In the following, the different functions and the contents of the folder are explained.
The code is illustrated using a single dataset, which is already included here and split into a single training set and a single test set.
If you need to use different datasets, please read the code carefully and modify it according to your needs. 
The code is designed to perform feature extraction, during which the images are extracted and generated sequentially, first for the training set 
and then for the test set.
Of course, if you need to generate all the images for the entire dataset first and only afterward split them into different folders, you will need to modify the code accordingly.
In any case, the code is commented line by line and should not be difficult to adapt to your specific needs.

TwoHotWaveletExample.m %main: read dataset, call function for feature extraction, run the ensemble and save the scores
%the size of the ensemble is the hyperparameter NumNet, in this example fixed to 2 (See line 16)

%notice that, in 12S sequence (shorter the COI), we have used
BandRepresentation_12S.m
%see line 85 and 126 of TwoHotWaveletFeatureExtraction.m


Folders:
\datasets  %fishes dataset, it is used in the example. In this folder, I also store the matrix of physico-chemical properties.
\functionNet %In this other folder, there is all the information related to the neural network architecture and its parameters.
\FunctionWavelet_FeatureExtraction%Finally, this folder contains the functions used to represent a given DNA barcoding sequence as an image.                           




