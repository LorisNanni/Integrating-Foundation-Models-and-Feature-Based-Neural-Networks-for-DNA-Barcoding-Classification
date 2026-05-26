
90_sdandard.txt   DNA bases pair physical properties
dna_sequence.mat an example of DNA sequence
2merProperties.mat  2-mer properties used in 2-hot


The tool includes the functions required to extract the representation of a DNA barcoding sequence considering the approches used
in eDNA: 2-hot (section 2.2.3) and chaos game based approach (section 2.2.1); 
Afterwards, you can use any network/classifier trained on these representations. 


The examples for feature extraction are:
ExampleFeatureExtractionGivenAsequence.m   for both 2-hot approach and chaos game based approach

For instance, the MATLAB code for the CNN1 and ATT networks is:

%         % Define the CNN1 architecture
%         layers = [
%             imageInputLayer(imageSize)   % Input layer for specified image size
%             convolution2dLayer(3, 16, 'Padding', 'same')
%             batchNormalizationLayer()
%             dropoutLayer()
%             reluLayer()
%             fullyConnectedLayer(8)
%             fullyConnectedLayer(numClasses)
%             softmaxLayer()
%             classificationLayer()
%             ];


%         % Define the ATT architecture
%         layers = [
%             imageInputLayer(imageSize)
%             flattenLayer('Name', 'flatten')
%             selfAttentionLayer(8,64)
%             bilstmLayer(100,Name="bilstm1")
%             batchNormalizationLayer()
%             fullyConnectedLayer(numClasses)
%             softmaxLayer()
%             classificationLayer()
%             ];
%


