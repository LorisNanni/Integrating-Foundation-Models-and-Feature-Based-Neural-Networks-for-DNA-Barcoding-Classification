clear all
warning off

gpuDevice(1)%to use GPU

%in this example, we use "fishes" dataset, see the folder \datasets
namedata='fishes';
maxLen=718;%max lenght of the sequences of that dataset

% In the following function, we create the structure where the images for all 
% the NumberNet feature extraction methods used are stored, so that the entire ensemble 
% can be trained, both for training and testing, since the dataset is already split into two parts. 
% In addition, the dataset is saved in a structure that allows it to be reused later. If you want to disable the saving step, 
% you can modify the function. In this case, the dataset is saved and, to make the process clearer, in the next line we load the 
% structure that has just been created.
NumNet=2;%Number of different feature extractions performed. For each feature extraction, a different network will be trained.
%for a quick test, we have set to 2. Hyperparameters used in
%TwoHotWaveletFeatureExtraction.m
TwoHotWaveletFeatureExtraction;


%load the images create with TwoHotWaveletFeatureExtraction.m
load(strcat(namedata,'ForTwoHotMatrix.mat'),'DATA','DAT')


%ensemble of neural nets
for NumberNet=1:length(DATA)%one net of each different feature extraction run

    close all force%close the training plot


    %split training/test
    DIV=DAT{3};%split - training / test set (see TwoHotWaveletFeatureExtraction.m)
    NX=DATA;%images
    label=DAT{2};%label vector
    DIM1=DAT{4};%number of training patterns
    DIM2=DAT{5};%numero of patterns

    trainPattern=(DIV(1:DIM1));%id of the patterns that belong to the training
    testPattern=(DIV(DIM1+1:DIM2));%id of the patterns that belong to the test
    TRlabel=label(trainPattern);%label of the training set
    TElabel=label(testPattern);%label of the test set
    numClasses = max(TRlabel);%number of classes 


    %create the training set
    clear trainingImages
    % Create the training images array
    imageSize=[size(NX{NumberNet}{DIV(1)},1) size(NX{NumberNet}{DIV(1)},2) size(NX{NumberNet}{DIV(1)},3)];
    trainingImages = zeros(imageSize(1), imageSize(2), imageSize(3), length(TRlabel));
    for pattern=1:length(TRlabel)
        trainingImages(:,:,:,pattern)=NX{NumberNet}{DIV(pattern)};
    end
    augmentedTrainingImages = augmentedImageDatastore(imageSize,trainingImages,categorical(TRlabel'));


    % Define the CNN architecture
    DefineArchitecture;

    % Training options with scheduled learning rate,In this example, we do not evaluate the performance on the training set every delta epochs, 
    % in order to make the system faster and therefore debug and modify it more quickly, 
    % also depending on how you formatted your dataset. 
    % Adjust the training parameters so that they are better suited to your specific problem.
    TrainingOpt;

    %create the test set
    clear testImages
    % Create the test images array
    testImages = zeros(imageSize(1), imageSize(2), imageSize(3), length(ceil(DIM1)+1:ceil(DIM2)));
    for pattern=ceil(DIM1)+1:ceil(DIM2)%id of the test patterns
        testImages(:,:,:,pattern-ceil(DIM1))=NX{NumberNet}{DIV(pattern)};
    end

    %train the net
    trainedNet = trainNetwork(augmentedTrainingImages, layers, options);

    %test the trained net, scoreTwoHot stores the scores of the net
    [outclass, scoreTwoHot{NumberNet}] =  classify(trainedNet,testImages);

    %save the scores for further fusion
    save(strcat(namedata,'TwoHotWavelet.mat'),'scoreTwoHot')


end
