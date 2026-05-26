clear all

%load the 2-mer properties, 
%the values are saved in the matrix "properties" (90*16)
%the sort of 2-mer in "TwoMerName"
load 2merProperties.mat

%read the sequence
sequence = load('dna_sequence.mat');
sequence=sequence.dna_sequence;

maxLen=650;% max length of the sequences of your dataset, it is used for padding
idProperty=1;%what property to use for feature extraction
featureVector2Hot=TwoHotProperties(sequence,maxLen,TwoMerName,properties,idProperty);

%to extract feature related to chaos game approach
kMer=5;%what value of 'k' in kMer to use (we have used 5 and 6)
featureVectorCG=FCGR_FeatureExtraction(sequence,kMer);