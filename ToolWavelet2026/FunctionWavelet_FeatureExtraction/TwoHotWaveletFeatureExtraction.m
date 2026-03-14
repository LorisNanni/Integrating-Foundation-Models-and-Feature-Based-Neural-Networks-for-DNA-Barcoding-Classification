
%Since the dataset is split into training and test sets in this file, 
% the code for generating images for both is provided below.

LoadDWT;%load set of the applied discrete wavelet


%read the physicochemical properties
fid=fopen('90_sdandard.txt','r')
fscanf(fid,'%s',1);
for TwoMer=1:16
    TwoMerName{TwoMer}=fscanf(fid,'%s',1);%nitrogenous bases pair
end
for prop=1:90 %for each property
    fscanf(fid,'%s',1); %name of the property
    for TwoMer=1:16
        properties(prop,TwoMer)=fscanf(fid,'%f',1);%value for each nitrogenous bases pair
    end
end
fclose(fid)



%% TRAINING SET
% % import FASTA file
sequences = fastaread(strcat(namedata,'Train.fas'));

% % Extract the DNA sequences from the FASTA data
dna_sequences = {sequences.Sequence};
headers = {sequences.Header};

for i = 1:length(dna_sequences)
    tokens = regexp(headers{i}, '\|', 'split');
    if length(tokens) >= 3
        className{i}= tokens{2};%extract the name of the label from the header
    else
        className{i}= tokens{end};%extract the name of the label from the header
    end

end

%Extract the labels and save them as numeric values.
nameClasses=unique(className);
label=zeros(length(dna_sequences),1);
for i = 1:length(dna_sequences)
    for j=1:length(nameClasses)
        if isequal(className{i},nameClasses{j})
            label(i)=j;
        end
    end
end


% Convert DNA sequences to images
for NumberNet=1:NumNet
    one_hot_map = eye(4);

    %random properties for creating a 3-channel image
    for band=1:3
        for prop=1:4 %for each net -> random properties
            randProp{NumberNet}(band,prop) = randi([1, 90]);
        end
    end

    %for each net (feature extraction) a random set of mother wavelet is
    %selected
    for band=1:3
        Selected=[];
        for choose=1:100
            [a,SW]=sort(rand(length(app),1));%app stores the possible wavelets
            Selected=[Selected; SW];
        end
        SelectedWave{NumberNet}{band}=Selected;
    end

    %image creation
    for band=1:3%3 channels
        for barcode = 1:length(dna_sequences)
            seq = upper(dna_sequences{barcode}); % Convert to uppercase to handle lower/uppercase letters
            seq(maxLen)=0;
            sequence_length=maxLen;
            clear one_hot_encoded_seq
            one_hot_encoded_seq=EstraiOneHot(seq,sequence_length,TwoMerName,properties,randProp,band,NumberNet);
            %size(one_hot_encoded_seq)  is "lengh of the sequence"*64 (16 pairs and 4 properties) 
            FinalRapp=BandRepresentation(one_hot_encoded_seq,app,SelectedWave,band,NumberNet);%wavelet approach
            DATA{NumberNet}{barcode}(:,:,band)=single(FinalRapp);%store the images
        end
    end
end

DIM1=length(dna_sequences);

%% TEST SET
sequences = fastaread(strcat(namedata,'Test.fas'));

% % Extract the DNA sequences from the FASTA data
dna_sequences = {sequences.Sequence};
headers = {sequences.Header};

for i = 1:length(dna_sequences)
    tokens = regexp(headers{i}, '\|', 'split');
    if length(tokens) >= 3
        className{i}= tokens{2};
    else
        className{i}= tokens{end};
    end

end

for i = 1:length(dna_sequences)
    for j=1:length(nameClasses)
        if isequal(className{i},nameClasses{j})
            label(i+DIM1)=j;
        end
    end
end

for NumberNet=1:NumNet
    for band=1:3
        % Convert DNA sequences to images
        for barcode = 1:length(dna_sequences)
            seq = upper(dna_sequences{barcode}); % Convert to uppercase to handle lower/uppercase letters
            seq(maxLen)=0;
            clear one_hot_encoded_seq
            one_hot_encoded_seq=EstraiOneHot(seq,sequence_length,TwoMerName,properties,randProp,band,NumberNet);
            FinalRapp=BandRepresentation(one_hot_encoded_seq,app,SelectedWave,band,NumberNet);
            %size(one_hot_encoded_seq)  is "lengh of the sequence"*64 (16 pairs and 4 properties)
            DATA{NumberNet}{barcode+DIM1}(:,:,band)=single(FinalRapp);%store the images

        end
    end
end

DAT{2}=label;
DAT{3}=1:barcode+DIM1;
DAT{4}=DIM1;
DAT{5}=barcode+DIM1;

% save(strcat(namedata,'ForTwoHotMatrix.mat'),'DATA','DAT','-v7.3')
