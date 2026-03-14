function FinalRapp=BandRepresentation(one_hot_encoded_seq,app,SelectedWave,band,NumberNet)
tmp=1;
clear FinalRapp
FinalRapp=zeros(64*3,1);
for features=1:64 
    vector=one_hot_encoded_seq(:,features);%apply wavelet to each row of the matrix obtained using EstraiOneHot
    [ca,cd] = dwtNoReduction(vector,app{SelectedWave{NumberNet}{band}(features)},'mode','sym');%discrete wavelet
    %ca-> approximation;  cd -> details
    FinalRapp(tmp,1:length(ca(1:3:end)))=ca(1:3:end);
    FinalRapp(tmp+1,1:length(ca(1+1:3:end)))=ca(1+1:3:end);
    FinalRapp(tmp+2,1:length(ca(1+2:3:end)))=ca(1+2:3:end);
    tmp=tmp+3;
end
FinalRapp(max(size(FinalRapp)),max(size(FinalRapp)))=0;%get a square matrix

%create a 224*224 image
if max([size(FinalRapp,1) size(FinalRapp,2)])>224
    FinalRapp=imresize(FinalRapp,[224 224]);
end
end