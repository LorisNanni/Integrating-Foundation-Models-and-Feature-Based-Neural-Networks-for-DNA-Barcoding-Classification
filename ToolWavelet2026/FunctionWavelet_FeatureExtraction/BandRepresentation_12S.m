function FinalRapp=BandRepresentation_12S(one_hot_encoded_seq,app,SelectedWave,band,NumberNet)
tmp=1;
clear FinalRapp
FinalRapp=zeros(18*3,1);
for features=1:18%In this case, we create a smaller matrix because the 12S sequence is much shorter than the sequences in the other datasets used.
    vector=one_hot_encoded_seq(:,features);%apply wavelet to each row of the matrix obtained using EstraiOneHot
    [ca,cd] = dwtNoReduction(vector,app{SelectedWave{NumberNet}{band}(features)},'mode','sym');%discrete wavelet
    %ca-> approximation;  cd -> details
    FinalRapp(tmp,1:length(ca(1:3:end)))=ca(1:3:end);
    FinalRapp(tmp+1,1:length(ca(1+1:3:end)))=ca(1+1:3:end);
    FinalRapp(tmp+2,1:length(ca(1+2:3:end)))=ca(1+2:3:end);
    tmp=tmp+3;
end

FinalRapp(max(size(FinalRapp)),max(size(FinalRapp)))=0;%get a square matrix

if max([size(FinalRapp,1) size(FinalRapp,2)])>56
    FinalRapp=imresize(FinalRapp,[56 56]);
end
end