function one_hot_encoded_seq=TwoHotProperties(dna_sequence,maxLen,TwoMerName,properties,idProperty)
%maxLen %max length for that dataset

if nargin==4
    idProperty=1;
end


seq = upper(dna_sequence); % Convert to uppercase to handle lower/uppercase letters
sequence_length=length(seq);

%for all the bases
for j = 1:sequence_length-1
    pairDNA=[seq(j) seq(j+1)];%a given pair of bases

    %numerical value of the pair of bases, it is used for finding
    %the related property value
    WhatPair=0;
    for TwoMer=1:16
        if isequal(TwoMerName{TwoMer},pairDNA)
            WhatPair=TwoMer;
        end
    end

    imageDNA=[1:16].*0;%inizialization of the column
    try
        imageDNA(WhatPair)=properties(idProperty,WhatPair);%assign to each pair a given value of a
        %property
    end
    one_hot_encoded_seq(j, :) = imageDNA;%matrix that represents the sequence
end

if sequence_length<maxLen
    one_hot_encoded_seq(maxLen,1)=0; %padding
end
