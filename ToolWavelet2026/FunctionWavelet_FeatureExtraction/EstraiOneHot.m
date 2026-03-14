function one_hot_encoded_seq=EstraiOneHot(seq,sequence_length,TwoMerName,properties,randProp,band,NumberNet)

one_hot_encoded_seq=zeros(sequence_length-1,64);
for j = 1:sequence_length-1 %for all the element of the sequence
    for prop=1:4%we use 4 different properties
        pairDNA=[seq(j) seq(j+1)];%a pair of elements of the sequence

        WhatPair=0;
        for TwoMer=1:16
            if isequal(TwoMerName{TwoMer},pairDNA)
                WhatPair=TwoMer;%identify the given pair
            end
        end

        imageDNA=[1:16].*0;
        try
            imageDNA(WhatPair)=properties(randProp{NumberNet}(band,prop),WhatPair);%assign to the given pair the selected property
        catch
            imageDNA=[1:16].*0;
        end
        one_hot_encoded_seq(j, 1+(prop-1)*16:prop*16) = imageDNA;%stores the encoded sequence
    end
end
end

