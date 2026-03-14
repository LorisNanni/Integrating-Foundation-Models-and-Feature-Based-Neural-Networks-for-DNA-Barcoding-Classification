%all the mother wavelet used in the following
tmp=1;
app{tmp}=  'haar'  ; tmp=tmp+1; %   : Haar wavelet
app{tmp}='db6'    ; tmp=tmp+1; %   : Daubechies wavelets
app{tmp}= 'sym6'    ; tmp=tmp+1; %  : Symlets
app{tmp}= 'coif2'  ; tmp=tmp+1; %   : Coiflets
app{tmp}= 'bior2.2'   ; tmp=tmp+1; %  : Biorthogonal wavelets
app{tmp}=  'rbio2.2'  ; tmp=tmp+1; %   : Reverse biorthogonal wavelets
app{tmp}= 'dmey'  ; tmp=tmp+1; %   : Discrete Meyer wavelet
app{tmp}= 'fk4'    ; tmp=tmp+1; %   : Fejer-Korovkin orthogonal wavelets
app{tmp}= 'beyl'   ; tmp=tmp+1; %  : Beylkin orthogonal wavelet
app{tmp}= 'vaid'   ; tmp=tmp+1; %  : Vaidyanathan orthogonal wavelet