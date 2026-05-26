function featureVector=FCGR_FeatureExtraction(dna_sequence,kMer)


% Convert DNA sequences to images
% This script computes the Frequency Chaos Game Representation
% for the input sequence (using k-mer probabilities)

data = upper(dna_sequence);  % Ensure sequence is in uppercase
kmer_count = count_kmers(data, kMer);
kmer_prob = probabilities(kmer_count, kMer, data);
chaos = chaos_game_representation(kmer_prob, kMer);

% Resize the chaos image to 224x224 using nearest-neighbor interpolation
chaos_resized = imresize(chaos, [224, 224], 'nearest');

% Normalize the color values to the range [0,1]
max_val = max(chaos_resized(:));
if max_val > 0
    featureVector = chaos_resized / max_val;
else
    featureVector = chaos_resized;
end
featureVector=featureVector.*255;%normalization to [0 255]
end


%% FUNCTIONS

function color = base_color(base)
% Returns a 1x3 color vector corresponding to a DNA base.
switch base
    case 'A'
        color = [1, 0, 0];   % Red
    case 'C'
        color = [0, 1, 0];   % Green
    case 'G'
        color = [0, 0, 1];   % Blue
    case 'T'
        color = [1, 1, 0];   % Yellow
    otherwise
        color = [0, 0, 0];   % Black for other characters
end
end

function color = custom_color(kmer)
% Computes a custom color for a k-mer by taking a weighted average of base colors.
total_bases = length(kmer);
color = zeros(1,3);
unique_bases = unique(kmer);
for i = 1:length(unique_bases)
    base = unique_bases(i);
    count = sum(kmer == base);
    color = color + (count/total_bases)*base_color(base);
end
end

function d = count_kmers(sequence, k)
% Counts all k-mers in the sequence (ignores those containing 'N').
d = containers.Map();
N = length(sequence) - k + 1;
for i = 1:N
    kmer = sequence(i:i+k-1);
    if isKey(d, kmer)
        d(kmer) = d(kmer) + 1;
    else
        d(kmer) = 1;
    end
end
% Remove keys containing 'N'
keys_list = keys(d);
for i = 1:length(keys_list)
    if contains(keys_list{i}, 'N')
        remove(d, keys_list{i});
    end
end
end

function probs = probabilities(kmer_count, k, sequence)
% Computes the probability of each k-mer from the k-mer counts.
probs = containers.Map();
N = length(sequence);
keys_list = keys(kmer_count);
for i = 1:length(keys_list)
    key = keys_list{i};
    count = kmer_count(key);
    probs(key) = count / (N - k + 1);
end
end

function chaos = chaos_game_representation(probabilities, k)
% Computes the chaos game representation image for given k-mer probabilities.
% The output image size is 2^k x 2^k x 3.
array_size = 2^k;  % since sqrt(4^k)=2^k
chaos = zeros(array_size, array_size, 3);

keys_list = keys(probabilities);
for i = 1:length(keys_list)
    key = keys_list{i};
    value = probabilities(key);
    maxx = double(array_size);
    maxy = double(array_size);
    posx = 1;
    posy = 1;
    for j = 1:length(key)
        charNow = key(j);
        if charNow == 'T'
            posx = posx + maxx/2;
        elseif charNow == 'C'
            posy = posy + maxy/2;
        elseif charNow == 'G'
            posx = posx + maxx/2;
            posy = posy + maxy/2;
        end
        maxx = maxx/2;
        maxy = maxy/2;
    end
    % Round to nearest index and ensure indices are within bounds
    ix = round(posy);
    iy = round(posx);
    ix = min(max(ix,1), array_size);
    iy = min(max(iy,1), array_size);
    color = custom_color(key);
    chaos(ix, iy, :) = color * value;
end
end


