layers = [
    imageInputLayer(imageSize)

    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.25)

    convolution2dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.25)

    convolution2dLayer(3, 128, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.25)

    % Fully connected
    fullyConnectedLayer(32)
    fullyConnectedLayer(numClasses)

    softmaxLayer
    classificationLayer
    ];