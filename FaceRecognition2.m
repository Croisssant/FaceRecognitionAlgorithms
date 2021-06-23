%{
=========================================================================
FaceRecognition2:
This method uses the pre-trained Convolutional Neural Network, VGG-FACE[1]
*    Title: VGG Face Descriptor
*    Author: Omkar M. Parkhi, Andrea Vedaldi, Andrew Zisserman
*    Availability: https://www.robots.ox.ac.uk/~vgg/software/vgg_face/


To obtain the .h5 file of the pre-trained model,
the model used here is first loaded in Keras along with the 
pre-trained weights of the model. 

Pre-Trained weights trained by David Sandberg, and converted to Keras by
Sefik Ilkin Serengil, availble at:
https://sefiks.com/2018/09/03/face-recognition-with-facenet-in-keras/

Loading model of the VGG-FACE guide written by Lakshmi Narayana Santha at: 
https://medium.com/analytics-vidhya/face-recognition-with-vgg-face-in-keras-96e6bc1951d5

Finally, my compiled version of the model including the weights is at:
https://uniofnottm-my.sharepoint.com/:u:/g/personal/hfykl2_nottingham_ac_uk/Ef-6tVG2arZLk7ex8gVGjs0BIPCquT5CF3ySKHV-CmT8yw?e=nI3Y32

** Download of importKerasNetwork is required to load the model into MATLAB
   https://uk.mathworks.com/help/deeplearning/ref/importkerasnetwork.html

** If you can't access my compiled model please let me know at hfykl2@nottingham.ac.uk

[1] O. M. Parkhi, A. Vedaldi, A. Zisserman, Deep Face Recognition, 
    British Machine Vision Conference, 2015
=========================================================================
%}
function  outputLabel = FaceRecognition2(trainPath, testPath)   
    
    %{
    *** Adding augmentated training images to the network to increase accuracy
    *** Uncomment to train the model with augemented images and replace
        imdsTrain
    imds1 = imageDatastore(trainPath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

    imds2 = imds1;

    imdsCombined = imageDatastore(cat(1,imds1.Files,imds2.Files));
    imdsCombined.Labels = cat(1,imds1.Labels,imds2.Labels);

    imdsCombined1 = imdsCombined;
    imdsCombined2 = imageDatastore(cat(1,imdsCombined.Files,imdsCombined1.Files));
    imdsCombined2.Labels = cat(1,imdsCombined.Labels,imdsCombined1.Labels);


    imdsTrain = shuffle(imdsCombined2);   
    imdsTrain.ReadFcn = @customReadDatastoreImage;
    %}
    
    % Loading training and testing data
    imdsTrain = imageDatastore(trainPath, ...
        'IncludeSubfolders',true, ...
        'LabelSource','foldernames');
   
    imdsTest = imageDatastore(testPath);

    % ================= TRAINING START =================
    % Loding the pre-trained model
    modelfile = 'my_vggface.h5';

    net = importKerasNetwork(modelfile, ...
          'OutputLayerType','classification');

    inputSize = net.Layers(1).InputSize;

    if isa(net,'SeriesNetwork') 
      lgraph = layerGraph(net.Layers); 
    else
      lgraph = layerGraph(net);
    end 

    [learnableLayer,classLayer] = findLayersToReplace(lgraph);

    numClasses = numel(categories(imdsTrain.Labels));
    
    % Initializing the replacement layers
    if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
        newLearnableLayer = fullyConnectedLayer(numClasses, ...
            'Name','new_fc', ...
            'WeightLearnRateFactor',15, ...
            'BiasLearnRateFactor',10);

    elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
        newLearnableLayer = convolution2dLayer(1,numClasses, ...
            'Name','new_conv', ...
            'WeightLearnRateFactor',15, ...
            'BiasLearnRateFactor',10);
    end

    lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);

    newClassLayer = classificationLayer('Name','new_classoutput');
    lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

    layers = lgraph.Layers;
    connections = lgraph.Connections;
    
    % Freezing the weights of the pre-trained network
    layers(1:49) = freezeWeights(layers(1:49));
    lgraph = createLgraphUsingConnections(layers,connections);
    
    % Resizing the images to feed into the pre-trained network
    augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
    augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);
    
    % Initializing the training parameters of the pre-trained network
    miniBatchSize = 15;
    options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',10, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'Verbose',false, ...
    'Plots','training-progress');
    
    net = trainNetwork(augimdsTrain,lgraph,options);
    
    % ================= TESTING START =================
    YPred = classify(net,augimdsTest);
    outputLabel = char(YPred);

end


%  ========================= DEEP LEARNING FUNCTIONS PROVIDED BY MATLAB =========================

% findLayersToReplace(lgraph) finds the single classification layer and the
% preceding learnable (fully connected or convolutional) layer of the layer
% graph lgraph.
function [learnableLayer,classLayer] = findLayersToReplace(lgraph)

if ~isa(lgraph,'nnet.cnn.LayerGraph')
    error('Argument must be a LayerGraph object.')
end

% Get source, destination, and layer names.
src = string(lgraph.Connections.Source);
dst = string(lgraph.Connections.Destination);
layerNames = string({lgraph.Layers.Name}');

% Find the classification layer. The layer graph must have a single
% classification layer.
isClassificationLayer = arrayfun(@(l) ...
    (isa(l,'nnet.cnn.layer.ClassificationOutputLayer')|isa(l,'nnet.layer.ClassificationLayer')), ...
    lgraph.Layers);

if sum(isClassificationLayer) ~= 1
    error('Layer graph must have a single classification layer.')
end
classLayer = lgraph.Layers(isClassificationLayer);


% Traverse the layer graph in reverse starting from the classification
% layer. If the network branches, throw an error.
currentLayerIdx = find(isClassificationLayer);
while true
    
    if numel(currentLayerIdx) ~= 1
        error('Layer graph must have a single learnable layer preceding the classification layer.')
    end
    
    currentLayerType = class(lgraph.Layers(currentLayerIdx));
    isLearnableLayer = ismember(currentLayerType, ...
        ['nnet.cnn.layer.FullyConnectedLayer','nnet.cnn.layer.Convolution2DLayer']);
    
    if isLearnableLayer
        learnableLayer =  lgraph.Layers(currentLayerIdx);
        return
    end
    
    currentDstIdx = find(layerNames(currentLayerIdx) == dst);
    currentLayerIdx = find(src(currentDstIdx) == layerNames);
    
end

end


% layers = freezeWeights(layers) sets the learning rates of all the
% parameters of the layers in the layer array |layers| to zero.

function layers = freezeWeights(layers)

for ii = 1:size(layers,1)
    props = properties(layers(ii));
    for p = 1:numel(props)
        propName = props{p};
        if ~isempty(regexp(propName, 'LearnRateFactor$', 'once'))
            layers(ii).(propName) = 0;
        end
    end
end

end


% lgraph = createLgraphUsingConnections(layers,connections) creates a layer
% graph with the layers in the layer array |layers| connected by the
% connections in |connections|.

function lgraph = createLgraphUsingConnections(layers,connections)

lgraph = layerGraph();
for i = 1:numel(layers)
    lgraph = addLayers(lgraph,layers(i));
end

for c = 1:size(connections,1)
    lgraph = connectLayers(lgraph,connections.Source{c},connections.Destination{c});
end

end

%  ========================= FUNCTION I CREATED FOR IMAGE AUGMENTATION =========================
function dataOut = customReadDatastoreImage(filename)
    % Random number to pick an augmentation
    r = randi([1 7],1);
    data = imread(filename);
    
    % List of augementation processes
    switch r
        case 1
            sigma = 1+5*rand; 
            dataOut = imgaussfilt(imnoise(data, 'gaussian'),sigma); 
        case 2
            
            dataOut = imnoise(data, 'gaussian'); 
        case 3
            sigma = 1+5*rand; 
            dataOut = imgaussfilt(data, sigma); 
            
        case 4
            tformR = randomAffine2d('Rotation',[10 45]); 
            outputView = affineOutputView(size(data),tformR);
            data = imwarp(data,tformR,'OutputView',outputView);

            tformS = randomAffine2d('Scale',[1.2,1.5]);
            outputView = affineOutputView(size(data),tformS);
            dataOut = imwarp(data,tformS,'OutputView',outputView);
            
        case 5
            tformR = randomAffine2d('Rotation',[-45 -10]); 
            outputView = affineOutputView(size(data),tformR);
            data = imwarp(data,tformR,'OutputView',outputView);

            tformS = randomAffine2d('Scale',[1.2,1.5]);
            outputView = affineOutputView(size(data),tformS);
            dataOut = imwarp(data,tformS,'OutputView',outputView);
            
        case 6
            contrastFactor = 1-0.2*rand; 
            brightnessOffset = 0.3*(rand-0.5); 
            dataOut = data.*contrastFactor + brightnessOffset;
        
        case 7 
            dataOut = data;
            
    end
end