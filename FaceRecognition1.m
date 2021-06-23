function  outputLabel=FaceRecognition1(trainPath, testPath)  
    folderNames=ls(trainPath);
    trainImgSet=zeros(600,600,3,length(folderNames)-2); % all images are 3 channels with size of 600x600
    labelImgSet=folderNames(3:end,:); % the folder names are the labels

    for i=3:length(folderNames)
        imgName=ls([trainPath, folderNames(i,:),'\*.jpg']);
        trainImgSet(:,:,:,i-2)= imread([trainPath, folderNames(i,:), '\', imgName]);
    end



    %% Prepare the training image for training of SVM

    % Extract HOG Features for training set 
    trainingFeatures = zeros(size(trainImgSet,4), 12996); % Matrix to store the feature vectors
    
    for i=1:size(trainImgSet,4)
        tmpI = im2double(imresize(rgb2gray(uint8(trainImgSet(:,:,:,i))), [300 300]));
        face = GetCroppedFace(tmpI); % Cropping the face with Viola-Jones algorithm    
        features = extractHOGFeatures(face); % HOG Feature extraction
        trainingFeatures(i,:) = (features-mean(features))/std(features); % Use zero-mean normalisation
        trainingLabel{i} = char(labelImgSet(i, :));    
    end
    
    % Train SVM Classifier with HoG Features
    t = templateLinear();
    faceClassifier = fitcecoc(trainingFeatures,trainingLabel,'Learners',t,'ObservationsIn','rows');


    %% Prepare the testing image and used for prediction by the SVM

    testImgNames=ls([testPath,'*.jpg']);
    outputLabel=[];
    testingFeatures = zeros(size(testImgNames,1), 12996); 
    for i=1:size(testImgNames,1)

        testImg = im2double(imresize(rgb2gray(uint8(imread([testPath, testImgNames(i,:)]))), [300 300]));
        face = GetCroppedFace(testImg);  % Cropping the face with Viola-Jones algorithm    
        queryFeatures = extractHOGFeatures(face); % HOG Feature extraction
        testingFeatures(i,:) = (queryFeatures-mean(queryFeatures))/std(queryFeatures); % Use zero-mean normalisation

    end
      
     personLabel = predict(faceClassifier,testingFeatures); % Predicting the test images
     outputLabel = char(personLabel);

end

% *** Function written to crop faces of images with the Viola-Jones algorithm
function face = GetCroppedFace(tmpI)
    faceDetector = vision.CascadeObjectDetector('FrontalFaceLBP'); %Create a detector object
    boxBoundary = step(faceDetector,tmpI); % Detect faces
    N = size(boxBoundary,1);
    
    for i=1:N
        face = imcrop(tmpI,boxBoundary(i,:)); % Get Cropped Face
    end
    
    if(isempty(boxBoundary))
        face = imresize(tmpI, [165 165]);
    
    else
        if(size(face,1) ~= 165)
            face = imresize(face, [165 165]);
        end
    end
end