clc
clear
close all
%%
load 'input.mat'
load 'output_new.mat'
trainD=input;
[m,n,o,p]=size(trainD);
output_label=output_new;
targetD=categorical(output_label);

%% Define Network Architecture
% Define the convolutional neural network architecture.
layers = [
    imageInputLayer([m n o])  %refers to number of features per sample
    convolution2dLayer(m,n,'Stride',4)
    reluLayer
    fullyConnectedLayer(100) % 384 refers to number of neurons in next FC hidden layer
    fullyConnectedLayer(100) % 384 refers to number of neurons in next FC hidden layer
    fullyConnectedLayer(4) % 6 refers to number of neurons in next output layer (number of output classes)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm',...
    'MaxEpochs',50, ...
    'Plots','training-progress');

net = trainNetwork(trainD,targetD',layers,options);

predictedLabels = classify(net,trainD)';
