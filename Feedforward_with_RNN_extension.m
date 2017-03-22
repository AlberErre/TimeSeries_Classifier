%% MODEL SELECTION
% Do a Direct Feedforward Net
ffwd = 1;
% Do a Recurrent Neural  Net
recn = 1;

%% DATA PREPROCESSING (NEW data)

testset = []
name = ['train_data_6inputs.csv'];
a = csvread(name);
testset = [testset;a];

trainset = []
name = ['new_data_mcdos.csv'];
a = csvread(name);
trainset = [trainset;a];

%% INPUTS AND OUTPUTS

% TRAIN DATA
x = trainset(:,2:7);
t = trainset(:,8); % now we have 5 categories
results = (dummyvar(t))';

% TEST DATA
xt = testset(:,2:7);
tt = testset(:,8); % now we have 5 categories
results_test = (dummyvar(tt))';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PREPARE INPUTS 
% Using an independent entry for each input to increase accuracy
x1 = trainset(:,2)';
x2 = trainset(:,3)';
x3 = trainset(:,4)';
x4 = trainset(:,5)';
x5 = trainset(:,6)';
x6 = trainset(:,7)';

z = {x1, x2, x3, x4, x5, x6};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% FEEDFORWARD NN (DIRECT) ---> "dnet"
if ffwd == 1

% NETWORK PARAMETERS
number_hidden_nodes = 21 % 21 is good!
training_method = 'trainlm' % 'trainlm' provide the best results!! (default)
    % In the report, comment the difference in efficient between gradient
    % descent (traingd, traingdm and traingdx) and Levenberg-Marquardt
    % (trainlm). gradient descent is slow in TIME, needs a lot of epochs
    % and have bad accuracy respect 'trainlm'. Ww must say it in the report
    % :D
dnet = feedforwardnet(number_hidden_nodes,training_method);
dnet.numInputs = 6
dnet.inputConnect = [1 1 1 1 1 1; 0 0 0 0 0 0]
dnet.divideFcn = 'dividerand' % the way we divide the data! (randomly)
dnet.layers{1}.transferFcn = 'elliotsig'; % tansig softmax satlins - elliotsig --> this is the best (activation function) 
dnet.layers{2}.transferFcn = 'elliotsig'; % tansig softmax satlins elliotsig
dnet.divideParam.trainRatio = 90/100; %90/100 is perfect!
dnet.divideParam.valRatio = 5/100;
dnet.divideParam.testRatio = 5/100;
dnet.trainParam.max_fail = 1000 % avoid stop training because of failed validation
dnet.trainParam.epochs = 200; % define number of epochs to train
                              % 50 epochs are enough for us to reach 
% TRAINING 
[dnet, tr] = train(dnet,z',results); view(dnet)
plotperform(tr) % this plots the performance in every iteration
%plottrainstate(tr)% display the training states values

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% TRAIN DATA (accuracy)

% predicting (train data)
feedforward_prediction = cell2mat(dnet(z'));
dnet_test = (vec2ind(results))';
dnet_predict = (vec2ind(feedforward_prediction))';
% Calculating accuracy (train set)
ACCURACY_feedforward = sum(dnet_predict == dnet_test)/length(dnet_test)*100;
ACCURACY_feedforward
ConfusionMat_dnet = confusionmat(dnet_test, dnet_predict);
ConfusionMat_dnet

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% TEST DATA (accuracy)

% Using an independent entry for each input to increase accuracy
x1_test = testset(:,2)';
x2_test = testset(:,3)';
x3_test = testset(:,4)';
x4_test = testset(:,5)';
x5_test = testset(:,6)';
x6_test = testset(:,7)';

z_test = {x1_test, x2_test, x3_test, x4_test, x5_test, x6_test};

% predicting (test data)
TEST_feedforward_prediction = cell2mat(dnet(z_test'));
TEST_dnet_test = (vec2ind(results_test))';
TEST_dnet_predict = (vec2ind(TEST_feedforward_prediction))';
% Calculating accuracy (test set)
TEST_ACCURACY_feedforward = sum(TEST_dnet_predict == TEST_dnet_test)/length(TEST_dnet_test)*100;
TEST_ACCURACY_feedforward
TEST_ConfusionMat_dnet = confusionmat(TEST_dnet_test, TEST_dnet_predict);
TEST_ConfusionMat_dnet

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                 
%% RECURRENT NN
                             
if recn == 1

% NETWORK PARAMETERS
number_hidden_nodes = 21 % 21 is good!
training_method = 'trainlm' % 'trainlm' provide the best results!! (default)
    % In the report, comment the difference in efficient between gradient
    % descent (traingd, traingdm and traingdx) and Levenberg-Marquardt
    % (trainlm). gradient descent is slow in TIME, needs a lot of epochs
    % and have bad accuracy respect 'trainlm'. Ww must say it in the report
    % :D
rnet = layrecnet(1:3,number_hidden_nodes,training_method);
               % 1:3 give us 88%
rnet.numInputs = 6;
rnet.inputConnect = [1 1 1 1 1 1; 0 0 0 0 0 0]
rnet. layerConnect = [1 0; 1 0]
rnet.divideFcn = 'dividerand'; % the way we divide the data!
rnet.layers{1}.transferFcn = 'elliotsig'; % tansig softmax satlins elliotsig --> this is the best (activation function) 
rnet.layers{2}.transferFcn = 'elliotsig';
rnet.divideParam.trainRatio = 90/100;
rnet.divideParam.valRatio = 5/100;
rnet.divideParam.testRatio = 5/100;
rnet.trainParam.max_fail = 100; % avoid stop training because of fail validation
rnet.trainParam.epochs = 20; % define number of epochs to train
                              % 60 epochs are enough for us!!

% TRAINING 
%[Xs,Xi,Ai,Ts] = preparets(rnet,cell2mat(z'),results');
[rnet r_tr] = train(rnet,z',results); view(rnet);
%[rnet, r_tr] = train(rnet,z',results); view(rnet)
plotperform(r_tr) % this plots the performance in every iteration
%plottrainstate(tr)% display the training states values

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% TRAIN DATA (Recurrent NN accuracy)

% predicting (train data)
recurrent_prediction = cell2mat(rnet(z'));
rnet_test = (vec2ind(results))';
rnet_predict = (vec2ind(recurrent_prediction))';
% Calculating accuracy (train set)
ACCURACY_recurrent = sum(rnet_predict == rnet_test)/length(rnet_test)*100;
ACCURACY_recurrent
ConfusionMat_rnet = confusionmat(rnet_test, rnet_predict);
ConfusionMat_rnet

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% TEST DATA (Recurrent NN accuracy)

% Using an independent entry for each input to increase accuracy
x1_test = testset(:,2)';
x2_test = testset(:,3)';
x3_test = testset(:,4)';
x4_test = testset(:,5)';
x5_test = testset(:,6)';
x6_test = testset(:,7)';

z_test = {x1_test, x2_test, x3_test, x4_test, x5_test, x6_test};

% predicting (test data)
TEST_recurrent_prediction = cell2mat(rnet(z_test'));
TEST_rnet_test = (vec2ind(results_test))';
TEST_rnet_predict = (vec2ind(TEST_recurrent_prediction))';
% Calculating accuracy (test set)
TEST_ACCURACY_recurrent = sum(TEST_rnet_predict == TEST_rnet_test)/length(TEST_rnet_test)*100;
TEST_ACCURACY_recurrent
TEST_ConfusionMat_rnet = confusionmat(TEST_rnet_test, TEST_rnet_predict);
TEST_ConfusionMat_rnet



