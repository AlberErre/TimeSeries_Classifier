%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% DATA PREPROCESSING (NEW data)

%dnet.performaFcn = 'mse'
%dnet.layers{2}.transferFcn = 'tansig'
% patternnet output = 'softmax', en lugar de 'tansig'

testset = []
name = ['train_data_6inputs.csv'];
a = csvread(name);
testset = [testset;a];

trainset = []
name = ['new_data_mcdos.csv'];
a = csvread(name);
trainset = [trainset;a];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

%% Support Vector Machines

miss_matrix = [0 1 12 1 1;
               1 0 1 1 1;
               1 1 0 1 1;
               1 1 1 0 1;
               1 1 1 1 0]; % Applying "12" to 3-1 internception,
% the accuracy reaches 79%

s = templateSVM('KernelFunction','linear'); %polynominal, linear, gaussian, rbf
svmnet = fitcecoc(x, t, 'Learners',s,'Cost',miss_matrix);
L = resubLoss(svmnet,'LossFun','classiferror');

% We need to improve and prove parameters, all avaible here:
% help ClassificationECOC  <--- HERE

% Predict Train Data
svpredict_train = predict(svmnet,x);

% Predict Test Data
svpredict_test = predict(svmnet,xt);

% ACCURACY Train Data
ACCURACY_sv_train = sum(svpredict_train == t)/length(t)*100
ConfusionMat_svnet_train = confusionmat(t', svpredict_train)
                                        
% ACCURACY Test Data
ACCURACY_sv_test = sum(svpredict_test == tt)/length(tt)*100
ConfusionMat_svnet_test = confusionmat(tt', svpredict_test)

