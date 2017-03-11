%% MODEL SELECTION
% Do a Direct Feedforward Net
ffwd = 1;
% Do a Recurrent Neural  Net
recn = 1;
% Do a Batch Feedforward Net
batch_ffwd = 0;
% Do a Multi-class classification
multi == 1;

%% DATA PREPROCESSING
i=15
trainset =[]
while i>3,
name = [num2str(i) '.csv'];
a = csvread(name);
trainset = [trainset;a]; %dont be stupid STUPID!
i=i-1
end

a=0
i=3
testset =[]
while i>0,
name = [num2str(i) '.csv'];
a = csvread(name);
testset = [testset;a]; %dont be stupid STUPID!
i=i-1
end

% Deleteall Zeroes
DeleteZero = trainset(:,5) == 0;
trainset(DeleteZero,:) = [];
DeleteZero = testset(:,5) == 0;
testset(DeleteZero,:) = [];

%% INPUTS AND OUTPUTS

% TRAIN DATA
x = trainset(:,2:4);
t = trainset(:,5);
results = dummyvar(t);

% TEST DATA
xt = testset(:,2:4);
tt = testset(:,5);
test_results = dummyvar(tt);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% FEEDFORWARD NN (DIRECT) ---> "dnet"
if ffwd == 1

% SETTING the algorithm with the inputs(x) and classification(t)
dnet = feedforwardnet(80,'trainscg');
dnet.trainParam.epochs = 100; % define 100 epochs to train

% TRAINING the algorithm using the x' and results', same result as applying (') directly above
dnet = train(dnet,x',results');
view(dnet)
                                                                             
% PREDICTING using the feedforward (training data)
feedforward_prediction = dnet(x');
direct_feedforward_performance = perform(dnet,feedforward_prediction,results') % performance obtained
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% FEEDFORWARD NN (MINI BATCH) ---> "net"
if batch_ffwd == 1
                            
% SETTING the algorithm with the inputs(x) and classification(t)
net = feedforwardnet(10,'trainscg'); %sgdm for stochastic! traingd is gradient decent function, backpropagation
mini_batch_size = 15; 
    % The "window" takes 15 inputs to train the net in every epoch
number_of_epochs = 100;
    % We have 1,520,606 examples, 
    %if we aproximate to 1,500,000 /15 = 100,000 epochs to feed all data
net.trainParam.epochs=1;
e=1;

% TRAINING the algorithm using the loop for batching
while i < number_of_epochs
    x_stochastic_init_chunk = randperm(size(x',2));
    % "x_stochastic_init_chunk" makes the Stocastic effect!!
    % QUITE IMPORTANT - basically it init the data randomly!

    for i=1:number_of_epochs % TRAINING
        k = x_stochastic_init_chunk(1+mini_batch_size*(i-1) : mini_batch_size*i);
        [net tr] = train(net, x(:,k)', results(:,k)'); % [] to save all trainings
    end
    %perf = perform(net,feedforward_prediction,results') % performance obtained
    perf(e) = mean(mean(abs(t-net(x'))))
    e=e+1;
end

% PREDICTING using the BATCH feedforward (training data)
%batch_feedforward_prediction = net(x'); % NO IDEA HOW TO IMPLEMENT IT, do I need to use []?
%batch_feedforward_performance = perform(net,batch_feedforward_prediction,results') % performance obtained
                                                                                
end
                                  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% RECURRENT NN
                             
if recn == 1
                            
% SETTING the algorithm with the inputs(x) and classification(t)
recurrent_net = layrecnet(1:10,10,'traingd'); %1:2 is the delay ; traingd is gradient decent function, backpropagation
          
% [Xs,Xi,Ai,Ts] = recurrent_net(recurrent_net,x',results');
% check: https://es.mathworks.com/help/nnet/ref/layrecnet.html
                             
% TRAINING the algorithm using the x' and results', same result as applying (') directly above
% coutputlayer = classificationLayer('Name','coutput') % WHAT is this??
recurrent_net = train(recurrent_net,x',results');
view(recurrent_net)
                                                                                         
% PREDICTING using Recurrent Neural Net (training data)
recurrent_net_prediction = recurrent_net(x'); % it should be 100% because it's using training data
recurrent_performance = perform(recurrent_net,recurrent_net_prediction,results')

end
           
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                             
%% TESTING THE ALGORITHMS

% Direct Feedforward Neural Net (TEST data)
direct_test_feedforward_prediction = dnet(xt');
direct_feedforward_performance = perform(dnet,direct_test_feedforward_prediction',test_results')

% Batch Feedforward Neural Net (TEST data)
%batch_test_feedforward_prediction = net(xt');  % NO IDEA, I need to use [] or something
%batch_feedforward_performance = perform(net,batch_test_feedforward_prediction',test_results')
                                  
% Recurrent Neural Net (TEST data)
recurrent_test_feedforward_prediction = recurrent_net(xt');
recurrent_feedforward_performance = perform(recurrent_net,recurrent_test_feedforward_prediction',test_results')
                                                      
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% MODELS ACCURACY
                                                      
% Direct Feedforward
                                                      
dnet_tind = vec2ind(test_results);
dnet_yind = vec2ind(direct_test_feedforward_prediction);
dnet_percentErrors = sum(dnet_tind ~= dnet_yind)/numel(dnet_tind);
accuracy_direct_feedforward = sum(dnet_tind == dnet_yind)/numel(dnet_tind);
accuracy_direct_feedforward

%c_matrix_direct_feedforward = confusionmat(dnet_yind,dnet_tind);
figure
plotconfusion(dnet_tind, dnet_yind);
                                                      
% Batch Feedforward
                                                      
%net_tind = vec2ind(test_results);
%net_yind = vec2ind(batch_test_feedforward_prediction);
%net_percentErrors = sum(net_tind ~= net_yind)/numel(net_tind);
%Accuracy_batch_feedforward = sum(net_tind == net_yind)/numel(net_tind);
%Accuracy_batch_feedforward
                                                      
%c_matrix_batch_feedfoward = confusionmat(net_yind,net_tind);
figure
%plotconfusion(net_tind, net_yind);
                                                                                                    
% Recurrent NN

recurrent_tind = vec2ind(test_results);
recurrent_yind = vec2ind(recurrent_test_feedforward_prediction);
recurrent_percentErrors = sum(recurrent_tind ~= recurrent_yind)/numel(recurrent_tind);
Accuracy_recurrentNN = sum(recurrent_tind == recurrent_yind)/numel(recurrent_tind);
Accuracy_recurrentNN
                                                      
%c_matrix_recurrent = confusionmat(recurrent_yind,recurrent_tind);
figure
plotconfusion(recurrent_tind, recurrent_yind);
                                                      
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% MULTI-CLASS classification (another algorithm by artur suggestions)
     
if multi == 1
                                                      
% SETTING the algorithm with the inputs(x) and classification(t)
multiclass = patternnet(80);
multiclass.trainParam.epochs = 100; % define 100 epochs to train
                                                      
% TRAINING the algorithm using the x' and results', same result as applying (') directly above
multiclass = train(multiclass,x',results');
view(multiclass)
                      
% PREDICTING using multi-class (training data)
multiclass_prediction = multiclass(x');
multiclass_performance = perform(multiclass,multiclass_prediction,results') % performance obtained
                                                                                                                                                 
% PREDICTING using multi-class (TEST data)
multiclass_test_prediction = multiclass(xt');
multiclass_test_performance = perform(multiclass,multiclass_test_prediction,test_results')

% ACCURACY
multi_tind = vec2ind(test_results);
multi_yind = vec2ind(multiclass_prediction);
multi_percentErrors = sum(multi_tind ~= multi_yind)/numel(multi_tind);
accuracy_multi_class = sum(multi_tind == multi_yind)/numel(multi_tind);
accuracy_multi_class
                                 
end
              





