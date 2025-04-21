% Demo for Incomplete Multi-view Subspace Clustering

clear; 
close all;
addpath('ClusteringMeasure');
addpath(genpath('Funs'))
data_path = './Data/';

% Algorithm Settings
Data_name = 'BBCSport';

miss_rate = 0.1;
views = 2;
%% Loading data
fprintf('Testing %s... Miss_Rate: %.2f\n',Data_name,miss_rate); 
load(fullfile(data_path,strcat(Data_name,'.mat')));
load(fullfile(data_path,strcat(Data_name,'_',num2str(miss_rate),'.mat')));
for k=1:views
    eval(sprintf('X{%d} = double(X%d);', k, k));
end
cls_num = length(unique(gt));
K = views;

%% Records
alg_name = 'TSpNF-IMVSC'; 
alg_time = zeros(10,1);
NMI = zeros(10,1);
ACC = zeros(10,1);
AR = zeros(10,1);
fscore = zeros(10,1);   
precision = zeros(10,1);
recall = zeros(10,1);
purity =zeros(10,1);

C1 = cell(10,1);     % clustering results
S1 = cell(10,1);     % affinity matrices
Out1 = cell(10,1);   % metrics

%% Algs Running
Y = X;
for iv=1:K
    [Y{iv}]=NormalizeData(X{iv});
end 

% Parameter settings
opts.lambda1 = 5e-6; %[1e-4,5e-5,1e-5,5e-6,1e-6];
opts.lambda2 = 1e-6; %[1e-4,5e-5,1e-5,5e-6,1e-6,5e-7,1e-7];
opts.lambda3 = 1e-5; %[5e-4,1e-4,5e-5,1e-5,5e-6];
opts.d = K;  % number of views
opts.flag_debug = 0;
opts.maxIter = 200;
opts.epsilon = 1e-7;
opts.mu = 1e-5; 
opts.max_mu = 1e10; 
opts.pho_mu = 2;  

for kk = 1:length(index)
    % Construct missing data
    Yi = Y;
    ind = index{kk};
    for i=1:length(Y)
        Yiv = Yi{i};
        indi = ind(:,i);
        pos = find(indi==0);
        Yiv(:,pos)=[]; 
        Yi{i} = Yiv;
    end   
    time_start = tic;

    [C, S, Out] = solving_IMVSC(Yi, ind, cls_num, gt, opts); 
    
    Out.time= toc(time_start);
    alg_time(kk) =  Out.time;
    NMI(kk) = Out.NMI;
    AR(kk) = Out.AR;
    ACC(kk) = Out.ACC;
    recall(kk) = Out.recall;
    precision(kk) = Out.precision;
    fscore(kk) = Out.fscore; 
    purity(kk) = Out.purity; 
    C1{kk} = C;
    S1{kk} = S;
    Out1{kk} = Out;
end
%% Results report
fprintf('%6s\t%12s\t%4s\t%4s\t%4s\t%4s\t%4s\t%4s\t%4s\t%4s\n','Stats', ...\
 'Algs', 'Time', 'NMI', 'AR', 'ACC', 'Recall', 'Pre', 'F-Score', 'Purity');
fprintf('%6s\t%12s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n',...\
'Mean', alg_name,mean(alg_time),mean(NMI),mean(AR),...\
mean(ACC),mean(recall),mean(precision),mean(fscore),mean(purity));   
fprintf('%6s\t%12s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n',...\
'Std', alg_name,std(alg_time),std(NMI),std(AR),...\
std(ACC),std(recall),std(precision),std(fscore),std(purity));

