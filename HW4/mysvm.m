
function [] = Main() %% Main Function --> Please uncomment a desire Step
clc;clear all;close all;
Step1_2;
%Step 3 % is about making a kernel to be used in following Steps
Step4_5
Step6_Primal
Step6_Dual
 Step7

Cvals = [0.1 1 10 100]; %% value of C
Cval = 1;
Step8(Cval)

Sigmas = [0.2 1 10]; %% value of sigma
sigma = 0.2;
Step9(sigma)
end

%%
function [] = Step1_2()%% Step 1  --- Soft margin

X=[1 0; 2 1; 2 -1; -1 0 ; -2 1; -2 -1]; %% input Data
Y=[1;1;1;-1;-1;-1]; %% input Label

%% initial Value for quadprog
N = size(X,1); %%  data Length 
K = Gram_Mat(X,X,'Linear',1); %% Calcute Gram matrix with data input and linear kernel and kernel parameter
H = (Y*Y').*K + 1e-5*eye(N); %% Calcute H for primal problem
f = ones(N,1);
A = [];
b = [];
LB = zeros(N,1); %% Lower band
UB = ones(N,1); %% Higher band
Aeq = Y';
beq = 0;
warning off

%%%%%%%%%%%%%%%%%%%%%%

C_val = [1]; %% value of C
for nb_c = 1:length(C_val); %% in loop for each C value 
    
    UB = C_val(nb_c).*UB; %% renew upper band based on C value
    alpha = quadprog(H,-f,A,b,Aeq,beq,LB,UB); %% Calcute Alpha with quadprog
    
    alphaY = sum(repmat(alpha.*Y,1,N).*K,1)';
    indx = find(alpha>1e-6); %% alpha thereshoilding
    clc
    alpha %% print alpha
    b = mean(Y(indx)-alphaY(indx)); %% Calcute Bias

    w = sum(repmat(alpha.*Y,1,2).*X,1)'; %% Calcute weights
    
    %% Plotting

    label = {'ko','ks'}; %% label for each data
    label_color = {[0 0 0],[1 1 1]}; %% label Color
    UY = unique(Y); %% unique the labels
    figure();
    indx = find(alpha>1e-6);
    plot(X(indx,1),X(indx,2),'ko','markersize',15,'markerfacecolor',[0.6 0.6 0.6],'markeredgecolor',[0.6 0.6 0.6]);%% plotting all input data
    hold on
    for i = 1:length(UY)
        indx = find(Y==UY(i));
        plot(X(indx,1),X(indx,2),label{i},'markerfacecolor',label_color{i}); %% plotting the each data in separattin mode
    end
    hold on
    f = @(x) -b-w(1)*x; %% creat boundery function
    ezplot(f) %% plot boundry
    xlim([min(X(:,1))-1 max(X(:,1))+1]);
    ylim([min(X(:,2))-1 max(X(:,2))+1]);
end


[Class_Labels,nbErrors,accuracySVM] = SVM_Test(w,b,X,Y)

end
% %{
%% Step 2 -- Classification Function
function [Class_Labels,nbErrors,accuracySVM] = SVM_Test(W,b,X,Y) %% Step 2
    nbData = size(X,1);
    X = X';
    Y = Y';
    for i = 1:nbData
        Class_Labels(i) = sign(W'*X(:,i)+b); %% Classified each data based on decision boundry
    end
    nbErrors = sum(Y~=Class_Labels); %% calcute number of misclassified data 
    accuracySVM = ((nbData - nbErrors)/nbData) * 100; %% calcute accuracy
end

%% Step 3 -- Gram Matrix Function
function Gr = Gram_Mat(A,B,kernel,var)
    if strcmp(kernel,'Linear')        
        Gr = A * B'; %% Gram matrix in Linear mode
    elseif strcmp(kernel,'Gaussian')
        Gr = sqrt(bsxfun(@plus, sum(A' .* A')', bsxfun(@minus, sum(B' .* B'), 2 * A * B'))); % Euclidean distance matrix
        Gr = exp(-(Gr.^2 / (2 * var.^2)));  %% Gram matrix in Gaussian mode
    end
end


%% Step 4 --- Dual problem (Uncomment because Comments are similar to Step1)

function [] = Step4_5()

    X=[1 0; 2 1; 2 -1; -1 0 ; -2 1; -2 -1]; %% input Data
    Y=[1;1;1;-1;-1;-1]; %% input Label

    %%%%%%%%%%%%%%%

    N = size(X,1);%% data Length 

    K = Gram_Mat(X,X,'Linear',1);  %% Calcute Gram matrix with data input and linear kernel and kernel parameter

    H = (Y*Y').*K ; %% Calcute H for primal problem
    f = ones(N,1);
    A = [];
    b = [];
    LB = zeros(N,1);  %% Lower band
    UB = ones(N,1);  %% Higher band

    Aeq = Y';
    beq = 0;
    warning off

    %%%%%%%%%%%%%%%%%%%%%%

    Cval = [1]; %% value of C
    for cv = 1:length(Cval);  %% in loop for each C value 

        UB = Cval(cv).*UB; %% renew upper band based on C value
        alpha = quadprog(H,-f,A,b,Aeq,beq,LB,UB); %% Calcute Alpha with quadprog
        alphaY = alpha.*Y;
        indx = find((alpha > 1e-6).*(alpha < Cval(cv)-1e-6));%% alpha thereshoilding
        btt = Y(indx)' - alphaY'*K(:, indx);
        w = sum(repmat(alpha.*Y,1,2).*X,1)'; %% Calcute weights
        clc
        alpha %% print alpha
        b = mean(btt) %% Calcute Bias

    end
[Class_Labels,nbErrors,accuracySVM] = SVM_Test_Dual(alpha,b,X,X,Y,Y,'Linear',1)    
end


function [Class_Labels,nbErrors,accuracySVM] = SVM_Test_Dual(alpha,b,D_train,D_test,L_train,L_test,kernel,par)

    % train label ???? --> we need train label for calculating w
    %%% we have alpha values, if we select some of them, then the bias
    %%% should change
    % alphaThresh = 1e-6;
    % indx = find(abs(alpha) > alphaThresh);


    nbData_Tr = size(D_train,2);
    K_tr = Gram_Mat(D_train,D_train,kernel,par); %% Gram matrix for train data
    w = sum(repmat(alpha.*L_train,1,nbData_Tr).*D_train,1)'; %% weight of train data

    K_te = Gram_Mat(D_test,D_test,kernel,par);%% Gram matrix for test data
    nbData_te = size(D_test,1);
    Y = L_test;
    for i = 1:nbData_te
        Class_Labels(i) = sign(w'*D_test(i,:)'+b); %% classifing  test data
    end
    nbErrors = sum(Y'~=Class_Labels);
    accuracySVM = ((nbData_te - nbErrors)/nbData_te) * 100;
end


%% Step 6 

function [] = Step6_Primal()
    X0 = [2 2; 2 1; 2 3; 1 2; 3 2 ; -2 -2; -2 -1; -2 -3; -1 -2; -3 -2 ; 2 -2; 2 -1; 2 -3; 1 -2; 3 -2 ; -2 2; -2 1; -2 3; -1 2; -3 2];
    X = [X0(:,1) X0(:,2) X0(:,1).*X0(:,2)];

    for i = 1:20 %% XOR boundy and data label based on their values
        if (X0(i,1)>= 0.5 && X0(i,2)>= 1) || (X0(i,1)>= 1 && X0(i,2)>= 0.5)
            Y(i,1) = -1;
        else
            Y(i,1) = 1;
        end
    end
    %% Plotting input data
    UY = unique(Y);
    ma = {'ko','ks'}; %% label for each data
    fc = {[0 0 0],[1 1 1]}; 
    for i = 1:length(UY)
        indx = find(Y==UY(i));
        scatter3(X(indx,1),X(indx,2),X(indx,3),ma{i},'markerfacecolor',fc{i})
        hold on
    end
    
    hold off

    %% primal Problem with Linear kernel (Comments : Step 1)

    N = size(X,1); %% data Length 
    N2 = size(X,2);
    K = Gram_Mat(X,X,'Linear',1); %% Calcute Gram matrix with data input and linear kernel and kernel parameter
    H = (Y*Y').*K + 1e-5*eye(N); %% Calcute H for primal problem
    f = ones(N,1);
    A = [];
    b = [];
    LB = zeros(N,1); %% Lower band 
    UB = ones(N,1); %% Higher band

    Aeq = Y';
    beq = 0;
    warning off

    %%%%%%%%%%%%%%%%%%%%%%

    Cvals = [1]; %% value of C
    for cv = 1:length(Cvals); %% in loop for each C value

        UB = Cvals(cv).*UB; %% renew upper band based on C value

        alpha = quadprog(H,-f,A,b,Aeq,beq,LB,UB);  %% Calcute Alpha with quadprog

        % Compute the bias
        alphaY = sum(repmat(alpha.*Y,1,N).*K,1)';
        indx = find(alpha>1e-6);
        b = mean(Y(indx)-alphaY(indx)); %% Calcute Bias
        w = sum(repmat(alpha.*Y,1,N2).*X,1)'; %% Calcute weights
    end

    [Class_Labels,nbErrors,accuracySVM] = SVM_Test(w,b,X,Y)

end






function [] = Step6_Dual()

    X0 = [2 2; 2 1; 2 3; 1 2; 3 2 ; -2 -2; -2 -1; -2 -3; -1 -2; -3 -2 ; 2 -2; 2 -1; 2 -3; 1 -2; 3 -2 ; -2 2; -2 1; -2 3; -1 2; -3 2];
    X = [X0(:,1) X0(:,2) X0(:,1).*X0(:,2)];

    for i = 1:20 %% XOR boundy and data label based on their values
        if (X0(i,1)>= 0.5 && X0(i,2)>= 1) || (X0(i,1)>= 1 && X0(i,2)>= 0.5)
            Y(i,1) = -1;
        else
            Y(i,1) = 1;
        end
    end

%%
    UY = unique(Y);
    ma = {'ko','ks'}; %% label for each data
    fc = {[0 0 0],[1 1 1]};

    for i = 1:length(UY)
        indx = find(Y==UY(i));
        scatter3(X(indx,1),X(indx,2),X(indx,3),ma{i},'markerfacecolor',fc{i})
        hold on
    end
    
    hold off

    %%

    N = size(X,1);  %% data Length 
    N2 = size(X,2);

    K = Gram_Mat(X,X,'Gaussian',1); %% Calcute Gram matrix with data input and linear kernel and kernel parameter

    H = (Y*Y').*K ; %% Calcute H for primal problem
    f = ones(N,1);
    A = [];
    b = [];
    LB = zeros(N,1);  %% Lower band
    UB = ones(N,1);  %% Higher band

    Aeq = Y';
    beq = 0;
    warning off

    %%%%%%%%%%%%%%%%%%%%%%

    Cval = [1]; %% value of C
    for cv = 1:length(Cval); %% in loop for each C value 

        UB = Cval(cv).*UB; %% renew upper band based on C value
        alpha = quadprog(H,-f,A,b,Aeq,beq,LB,UB); %% Calcute Alpha with quadprog

        alphaY = alpha.*Y;
        indx = (find((alpha > 1e-4).*(alpha < Cval-1e-6))); %% alpha thereshoilding

    %   indx = find(alpha>1e-6);
        b = mean(alpha(indx)-alphaY(indx));
    %     b = 1 / sqrt(sum(alpha(indx)+(1/Cval)*(alpha'*alpha)));
    w = sum(repmat(alpha.*Y,1,N2).*X,1)'; %% Calcute weights
    end


    [Class_Labels,nbErrors,accuracySVM] = SVM_Test_Dual(alpha,b,X,X,Y,Y,'Linear',1)

    UY = unique(Y);
    ma = {'ko','ks'}; %% label for each data
    fc = {[0 0 0],[1 1 1]}; %% label Color
    figure()

    for i = 1:length(UY)
        indx = find(Class_Labels==UY(i));
        scatter3(X(indx,1),X(indx,2),X(indx,3),ma{i},'markerfacecolor',fc{i})
        hold on
    end

end

%% Step 7 --- Loading Fisher Dataset

function [] = Step7()
    load fisheriris %%Load Fisher dataset
    xmeas = meas;
    xdata = xmeas( 51:end , 3:4); %%select samples and features
    X=xdata; % each comun is a data point
    label = species;
    group = label( 51:end );
    Y=ones(length(group),1); %a column vector
    for i=1:length(group)
        if strcmp(group{i},'virginica')
            Y(i)=-1;
        end
    end
    
    Cval = [1]; %% value of C
%%
    UY = unique(Y);
    ma = {'ko','ks'}; %% label for each data
    fc = {[0 0 0],[1 1 1]};

    for i = 1:length(UY)
        indx = find(Y==UY(i));
        scatter(X(indx,1),X(indx,2),ma{i},'markerfacecolor',fc{i})
        hold on
    end
    
    hold off

    %% Linear evaluation

    sprintf('Linear Evaluation\n')
    N = size(X,1);%% data Length 

    K = Gram_Mat(X,X,'Linear',1);  %% Calcute Gram matrix with data input and linear kernel and kernel parameter

    H = (Y*Y').*(K + (1/Cval)*eye(N)) ; %% Calcute H for primal problem
    f = ones(N,1);
    A = [];
    b = [];
    LB = zeros(N,1);  %% Lower band
    UB = ones(N,1);  %% Higher band

    Aeq = Y';
    beq = 0;
    warning off

    %%%%%%%%%%%%%%%%%%%%%%

    for cv = 1:length(Cval);  %% in loop for each C value 

        UB = Cval(cv).*UB; %% renew upper band based on C value
        alpha = quadprog(H,-f,A,b,Aeq,beq,LB,UB); %% Calcute Alpha with quadprog
        alphaY = alpha.*Y;

        indx = (find((alpha > 1e-4).*(alpha < Cval-1e-6))); %% alpha thereshoilding

    %   indx = find(alpha>1e-6);
        b = mean(alpha(indx)-alphaY(indx));
    %     b = 1 / sqrt(sum(alpha(indx)+(1/Cval)*(alpha'*alpha)));
    w = sum(repmat(alpha.*Y,1,2).*X,1)'; %% Calcute weights

    end
    [Class_Labels,nbErrors,accuracySVM] = SVM_Test_Dual(alpha,b,X,X,Y,Y,'Linear',1)
    
        %% Gaussian evaluation sigma =1

    sprintf('Gaussian Evaluation sigma = 1\n')
    N = size(X,1);%% data Length 

    K = Gram_Mat(X,X,'Gaussian',1);  %% Calcute Gram matrix with data input and linear kernel and kernel parameter

    H = (Y*Y').*(K + (1/Cval)*eye(N)) ; %% Calcute H for primal problem
    f = ones(N,1);
    A = [];
    b = [];
    LB = zeros(N,1);  %% Lower band
    UB = ones(N,1);  %% Higher band

    Aeq = Y';
    beq = 0;
    warning off

    %%%%%%%%%%%%%%%%%%%%%%

    Cval = [1]; %% value of C
    for cv = 1:length(Cval);  %% in loop for each C value 

        UB = Cval(cv).*UB; %% renew upper band based on C value
        alpha = quadprog(H,-f,A,b,Aeq,beq,LB,UB); %% Calcute Alpha with quadprog
        alphaY = alpha.*Y;

        indx = (find((alpha > 1e-4).*(alpha < Cval-1e-6))); %% alpha thereshoilding

    %   indx = find(alpha>1e-6);
        b = mean(alpha(indx)-alphaY(indx));
    %     b = 1 / sqrt(sum(alpha(indx)+(1/Cval)*(alpha'*alpha)));
        w = sum(repmat(alpha.*Y,1,2).*X,1)'; %% Calcute weights

    end
    [Class_Labels,nbErrors,accuracySVM] = SVM_Test_Dual(alpha,b,X,X,Y,Y,'Gaussian',1)
    
        %% Gaussian evaluation sigma =0.1

    sprintf('Gaussian Evaluation sigma = 0.1\n')
    N = size(X,1);%% data Length 

    K = Gram_Mat(X,X,'Gaussian',0.1);  %% Calcute Gram matrix with data input and linear kernel and kernel parameter

    H = (Y*Y').*(K + (1/Cval)*eye(N)) ; %% Calcute H for primal problem
    f = ones(N,1);
    A = [];
    b = [];
    LB = zeros(N,1);  %% Lower band
    UB = ones(N,1);  %% Higher band

    Aeq = Y';
    beq = 0;
    warning off

    %%%%%%%%%%%%%%%%%%%%%%

    Cval = [1]; %% value of C
    for cv = 1:length(Cval);  %% in loop for each C value 

        UB = Cval(cv).*UB; %% renew upper band based on C value
        alpha = quadprog(H,-f,A,b,Aeq,beq,LB,UB); %% Calcute Alpha with quadprog
        alphaY = alpha.*Y;

        indx = (find((alpha > 1e-4).*(alpha < Cval-1e-6))); %% alpha thereshoilding

    %   indx = find(alpha>1e-6);
        b = mean(alpha(indx)-alphaY(indx));
    %     b = 1 / sqrt(sum(alpha(indx)+(1/Cval)*(alpha'*alpha)));
        w = sum(repmat(alpha.*Y,1,2).*X,1)'; %% Calcute weights

    end
    [Class_Labels,nbErrors,accuracySVM] = SVM_Test_Dual(alpha,b,X,X,Y,Y,'Gaussian',1)
end


%% Step 8 --- 

function [] = Step8(Cval)
% %% Train dataset
x = randn(50,2);
y = gaussmf(x , [1 0]);
l1 = ones(50,1);

X_Train = y;
Y_Train = l1;

x = randn(50,2);
y = gaussmf(x , [1 2]);
l2 = -1*ones(50,1);

X_Train = [X_Train ; y];
Y_Train = [Y_Train ; l2];

% Test dataset

x = randn(1000,2);
y = gaussmf(x , [1 0]);
l1 = ones(1000,1);

X_Test = y;
Y_Test = l1;

x = randn(1000,2);
y = gaussmf(x , [1 2]);
l2 = -1*ones(1000,1);

X_Test  = [X_Test  ; y];
Y_Test = [Y_Test ; l2];

      %% Gaussian evaluation sigma = 0.1

     
    sprintf(strcat('Gaussian Evaluation sigma = 1 C = ' , num2str(Cval),' \n'))
    N = size(X_Train,1);%% data Length 

    K = Gram_Mat(X_Train,X_Train,'Gaussian',1);  %% Calcute Gram matrix with data input and linear kernel and kernel parameter

    H = (Y_Train*Y_Train').*(K + (1/Cval)*eye(N)) ; %% Calcute H for primal problem
    f = ones(N,1);
    A = [];
    b = [];
    LB = zeros(N,1);  %% Lower band
    UB = ones(N,1);  %% Higher band

    Aeq = Y_Train';
    beq = 0;
    warning off

    %%%%%%%%%%%%%%%%%%%%%%

    UB = Cval.*UB; %% renew upper band based on C value
    alpha = quadprog(H,-f,A,b,Aeq,beq,LB,UB); %% Calcute Alpha with quadprog
    alphaY = 1 - alpha.*Y_Train;

    indx = (find((alpha > 1e-4).*(alpha < Cval-1e-6))); %% alpha thereshoilding

%   indx = find(alpha>1e-6);
    b = mean(alpha(indx)-alphaY(indx));
%     b = 1 / sqrt(sum(alpha(indx)+(1/Cval)*(alpha'*alpha)));
    w = sum(repmat(alpha.*Y_Train,1,2).*X_Train,1)'; %% Calcute weights

    [Class_Labels,nbErrors,accuracySVM] = SVM_Test_Dual(alpha,b,X_Train,X_Test,Y_Train,Y_Test,'Gaussian',1)
        
%% Plotting

    label = {'ko','ks'}; %% label for each data
    label_color = {[0 0 0],[1 1 1]}; %% label Color
    UY = unique(Y_Train); %% unique the labels
    figure(1);
    ylabel('Training Dataset')
    hold on
    for i = 1:length(UY)
        indx = find(Y_Train==UY(i));
        plot(X_Train(indx,1),X_Train(indx,2),label{i},'markerfacecolor',label_color{i}); %% plotting the each data in separattin mode
    end
    hold on
    f = @(x) -b-w(1)*x; %% creat boundery function
    ezplot(f) %% plot boundry
    
    
    xlim([min(X_Train(:,1))-0.25 max(X_Train(:,1))+0.25]);
    ylim([min(X_Train(:,2))-0.25 max(X_Train(:,2))+0.25]);
    
    figure(2)
    hold on
    ylabel('Testing Dataset')
    label = {'ko','ks'}; %% label for each data
    label_color = {[1 0 0],[0 1 0]}; %% label Color
    UY = unique(Y_Test); %% unique the labels
    for i = 1:length(UY)
        indx = find(Y_Test==UY(i));
        plot(X_Test(indx,1),X_Test(indx,2),label{i},'markerfacecolor',label_color{i}); %% plotting the each data in separattin mode
    end
    
    f = @(x) -b-w(1)*x; %% creat boundery function
    ezplot(f) %% plot boundry
    
    xlim([min(X_Train(:,1))-0.25 max(X_Train(:,1))+0.25]);
    ylim([min(X_Train(:,2))-0.25 max(X_Train(:,2))+0.25]);
end


%% Step 9 --- 

function [] = Step9(sigma)
% %% Train dataset
x = randn(50,2);
y = gaussmf(x , [1 0]);
l1 = ones(50,1);

X_Train = y;
Y_Train = l1;

x = randn(50,2);
y = gaussmf(x , [1 2]);
l2 = -1*ones(50,1);

X_Train = [X_Train ; y];
Y_Train = [Y_Train ; l2];

% Test dataset

x = randn(1000,2);
y = gaussmf(x , [1 0]);
l1 = ones(1000,1);

X_Test = y;
Y_Test = l1;

x = randn(1000,2);
y = gaussmf(x , [1 2]);
l2 = -1*ones(1000,1);

X_Test  = [X_Test  ; y];
Y_Test = [Y_Test ; l2];

      %% Gaussian evaluation C = 1

     
    Cval = 1;
    sprintf(strcat('Gaussian Evaluation C = 1 sigma = ' , num2str(sigma),' \n'))
    N = size(X_Train,1);%% data Length 

    K = Gram_Mat(X_Train,X_Train,'Gaussian',sigma);  %% Calcute Gram matrix with data input and linear kernel and kernel parameter

    H = (Y_Train*Y_Train').*(K + (1/Cval)*eye(N)) ; %% Calcute H for primal problem
    f = ones(N,1);
    A = [];
    b = [];
    LB = zeros(N,1);  %% Lower band
    UB = ones(N,1);  %% Higher band

    Aeq = Y_Train';
    beq = 0;
    warning off

    %%%%%%%%%%%%%%%%%%%%%%

    UB = Cval.*UB; %% renew upper band based on C value
    alpha = quadprog(H,-f,A,b,Aeq,beq,LB,UB); %% Calcute Alpha with quadprog
    alphaY = 1- alpha.*Y_Train;
    indx = find(alpha>1e-6);
    b = mean(alpha(indx)-alphaY(indx));
    w = sum(repmat(alpha.*Y_Train,1,2).*X_Train,1)'; %% Calcute weights

    [Class_Labels,nbErrors,accuracySVM] = SVM_Test_Dual(alpha,b,X_Train,X_Test,Y_Train,Y_Test,'Gaussian',1)
    
%% Plotting

    label = {'ko','ks'}; %% label for each data
    label_color = {[0 0 0],[1 1 1]}; %% label Color
    UY = unique(Y_Train); %% unique the labels
    figure(1);
    ylabel('Training Dataset')
    hold on
    for i = 1:length(UY)
        indx = find(Y_Train==UY(i));
        plot(X_Train(indx,1),X_Train(indx,2),label{i},'markerfacecolor',label_color{i}); %% plotting the each data in separattin mode
    end
    hold on
    f = @(x) -b-w(1)*x; %% creat boundery function
    ezplot(f) %% plot boundry
    
    
    xlim([min(X_Train(:,1))-0.25 max(X_Train(:,1))+0.25]);
    ylim([min(X_Train(:,2))-0.25 max(X_Train(:,2))+0.25]);
    
    figure(2)
    hold on
    ylabel('Testing Dataset')
    label = {'ko','ks'}; %% label for each data
    label_color = {[1 0 0],[0 1 0]}; %% label Color
    UY = unique(Y_Test); %% unique the labels
    for i = 1:length(UY)
        indx = find(Y_Test==UY(i));
        plot(X_Test(indx,1),X_Test(indx,2),label{i},'markerfacecolor',label_color{i}); %% plotting the each data in separattin mode
    end
    
    f = @(x) -b-w(1)*x; %% creat boundery function
    ezplot(f) %% plot boundry
    
    xlim([min(X_Train(:,1))-0.25 max(X_Train(:,1))+0.25]);
    ylim([min(X_Train(:,2))-0.25 max(X_Train(:,2))+0.25]);
end
% %}