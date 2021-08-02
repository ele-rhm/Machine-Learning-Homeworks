%% This program was written by Elahe Rahimi%%
clc;
clear all;
close all;
%% Step 1:
load('GradBoostTrainDATA.mat','PTrain','NTrain','x1');
figure,
axis([-5 10 -5 10])
plot(NTrain(:,1),NTrain(:,2),'r*')
hold on
axis([-5 10 -5 10])
plot(PTrain(:,1),PTrain(:,2),'bo')
x1=[PTrain ; NTrain];
y1=[ones(500,1) ; -1*ones(500,1)];
%% Step 2:
alpha=1;
T=4;
MAX1=max(x1(:,1));
MAX2=max(x1(:,2));

MIN1=min(x1(:,1));
MIN2=min(x1(:,2));

TS1=(MAX1-MIN1)/100;  %threshold step 1
TS2=(MAX2-MIN2)/100;   % threshold step 2

%% Step 3:
fi=zeros(1000,1);
winit=exp(0);
w=winit*ones(1,size(x1,1)); %initial weight
% 
% %% Step 4:
% % first dimension
% k=1;
% Threshold = MIN1;
% while Threshold <= MAX1
%     for i=1:size(x1,1)
%         if x1(i,1) <= Threshold
%             G1(i,1)=-1;
%         else
%             G1(i,1)=1;
%         end
%     end  
%     %Parity to right
%     EXPLOSS1(k,1)= sum(y1(:,1).*w'.*G1(:,1));
%     EXPLOSS1(k,2)= Threshold;
%     EXPLOSS1(k,3)= 1; %right
%     k=k+1;
%     
%     clear G1
%     for i=1:size(x1,1)
%         if x1(i,1) > Threshold
%             G1(i,1)=-1;
%         else
%             G1(i,1)=1;
%         end
%     end       
%     %parity to left
%     EXPLOSS1(k,1)= sum(y1(:,1).*w'.*G1(:,1));
%     EXPLOSS1(k,2)= Threshold;
%     EXPLOSS1(k,3)= -1; %left
%     k=k+1;
%     
%     Threshold=Threshold+TS1;
% end
% 
% clear G1
% % second dimension
% k=1;
% Threshold = MIN2;
% while Threshold <= MAX2
%     for i=1:size(x1,1)
%         if x1(i,2) <= Threshold
%             G1(i,1)=-1;
%         else
%             G1(i,1)=1;
%         end
%     end  
%     %Parity to right
%     EXPLOSS2(k,1)= sum(y1(:,1).*w'.*G1(:,1));
%     EXPLOSS2(k,2)= Threshold;
%     EXPLOSS2(k,3)= 1; %right
%     k=k+1;
%     
%     clear G1
%     for i=1:size(x1,1)
%         if x1(i,2) > Threshold
%             G1(i,1)=-1;
%         else
%             G1(i,1)=1;
%         end
%     end       
%     %parity to left
%     EXPLOSS2(k,1)= sum(y1(:,1).*w'.*G1(:,1));
%     EXPLOSS2(k,2)= Threshold;
%     EXPLOSS2(k,3)= -1; %left
%     k=k+1;
%     
%     Threshold=Threshold+TS2;
% end
% 
% [~,m1]=max(EXPLOSS1(:,1));
% [~,m2]=max(EXPLOSS2(:,1));
% M1=EXPLOSS1(m1,:);
% M2=EXPLOSS2(m2,:);
% 
% [R V]=max(EXPLOSS2(:,1))
% [l z]=max(EXPLOSS1(:,1))
% 
% g=zeros(1000,1);
% thre=0.99
% [r c v]=find(x1(:,2)>thre);
% g(r)=-1;
% [r c v]=find(x1(:,2)<=thre);
% g(r)=+1;
% %% Step 5:
% 
% fi=fi+alpha*g;
%     
%     w=exp(-y1.*fi); % update weights 
% %% Step 6:
% wPLC=find(w==0);
% w(wPLC)=realmin;
% wPLC=find(w==inf);
% w(wPLC)=realmax;
% CurrentRISK=sum(w);

%% Step 7:
        gstar=zeros(4,4);

w=w';
for E=1:4
k=1;
Threshold = MIN1;
while Threshold <= MAX1
    for i=1:size(x1,1)
        if x1(i,1) <= Threshold
            G1(i,1)=-1;
        else
            G1(i,1)=1;
        end
    end  
    %Parity to right
    EXPLOSS1(k,1)= sum(y1(:,1).*w.*G1(:,1));
    EXPLOSS1(k,2)= Threshold;
    EXPLOSS1(k,3)= 1; %right
    k=k+1;
    
    clear G1
    for i=1:size(x1,1)
        if x1(i,1) > Threshold
            G1(i,1)=-1;
        else
            G1(i,1)=1;
        end
    end       
    %parity to left
    EXPLOSS1(k,1)= sum(y1(:,1).*w.*G1(:,1));
    EXPLOSS1(k,2)= Threshold;
    EXPLOSS1(k,3)= -1; %left
    k=k+1;
    
    Threshold=Threshold+TS1;
end

clear G1
% second dimension
k=1;
Threshold = MIN2;
while Threshold <= MAX2
    for i=1:size(x1,1)
        if x1(i,2) <= Threshold
            G1(i,1)=-1;
        else
            G1(i,1)=1;
        end
    end  
    %Parity to right
    EXPLOSS2(k,1)= sum(y1(:,1).*w.*G1(:,1));
    EXPLOSS2(k,2)= Threshold;
    EXPLOSS2(k,3)= 1; %right
    k=k+1;
    
    clear G1
    for i=1:size(x1,1)
        if x1(i,2) > Threshold
            G1(i,1)=-1;
        else
            G1(i,1)=1;
        end
    end       
    %parity to left
    EXPLOSS2(k,1)= sum(y1(:,1).*w.*G1(:,1));
    EXPLOSS2(k,2)= Threshold;
    EXPLOSS2(k,3)= -1; %left
    k=k+1;
    
    Threshold=Threshold+TS2;
end

[v11,m11]=max(EXPLOSS1(:,1));
[v22,m22]=max(EXPLOSS2(:,1));
g=zeros(1000,1);

if v11>v22
   m1=m11
else
    m1=m22
end
    if m1==m11
        [r c ~]=find(x1(:,1)>EXPLOSS1(m1,2));
       
        g(r)=1;
         [r c ~]=find(x1(:,1)<EXPLOSS1(m1,2));
        g(r)=-1;
        g=EXPLOSS1(m1,3)*g;
gstar(E,1:3)=EXPLOSS1(m1,:);
gstar(E,4)=1;

    else
         %mire too dim 2
         [r c ~]=find(x1(:,2)>EXPLOSS2(m1,2));
        g(r)=1;
         [r c ~]=find(x1(:,2)<EXPLOSS2(m1,2));
        g(r)=-1;
        g=EXPLOSS2(m1,3)*g;
gstar(E,1:3)=EXPLOSS2(m1,:);
gstar(E,4)=2

    end
    
fi=fi+alpha*g;
 w=exp(-y1.*fi); % update weights 

wPLC=find(w==0);
w(wPLC)=realmin;
wPLC=find(w==inf);
w(wPLC)=realmax;
CurrentRISK=sum(w);
C(E)=CurrentRISK;
end

figure
stem(C,'fill') 
title('Exponential Risk at each iteration')
axis([0 5 0 800])

%% Step 8:
load('GradBoostTestDATA.mat','PTest','NTest','x1'); % load test data
figure,
axis([-5 10 -5 10])
plot(NTest(:,1),NTest(:,2),'r*')
hold on
axis([-5 10 -5 10])
plot(PTest(:,1),PTest(:,2),'bo')
x1=[PTest ; NTest];

%% Step 9:
% 
  g=nan(2000,1);
 fi=zeros(2000,1);
  
for i =1:4 
    thr=gstar(i,2);
    if gstar(i,4)==1  % Dimension 1
    [r,~,~]=find(x1(:,1)>thr);
    g(r)=1;
    [r,~,~]=find(x1(:,1)<thr);
    g(r)=-1;
    g=g*gstar(i,3); %  Conidering parity
    fi=fi+g;

    end
        if gstar(i,4)==2 % Dimension 2
    [r,~,~]=find(x1(:,2)>thr);
    g(r)=1;
    [r,~,~]=find(x1(:,2)<thr);
    g(r)=-1;
    g=g*gstar(i,3); 
    fi=fi+g;

        end
       
end


[r,~,~]=find(fi==0);
fi(r)=-1;
y=[ones(1000,1);-ones(1000,1)];

 fi=sign(fi);
a=fi~=y;
Error9= sum(a)
 
for i=1:2000
    if fi(i)<=0
        plot(x1(i,1),x1(i,2),'r*')
        hold on
    else if fi>0 
        plot(x1(i,1),x1(i,2),'bo')
        hold on
        end
    end
end


%% Step 10:

load('GradBoostTrainDATA.mat','PTrain','NTrain','x1');
figure,
axis([-5 10 -5 10])
plot(NTrain(:,1),NTrain(:,2),'r*')
hold on
axis([-5 10 -5 10])
plot(PTrain(:,1),PTrain(:,2),'bo')
x1=[PTrain ; NTrain];
y1=[ones(500,1) ; -1*ones(500,1)];

alpha=1;

MAX1=max(x1(:,1));
MAX2=max(x1(:,2));

MIN1=min(x1(:,1));
MIN2=min(x1(:,2));

TS1=(MAX1-MIN1)/100;  %threshold step 1
TS2=(MAX2-MIN2)/100;   % threshold step 2

fi=zeros(1000,1);
winit=exp(0);
w=winit*ones(1,size(x1,1)); %initial weight
% 
% %% Step 4:
% % first dimension
% k=1;
% Threshold = MIN1;
% while Threshold <= MAX1
%     for i=1:size(x1,1)
%         if x1(i,1) <= Threshold
%             G1(i,1)=-1;
%         else
%             G1(i,1)=1;
%         end
%     end  
%     %Parity to right
%     EXPLOSS1(k,1)= sum(y1(:,1).*w'.*G1(:,1));
%     EXPLOSS1(k,2)= Threshold;
%     EXPLOSS1(k,3)= 1; %right
%     k=k+1;
%     
%     clear G1
%     for i=1:size(x1,1)
%         if x1(i,1) > Threshold
%             G1(i,1)=-1;
%         else
%             G1(i,1)=1;
%         end
%     end       
%     %parity to left
%     EXPLOSS1(k,1)= sum(y1(:,1).*w'.*G1(:,1));
%     EXPLOSS1(k,2)= Threshold;
%     EXPLOSS1(k,3)= -1; %left
%     k=k+1;
%     
%     Threshold=Threshold+TS1;
% end
% 
% clear G1
% % second dimension
% k=1;
% Threshold = MIN2;
% while Threshold <= MAX2
%     for i=1:size(x1,1)
%         if x1(i,2) <= Threshold
%             G1(i,1)=-1;
%         else
%             G1(i,1)=1;
%         end
%     end  
%     %Parity to right
%     EXPLOSS2(k,1)= sum(y1(:,1).*w'.*G1(:,1));
%     EXPLOSS2(k,2)= Threshold;
%     EXPLOSS2(k,3)= 1; %right
%     k=k+1;
%     
%     clear G1
%     for i=1:size(x1,1)
%         if x1(i,2) > Threshold
%             G1(i,1)=-1;
%         else
%             G1(i,1)=1;
%         end
%     end       
%     %parity to left
%     EXPLOSS2(k,1)= sum(y1(:,1).*w'.*G1(:,1));
%     EXPLOSS2(k,2)= Threshold;
%     EXPLOSS2(k,3)= -1; %left
%     k=k+1;
%     
%     Threshold=Threshold+TS2;
% end
% 
% [~,m1]=max(EXPLOSS1(:,1));
% [~,m2]=max(EXPLOSS2(:,1));
% [R V]=max(EXPLOSS2(:,1))
% [l z]=max(EXPLOSS1(:,1))
% 
% g=zeros(1000,1);
% thre=0.99
% [r c v]=find(x1(:,2)>thre);
% g(r)=-1;
% [r c v]=find(x1(:,2)<=thre);
% g(r)=+1;
% %% Step 5:
% 
% fi=fi+alpha*g;
%     
%     w=exp(-y1.*fi); % update weights 
% %% Step 6:
% wPLC=find(w==0);
% w(wPLC)=realmin;
% wPLC=find(w==inf);
% w(wPLC)=realmax;
% CurrentRISK=sum(w);


        gstar=zeros(4,4);

w=w';
for E=1:4
k=1;
Threshold = MIN1;
while Threshold <= MAX1
    for i=1:size(x1,1)
        if x1(i,1) <= Threshold
            G1(i,1)=-1;
        else
            G1(i,1)=1;
        end
    end  
    %Parity to right
    EXPLOSS1(k,1)= sum(y1(:,1).*w.*G1(:,1));
    EXPLOSS1(k,2)= Threshold;
    EXPLOSS1(k,3)= 1; %right
    k=k+1;
    
    clear G1
    for i=1:size(x1,1)
        if x1(i,1) > Threshold
            G1(i,1)=-1;
        else
            G1(i,1)=1;
        end
    end       
    %parity to left
    EXPLOSS1(k,1)= sum(y1(:,1).*w.*G1(:,1));
    EXPLOSS1(k,2)= Threshold;
    EXPLOSS1(k,3)= -1; %left
    k=k+1;
    
    Threshold=Threshold+TS1;
end

clear G1
% second dimension
k=1;
Threshold = MIN2;
while Threshold <= MAX2
    for i=1:size(x1,1)
        if x1(i,2) <= Threshold
            G1(i,1)=-1;
        else
            G1(i,1)=1;
        end
    end  
    %Parity to right
    EXPLOSS2(k,1)= sum(y1(:,1).*w.*G1(:,1));
    EXPLOSS2(k,2)= Threshold;
    EXPLOSS2(k,3)= 1; %right
    k=k+1;
    
    clear G1
    for i=1:size(x1,1)
        if x1(i,2) > Threshold
            G1(i,1)=-1;
        else
            G1(i,1)=1;
        end
    end       
    %parity to left
    EXPLOSS2(k,1)= sum(y1(:,1).*w.*G1(:,1));
    EXPLOSS2(k,2)= Threshold;
    EXPLOSS2(k,3)= -1; %left
    k=k+1;
    
    Threshold=Threshold+TS2;
end

[v11,m11]=max(EXPLOSS1(:,1));
[v22,m22]=max(EXPLOSS2(:,1));
g=zeros(1000,1);

if v11>v22
   m1=m11
else
    m1=m22
end
    if m1==m11
        [r c ~]=find(x1(:,1)>EXPLOSS1(m1,2));
       
        g(r)=1;
         [r c ~]=find(x1(:,1)<EXPLOSS1(m1,2));
        g(r)=-1;
        g=EXPLOSS1(m1,3)*g;
gstar(E,1:3)=EXPLOSS1(m1,:);
gstar(E,4)=1;

    else
         %mire too dim 2
         [r c ~]=find(x1(:,2)>EXPLOSS2(m1,2));
        g(r)=1;
         [r c ~]=find(x1(:,2)<EXPLOSS2(m1,2));
        g(r)=-1;
        g=EXPLOSS2(m1,3)*g;
gstar(E,1:3)=EXPLOSS2(m1,:);
gstar(E,4)=2

    end
    
fi=fi+alpha*g;
w=1/2*log10(1+exp(-(2)*y1.*fi)); % update weights
wPLC=find(w==0);
w(wPLC)=realmin;
wPLC=find(w==inf);
w(wPLC)=realmax;
CurrentRISK(E)=sum(w);
end


load('GradBoostTestDATA.mat','PTest','NTest','x1'); % load test data
figure,
axis([-5 10 -5 10])
plot(NTest(:,1),NTest(:,2),'r*')
hold on
axis([-5 10 -5 10])
plot(PTest(:,1),PTest(:,2),'bo')
x1=[PTest ; NTest];


% 
  g=nan(2000,1);
 fi=zeros(2000,1);
  
for i =1:4 
    thr=gstar(i,2);
    if gstar(i,4)==1  % Dimension 1
    [r,~,~]=find(x1(:,1)>thr);
    g(r)=1;
    [r,~,~]=find(x1(:,1)<thr);
    g(r)=-1;
    g=g*gstar(i,3); %  Conidering parity
    fi=fi+g;

    end
        if gstar(i,4)==2 % Dimension 2
    [r,~,~]=find(x1(:,2)>thr);
    g(r)=1;
    [r,~,~]=find(x1(:,2)<thr);
    g(r)=-1;
    g=g*gstar(i,3); 
    fi=fi+g;

        end
       
end


[r,~,~]=find(fi==0);
fi(r)=-1;
y=[ones(1000,1);-ones(1000,1)];

 fi=sign(fi);
a=fi~=y;
Error10= sum(a)
 
for i=1:2000
    if fi(i)<=0
        plot(x1(i,1),x1(i,2),'r*')
        hold on
    else if fi>0 
        plot(x1(i,1),x1(i,2),'bo')
        hold on
        end
    end
end

