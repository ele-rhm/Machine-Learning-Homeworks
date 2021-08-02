%%Written By Elahe Rahimi.%%
clc;
close all;
clear all;
%% Step 1:

load('TrainDATAtoyGaussian1D.mat','x1');
%Initialization:
mus= [0,3];
sigmas=[0.5,1];
PIs=[0.5 0.5];
C=2;
epsilon=2;
Counter=0;

while epsilon>0.01
    for j=1:C
      A(:,j)=mvnpdf(x1,mus(j),diag(sigmas(j))).*PIs(j);

    end
    
   B=repmat(sum(A,2),1,2); 
      
   h=A./B; 
   Qold=sum(sum(h.*log(A)));
   for j=1:C
       mus(j)=sum(h(:,j).*x1)./sum(h(:,j)); 
       sigmas(j)=sum(h(:,j).*(x1-repmat(mus(j),1000,1)).*(x1-repmat(mus(j),1000,1)))./sum(h(:,j));
       PIs=sum(h)/1000;
   end
   Counter=Counter+1;
   for j=1:C
       A(:,j)=mvnpdf(x1,mus(j),diag(sigmas(j))).*PIs(j);
   Qnew=sum(sum(h.*log(A)));
epsilon=Qnew-Qold;
   end
   end

%% Part 2:
clear all;
load('TrainDATAtoyGaussian2D.mat','x1');
% Initial Values:
mus= [0 3 ; 0 0];
sigmas=[0.4 0.8; 0.6 0.7];
mus= cat(3,mus(:,1)', mus(:,2)');
sigmas=cat(3,sigmas(:,1)',sigmas(:,2)'); 
PIs=[0.5 0.5];
epsilon=10;
Counter=0;
C=2;
while epsilon>0.01
    for j=1:C
      A(:,j)=mvnpdf(x1,mus(:,:,j),diag(sigmas(:,:,j))).*PIs(j);

    end
    
   B=repmat(sum(A,2),1,2); 
      
   h=A./B; 
   Qold=sum(sum(h.*log(A)));
   for j=1:C
       mus(:,:,j)=(sum(repmat(h(:,j),1,2).*x1))./sum(h(:,j));  

       sigmas(:,:,j)=sum(repmat(h(:,j),1,2).*(x1-repmat(mus(:,:,j),1000,1)).*(x1-repmat(mus(:,:,j),1000,1)))./sum(h(:,j));
       PIs=sum(h)/1000;
   end
   Counter=Counter+1;
   for j=1:C
       A(:,j)=mvnpdf(x1,mus(:,:,j),diag(sigmas(:,:,j))).*PIs(j);
   Qnew=sum(sum(h.*log(A)));
epsilon=Qnew-Qold;
   end
   end
mus=[mus(:,:,1), mus(:,:,2)];
sigmas=[sigmas(:,:,1), sigmas(:,:,2)];
PIs=[PIs(1) PIs(2)];
%% Part 3:
clear all;
load('TrainingSamplesDCT_8_new');
x1=TrainsampleDCT_BG;
load('INITValues_BackGround.mat','muinit','varinit','PIS');
C=4;
epsilon=2;
Counter=0;
mus_bg= cat(3,muinit(:,1)',muinit(:,2)',muinit(:,3)',muinit(:,4)');
sigmas_bg=cat(3,varinit(:,1)',varinit(:,2)',varinit(:,3)',varinit(:,4)');
PIs_bg=PIS;


while epsilon>0.01
    for j=1:C
      A(:,j)=mvnpdf(x1,mus_bg(:,:,j),diag(sigmas_bg(:,:,j))).*PIs_bg(j);

    end
    
   B=repmat(sum(A,2),1,4);
      
   h=A./B; 
   Qold=sum(sum(h.*log(A)));
   for j=1:C
       mus_bg(:,:,j)=(sum(repmat(h(:,j),1,64).*x1))./sum(h(:,j));  
       sigmas_bg(:,:,j)=sum(repmat(h(:,j),1,64).*(x1-repmat(mus_bg(:,:,j),length(x1),1)).*(x1-repmat(mus_bg(:,:,j),length(x1),1)))./sum(h(:,j));
       PIs_bg=sum(h)/length(x1);
   end
   Counter=Counter+1;
   for j=1:C
       A(:,j)=mvnpdf(x1,mus_bg(:,:,j),diag(sigmas_bg(:,:,j))).*PIs_bg(j);
   end
   Qnew=sum(sum(h.*log(A)));
epsilon=Qnew-Qold;
   end

   save Background mus_bg sigmas_bg PIs_bg
mus_bg=[mus_bg(:,:,1)', mus_bg(:,:,2)',mus_bg(:,:,3)',mus_bg(:,:,4)'];
disp(mus_bg)
sigmas_bg=[sigmas_bg(:,:,1)', sigmas_bg(:,:,2)',sigmas_bg(:,:,3)',sigmas_bg(:,:,4)'];


%% Part4 :
clear all;
load('TrainingSamplesDCT_8_new');
x1=TrainsampleDCT_FG;
load('INITValues_ForeGround.mat','muinit','varinit','PIS');
mus_fg= cat(3,muinit(:,1)',muinit(:,2)',muinit(:,3)',muinit(:,4)');
sigmas_fg=cat(3,varinit(:,1)',varinit(:,2)',varinit(:,3)',varinit(:,4)');
PIs_fg=PIS;


epsilon=2;
Counter=0;
C=4;
while epsilon > .01

    for j=1:C
        A(:,j)=mvnpdf(x1,mus_fg(:,:,j),diag(sigmas_fg(:,:,j)))*PIs_fg(j);
    end
    h=A./repmat(sum(A,2),1,4);
    Qold=sum(sum(h.*log(A)));

    for j=1:C
        mus_fg(:,:,j)=(sum(repmat(h(:,j),1,64).*x1))./sum(h(:,j));
        PIs_fg=sum(h)/250;
        sigmas_fg(:,:,j)=sum(repmat(h(:,j),1,64).*...
            (x1-repmat(mus_fg(:,:,j),250,1)).*...
            (x1-repmat(mus_fg(:,:,j),250,1)))./sum(h(:,j));
    end
    Counter=Counter+1;
    for j=1:4
        A(:,j)=mvnpdf(x1,mus_fg(:,:,j),diag(sigmas_fg(:,:,j)))*PIs_fg(j);
    end
    Qnew=sum(sum(h.*log(A)));
    epsilon=Qnew-Qold;
end

save Foreground mus_fg sigmas_fg PIs_fg

mus_fg=[mus_fg(:,:,1)',mus_fg(:,:,2)',mus_fg(:,:,3)',mus_fg(:,:,4)'];
sigmas_fg=[sigmas_fg(:,:,1)',sigmas_fg(:,:,2)',sigmas_fg(:,:,3)',sigmas_fg(:,:,4)'];


%% Part5 :
I=im2double(imread('cheetah.bmp'));
Imask=im2double(imread('cheetah_mask.bmp')); 
figure;
imshow(Imask);
imshow(I);title('cheetah')

zigzag=load('zig-zag pattern.txt');
[val, plc]=sort(zigzag(:));
load('Foreground.mat')
load('Background.mat')


P=zeros(1,4);
I2=nan(size(I));
C=4;
[r c]=size(I);

for i=4:r-4
    for j=4:c-4
        dctblock = dct2(I(i-3:i+4,j-3:j+4));
        dctblock = dctblock(plc);
        for t=1:C
            P(t)=mvnpdf(dctblock',mus_bg(:,:,t),diag(sigmas_bg(:,:,t)))*PIs_bg(t);
        end
        Prob_BG=sum(P);
        for t=1:C
            P(t)=mvnpdf(dctblock',mus_fg(:,:,t),diag(sigmas_fg(:,:,t)))*PIs_fg(t);
        end
        Prob_FG=sum(P);

        if Prob_FG*250>Prob_BG*1053
            I2(i,j)=1;
            else
            I2(i,j)=0;
        end
    end
end
figure(3);
imshow(I2); title('cheeta after using EM algorithm')

