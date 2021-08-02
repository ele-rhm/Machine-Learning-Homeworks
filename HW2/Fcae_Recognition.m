%%Written By Elahe Rahimi%%
clc;
clear all;
close all;
%% Step 1:
I=imread(['orl_faces\s' int2str(1) '\' int2str(1) '.pgm' ]);
I=im2double(I);
figure, imshow(I);
[r c]=size(I);
disp('How many rows (r) and columns (c) does the matrix I have?')
disp([r c])

TrainData=imread(['orl_faces\s' int2str(1) '\' int2str(1) '.pgm' ]);
TrainData=im2double(TrainData);

%% Step 2:
TI=5; % number of images per person

for i=1:40
    for j=1:5;
    TrainData=imread(['orl_faces\s' int2str(i) '\' int2str(j) '.pgm']);
    [r,c]=size(TrainData);
      TrainData=im2double(TrainData);
TrainData=reshape(TrainData,r*c,1);

    TR(:,TI*(i-1)+j)=TrainData;  

    end
end
    save TR
   

MEAN=mean(TR);
ZTR = bsxfun(@minus,TR,MEAN);


%% Step 3:
TI=5;  % number of images per person

for k=1:40
    for l=1:5;
    TestData=imread(['C:/orl_faces\s' int2str(k) '\' int2str(l+5) '.pgm']);
   [ u,v]=size(TestData);
      TestData=im2double(TestData);
TestData=reshape(TestData,u*v,1);
    TS(:,TI*(k-1)+l)=TestData;  
    end
end
save TS;
 
MEAN_S=mean(TS);
ZTE= bsxfun(@minus,TS,MEAN_S);
%% Step 4:
d=zeros(200,200);
for ii=1:200
  for  jj=1:200
    temp=TS(:,ii)-TR(:,jj);
   
 d(ii,jj)= norm(temp);
  end
end
 [v ind] = min(d,[],2);
 counter=0;
 for i=1:200
    if ceil(ind(i)/5)~= ceil(i/5);
        counter=counter+1 ;  
    end

 end
disp('how many faces in the test data set are recognized correctly and how many are not recognized correctly? 20 & 180');

  %% Step 5:
  
 [U S V]=svd(ZTR);
 
 %% Step 6:
 
figure;  semilogy(diag(S), '.');
MaxS=max(diag(S));
MinS=min(diag(S));
disp('What do you think is a good point for truncating the SVD? 45')

%% Step 7:

figure;
  for i=1:10;
    subplot(2,5,i)
imshow( reshape(U(:,i),r,c),[] )
  end

      figure;
for i=1:10
      subplot(2,5,i)
imshow( reshape(U(:,200-i),r,c),[] )
end
%% Step 8:

UKEEP=U(:,1:60);
PROJD=(ZTR'*UKEEP)';

%% Step 9:

PROJT=(ZTE'*UKEEP)';

%% Step :

for h=1:200
    for f=1:200
diff=(PROJT(:,h)-PROJD(:,f));
mm(h,f)=norm(diff);
    end 
end

 [w ind2] = min(mm,[],2);
 counter2=0;
 for i=1:200
    if ceil(ind2(i)/5)~= ceil(i/5);
        counter2=counter2+1 ;  
    end

 end
disp('How many faces are not recognized correctly?')
 disp(counter);
 %% Step 11:
 
 mm1=zeros(200,200);
for nk=1:200
    UKEEP=U(:,1:nk);
PROJD=(ZTR'*UKEEP)';
PROJT=(ZTE'*UKEEP)';
 for h1=1:200
    for f1=1:200
diff=(PROJD(:,f1)-PROJT(:,h1));
mm1(h1,f1)=norm(diff);
    end 
 end
 [y ind3] = min(mm1,[],2);

 counter3(nk)=0;
 for t=1:200
    if ceil(ind3(t)/5)~= ceil(t/5);
        counter3(nk)=counter3(nk)+1 ;  
    end
 end

end    
[val nk]=min(counter3);

figure;
plot(counter3);
disp('What is the best NK for your test data set? nk=53')
disp('How much memory do you save with this NK compared to METHOD-1? (10304*10304)-(10304*45)=1597120');




