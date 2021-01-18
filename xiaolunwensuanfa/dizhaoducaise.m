%% ���նȲ�ɫͼ��
close all;clear ;clc;
%% 
tic;
%       img=imread('E:\picture\huaping.bmp','bmp');
       img=imread('E:\picture\friut.bmp','bmp');
%     img=imread('E:\picture\he.bmp','bmp');
%     img=imread('E:\picture\sea.bmp','bmp');
%      img=imread('E:\picture\cat.bmp','bmp');



img=mat2gray(img);  %��������ӳ�䵽[0,1];
[m,n,dim]=size(img);
% imshow(img);
%%ͼ���RGB
R=img(:,:,1);
G=img(:,:,2);
B=img(:,:,3);

%%ͼ���RGB2HSV
H=zeros(m,n);   %ɫ���
S=zeros(m,n);   %���Ͷ�
V=zeros(m,n);   %����
for i=1:m
   for j=1:n
       r=R(i,j);
       g=G(i,j);
       b=B(i,j);
       MAX=max([r,g,b]);
       MIN=min([r,g,b]);

       if MAX==MIN
            H(i,j)=0;
       elseif MAX==r && g>=b
            H(i,j)=60*(g-b)/(MAX-MIN);
       elseif MAX==r && g<b
            H(i,j)=60*(g-b)/(MAX-MIN)+360;
       elseif MAX==g
            H(i,j)=60*(b-r)/(MAX-MIN)+120;
       elseif MAX==b
            H(i,j)=60*(r-g)/(MAX-MIN)+240;
       end

       if MAX==0
            S(i,j)=0;
       else
            S(i,j)=1-MIN/MAX;
       end

       V(i,j)=MAX;
   end
end
figure;
imshow(img);title('ԭʼͼ��'); 

figure;
subplot(1,3,1);imshow(H);title('HSV�ռ�H����ͼ��');
subplot(1,3,2);imshow(S);title('HSV�ռ�S����ͼ��');
subplot(1,3,3);imshow(V);title('HSV�ռ�V����ͼ��');


% gamma�任�����Ҷ�
f = mat2gray(V);
gamma1 = 0.1;
g_0_1 = f.^gamma1; 
gamma2 = 0.2;
g_0_2 = f.^gamma2; 
gamma3 = 0.4;
g_0_4 = f.^gamma3;
gamma4 = 0.6;
g_0_6 = f.^gamma4;
gamma5 = 0.8;
g_0_8 = f.^gamma5; 
gamma6 = 1;
g_1 = f.^gamma6;  
gamma7 = 2.5;
g_2_5 = f.^gamma7; 
gamma8 = 5;
g_5 = f.^gamma8;   
figure();  
subplot(3,3,1);  
imshow(f,[0 1]);  
xlabel('a) Original Image');  
subplot(3,3,2);  
imshow(g_0_1,[0 1]);  
xlabel('b) \gamma =0.1'); 
subplot(3,3,3);  
imshow(g_0_2,[0 1]);  
xlabel('c) \gamma =0.2'); 
subplot(3,3,4);  
imshow(g_0_4,[0 1]);  
xlabel('d) \gamma=0.4'); 
subplot(3,3,5);  
imshow(g_0_6,[0 1]);  
xlabel('e) \gamma=0.6'); 
subplot(3,3,6);  
imshow(g_0_8,[0 1]);  
xlabel('f) \gamma=0.8'); 
subplot(3,3,7);  
imshow(g_1,[0 1]);  
xlabel('g) \gamma=1'); 
subplot(3,3,8);  
imshow(g_2_5,[0 1]);  
xlabel('h) \gamma=2.5'); 
subplot(3,3,9);  
imshow(g_5,[0 1]);  
xlabel('j) \gamma=5'); 

figure;
subplot(1,2,1);
imhist(f);
title('v����ֱ��ͼ');
subplot(1,2,2);
imhist(g_0_4);
title('gamma=0.4��v����ֱ��ͼ');



%���޶Ա�����Ӧֱ��ͼ���⻯
  X=adapthisteq(g_0_4);  %���޶Ա�����Ӧֱ��ͼ���⻯   ����Ĭ��ֵ
%  X = adapthisteq(g_0_4, 'NumTiles', [2 2]);
figure,
subplot(121),imshow(g_0_4);
title('gamma=0.4��v����ͼ��')
subplot(122),imshow(X);
title('���⻯��')
figure;
imhist(X);
title('���⻯��v����ֱ��ͼ');


HSV1=H+S+X;


%%ͼ��HSV2RGB
for i=1:m
    for j=1:n
        h=floor(H(i,j)/60);
        f=H(i,j)/60-h;
        v=X(i,j);
        s=S(i,j);
        p=v*(1-s);
        q=v*(1-f*s);
        t=v*(1-(1-f)*s);

        if h==0
            R(i,j)=v;G(i,j)=t;B(i,j)=p;
        elseif h==1
            R(i,j)=q;G(i,j)=v;B(i,j)=p;            
        elseif h==2
            R(i,j)=p;G(i,j)=v;B(i,j)=t;            
        elseif h==3
            R(i,j)=p;G(i,j)=q;B(i,j)=v;            
        elseif h==4
            R(i,j)=t;G(i,j)=p;B(i,j)=v;            
        elseif h==5
            R(i,j)=v;G(i,j)=p;B(i,j)=q;            
        end
    end
end
HSV1(:,:,1)=R;
HSV1(:,:,2)=G;
HSV1(:,:,3)=B;
figure,imshow(HSV1),title('��ԭ��rgbͼ��');
toc;


% �͹�����
X = double(img);  
 Z = double(HSV1);  
 A = Z-X;
B = X.*Z;
 D = Z-X;
MSE = sum(D(:).*D(:))/numel(D);%���������MSE
PSNR = 10*log10(255^2/MSE);%��ֵ�����
% SNR = 10*log10(sum(X(:).*X(:))/MSE/numel(Z));%�����SNR
%  MAE=mean(mean(abs(D)));%ƽ���������
%  %����Ϊ�ṹ���ƶ�SSIM
% ux=sum(X(:).*X(:))/numel(X);
% uy=sum(Z(:).*Z(:))/numel(Z);
% sigmoidx=sum(X(:).*X(:)-ux)/numel(X);
% sigmoidy=sum(Z(:).*Z(:)-uy)/numel(Z);
% sigmoidxy=sum(B(:).*B(:))/(numel(B)*ux*uy)-ux*uy;
% SSIM=(2*ux*uy)*(2*sigmoidxy)/(ux*ux+uy*uy)/(sigmoidx*sigmoidx+sigmoidy*sigmoidy);


display(MSE);%���������MSE
display(PSNR);%��ֵ�����
% display(SNR);%�����SNR
%  display(SSIM);%�ṹ������SSIM
% 

% % �Աȶȣ��������ػҶ�ֵ����Χ4�������ػҶ�ֵ֮���ƽ��֮�ͣ���������ƽ����ĸ�����
% [m,n] = size(HSV1);%��ԭʼͼ�������m������n
% g = padarray(HSV1,[1 1],'symmetric','both');%��ԭʼͼ�������չ������50*50��ͼ����չ����52*52��ͼ��
% %��չֻ�Ƕ�ԭʼͼ����ܱ����ؽ��и��Ƶķ�������
% [r,c] = size(g);%����չ��ͼ�������r������c
% g = double(g);  %����չ��ͼ��ת���˫���ȸ�����
% k = 0;  %����һ��ֵk����ʼֵΪ0
% fori=2:r-1
% forj=2:c-1
% k = k+(g(i,j-1)-g(i,j))^2+(g(i-1,j)-g(i,j))^2+(g(i,j+1)-g(i,j))^2+(g(i+1,j)-g(i,j))^2;
% cg = k/(4*(m-2)*(n-2)+3*(2*(m-2)+2*(n-2))+4*2);%��ԭʼͼ��Աȶ�
% display(cg);%�Աȶ�



% [C,L]=size(HSV1); %��ͼ��Ĺ��
% Img_size=C*L; %ͼ�����ص���ܸ���
% G=256; %ͼ��ĻҶȼ�
% H_x=0;
% nk=zeros(G,1);%����һ��G��1�е�ȫ�����
% for i=1:C
% for j=1:L
% Img_level=HSV1(i,j)+1; %��ȡͼ��ĻҶȼ�
% nk(Img_level)=nk(Img_level)+1; %ͳ��ÿ���Ҷȼ����صĵ���
% end
% end
% for k=1:G  %ѭ��
% Ps(k)=nk(k)/Img_size; %����ÿһ�����ص�ĸ���
% if Ps(k)~=0; %������ص�ĸ��ʲ�Ϊ��
% H_x=-Ps(k)*log2(Ps(k))+H_x;%����ֵ�Ĺ�ʽ
% end
% end
% H_x  %��ʾ��ֵ