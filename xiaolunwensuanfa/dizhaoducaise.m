%% 低照度彩色图像
close all;clear ;clc;
%% 
tic;
%       img=imread('E:\picture\huaping.bmp','bmp');
       img=imread('E:\picture\friut.bmp','bmp');
%     img=imread('E:\picture\he.bmp','bmp');
%     img=imread('E:\picture\sea.bmp','bmp');
%      img=imread('E:\picture\cat.bmp','bmp');



img=mat2gray(img);  %任意区间映射到[0,1];
[m,n,dim]=size(img);
% imshow(img);
%%图像的RGB
R=img(:,:,1);
G=img(:,:,2);
B=img(:,:,3);

%%图像的RGB2HSV
H=zeros(m,n);   %色相角
S=zeros(m,n);   %饱和度
V=zeros(m,n);   %明度
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
imshow(img);title('原始图像'); 

figure;
subplot(1,3,1);imshow(H);title('HSV空间H分量图像');
subplot(1,3,2);imshow(S);title('HSV空间S分量图像');
subplot(1,3,3);imshow(V);title('HSV空间V分量图像');


% gamma变换调整灰度
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
title('v分量直方图');
subplot(1,2,2);
imhist(g_0_4);
title('gamma=0.4后v分量直方图');



%有限对比自适应直方图均衡化
  X=adapthisteq(g_0_4);  %有限对比自适应直方图均衡化   采用默认值
%  X = adapthisteq(g_0_4, 'NumTiles', [2 2]);
figure,
subplot(121),imshow(g_0_4);
title('gamma=0.4的v分量图像')
subplot(122),imshow(X);
title('均衡化后')
figure;
imhist(X);
title('均衡化后v分量直方图');


HSV1=H+S+X;


%%图像HSV2RGB
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
figure,imshow(HSV1),title('还原的rgb图像');
toc;


% 客观评价
X = double(img);  
 Z = double(HSV1);  
 A = Z-X;
B = X.*Z;
 D = Z-X;
MSE = sum(D(:).*D(:))/numel(D);%均方根误差MSE
PSNR = 10*log10(255^2/MSE);%峰值信噪比
% SNR = 10*log10(sum(X(:).*X(:))/MSE/numel(Z));%信噪比SNR
%  MAE=mean(mean(abs(D)));%平均绝对误差
%  %以下为结构相似度SSIM
% ux=sum(X(:).*X(:))/numel(X);
% uy=sum(Z(:).*Z(:))/numel(Z);
% sigmoidx=sum(X(:).*X(:)-ux)/numel(X);
% sigmoidy=sum(Z(:).*Z(:)-uy)/numel(Z);
% sigmoidxy=sum(B(:).*B(:))/(numel(B)*ux*uy)-ux*uy;
% SSIM=(2*ux*uy)*(2*sigmoidxy)/(ux*ux+uy*uy)/(sigmoidx*sigmoidx+sigmoidy*sigmoidy);


display(MSE);%均方根误差MSE
display(PSNR);%峰值信噪比
% display(SNR);%信噪比SNR
%  display(SSIM);%结构相似性SSIM
% 

% % 对比度：中心像素灰度值与周围4近邻像素灰度值之差的平方之和，除以以上平方项的个数。
% [m,n] = size(HSV1);%求原始图像的行数m和列数n
% g = padarray(HSV1,[1 1],'symmetric','both');%对原始图像进行扩展，比如50*50的图像，扩展后变成52*52的图像，
% %扩展只是对原始图像的周边像素进行复制的方法进行
% [r,c] = size(g);%求扩展后图像的行数r和列数c
% g = double(g);  %把扩展后图像转变成双精度浮点数
% k = 0;  %定义一数值k，初始值为0
% fori=2:r-1
% forj=2:c-1
% k = k+(g(i,j-1)-g(i,j))^2+(g(i-1,j)-g(i,j))^2+(g(i,j+1)-g(i,j))^2+(g(i+1,j)-g(i,j))^2;
% cg = k/(4*(m-2)*(n-2)+3*(2*(m-2)+2*(n-2))+4*2);%求原始图像对比度
% display(cg);%对比度



% [C,L]=size(HSV1); %求图像的规格
% Img_size=C*L; %图像像素点的总个数
% G=256; %图像的灰度级
% H_x=0;
% nk=zeros(G,1);%产生一个G行1列的全零矩阵
% for i=1:C
% for j=1:L
% Img_level=HSV1(i,j)+1; %获取图像的灰度级
% nk(Img_level)=nk(Img_level)+1; %统计每个灰度级像素的点数
% end
% end
% for k=1:G  %循环
% Ps(k)=nk(k)/Img_size; %计算每一个像素点的概率
% if Ps(k)~=0; %如果像素点的概率不为零
% H_x=-Ps(k)*log2(Ps(k))+H_x;%求熵值的公式
% end
% end
% H_x  %显示熵值