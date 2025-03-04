%% 1

%Remove any variables in memory space
clc;
clear all;
close all;

%Part1
%Loading image
img=imread("q1.jpg");
figure('Name','Original Image'); 
imshow(img);
%grayscale image
gray_img=rgb2gray(img);
figure('Name','Gray Scale Image'); 
imshow(gray_img);
%Get Fourier Transform of an image
F = fft2(gray_img);
figure('Name','Fourier transform of an image'); 
imshow(abs(F), []);
%Get the centered spectrum
Fsh = fftshift(F);
figure('Name','Centered fourier transform of Image');  
imshow(abs(Fsh), []);
%apply log transform
log_img = log(1+abs(Fsh));
log_img=log_img-min(log_img(:)); 
log_img=log_img/max(log_img(:));
figure('Name','Log fourier transform of Image');
imshow(log_img);
%Writing image
imwrite(log_img,"q1-res1.png");

%Part2
%Get Fourier Transform and centered spectrum of an image
Fshc= fftshift(fft2(img));
%Filtering image
[cL, cH] = getfilters(50);
h_ft = Fshc .* cH;
%Add image to filter image
Fshc=Fshc+h_ft;
%Get invers Fourier Transform and centered spectrum of an image
high_filtered_image = ifft2(ifftshift(Fshc), 'symmetric');
high_f = uint8(abs(high_filtered_image));
%Show image
figure('Name','sharp image');
imshow(high_f);
%Writing image
imwrite(high_f,"q1-res2.png");

%Part3
k=1;
Fshco= fftshift(fft2(img));
%imgg=double(img)+abs(k.*ifft(ifftshift(4.*(pi.^2).*(abs(Fshco).^2).*Fshco)));
for i=1:565
   for j=1:565
       imgg(i,j)=double(img(i,j))+abs(k.*(ifft(ifftshift(4.*(pi.^2).*(j.^2+i.^2).*Fshco(i,j)))));
   end
end
imgg=imgg-min(imgg(:)); 
imgg=imgg/max(imgg(:));
figure('Name','sharp2 image');
imshow(imgg);
%Writing image
imwrite(imgg,"q1-res3.png");
%% 2

%Remove any variables in memory space
clear all;
close all;
clc;

%loading image
[img,map] =imread("pic.jpg");
figure('Name','Original Image'); 
imshow(img);
% figure('Name','Original Image'); 
% imhist(p_image);
%Using Matlab function
p_image=adapthisteq(img,'NBins',32384,'Distribution','uniform','ClipLimit',0.012);
figure('Name','Resault Image'); 
imshow(p_image);
%Writing image
imwrite(p_image,"Q2.jpg")
%%  3
clc;
clear;
clear all;
close all;
%Loading image
marilyne=imread("marilyn.jpg");
einstein=imread("einstein.jpg");
% Compute FFT of the grey image 
% Frequency scaling
% Convert the image to grey 
einstein=rgb2gray(einstein);
Fmarilyne=fftshift(fft2(marilyne));
Feinstein=fftshift(fft2(einstein));
%Resizing image
Fmarilyne(235,:)=[];

% Gaussian Filter Response Calculation
[M N]=size(fft2(einstein)); % Image size
R=2; %Filter size parameter 
X=0:N-1;
Y=0:M-1;
[X, Y]=meshgrid(X,Y);
Cx=0.5*N;
Cy=0.5*M;
Lo=exp(-((X-Cx).^2+(Y-Cy).^2)./(2*R).^2);
Hi=1-Lo; % High pass filter=1-low pass filter
%Get invers Fourier Transform and centered spectrum of an image
B1=ifft2(ifftshift(Fmarilyne.*Lo));
B2=ifft2(ifftshift(Feinstein.*Hi));
%Showing image
figure('Name','low pass filtered image')
imshow(abs(B1),[12 290]), colormap gray
figure('Name','low pass filtered image')
imshow(abs(B2),[12 290]), colormap gray
%Show resault
figure('Name','Final image')
imshow((abs(B2)+abs(B1))/2,[12 290]), colormap gray

%% function

function [cL,cH] = getfilters(radius)
    [x,y] = meshgrid(-282:282,-282:282);
    z = sqrt(x.^2+y.^2);
    cL = z < radius;
    cH = ~cL;
end
