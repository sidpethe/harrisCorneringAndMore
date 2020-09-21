%%
%   CLAB3 Task-1: Harris Corner Detector
%
close all;
clear all;

img = imread('Lenna.png');
imshow(img);
bw=(rgb2gray(img));

sigma = 2; thresh = 0.01; sze = 11; disp= 0;

dy = [-1 0 1;-1 0 1; -1 0 1];
dx = dy';

Ix = conv2(bw,dx,'same');
Iy = conv2(bw,dy,'same');

g = fspecial('gaussian',max(1,fix(6*sigma)),sigma);

Ix2 = conv2(Ix.^2,g,'same');
Iy2 = conv2(Iy.^2,g,'same');
Ixy = conv2(Ix.*Iy,g,'same');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Task: Compute the Harris Cornerness R
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A=sum(sum(Ix2));
% B=sum(sum(Ixy));
% C=sum(sum(Iy2));
% H=[A, B;B,C]; 
W=ones(sze+1,sze+1);
A=conv2(Ix2,W,'same');
B=conv2(Ixy,W,'same');
C=conv2(Iy2,W,'same');
R=zeros(size(Ix2,1),size(Ix2,2));
R1=zeros(size(Ix2,1),size(Ix2,2));
for row=1:size(Ix2,1)
    for col=1:size(Ix2,2)
        H=[A(row,col),B(row,col);B(row,col),C(row,col)];
        R1(row,col)=det(H)-((trace(H))^2);
    end
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Task: Perform non-maximum suppression and threshold here 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for row=1:size(R,1)
    for col=1:size(R,2)

         if  ((R1(row,col))>thresh)
             R(row,col)=R1(row,col);
         end
    end
end

%   Plot the corners on the image
[rows,cols] = find(R);
imshow(uint8(bw));
hold on;
p = [cols, rows];
plot(p(:,1),p(:,2),'or');
title('Harris Corners');
