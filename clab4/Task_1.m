clc;
clear all;
close all;

%% Task 1:
%Read all the 135 training face images, vectorise
% each image as a single column vector, and collect all
% vectors as a big data matrix A. You can use reshape() to
% convert the matrix into a vector, or simply typing I(:)
% will return the vectorised form of an image matrix I.
% If you have resized the image as for example 100x100,
% then the matrix A would have the size of 10000x135
% (the number of pixels times the number of images).
% Don’t forget to remove the mean vector from the matrix A.

trainpath = 'yalefaces_1/trainingset/';
testpath = 'yalefaces_1/testset/';
file = 's*';
train_filenames = dir([trainpath file]);    % return a structure with filenames
test_filenames = dir([testpath file]);      % return a structure with filenames
train_size = length(train_filenames);
test_size = length(test_filenames);

for i = 1 : train_size
    filename = [trainpath train_filenames(i).name];   % filename in the list
    Img(:,:,i) = imread(filename);
end

Img1 = imresize (Img , [100 100]);
%A = Img1(:,:,1);
Img2 = reshape (Img1, [10000 train_size]);

%I_mean_c = mean(Img2);
I_mean_r = mean(Img2,2);
% for i = 1 : 10000
% Img_cmean(i,:) = double(Img2(i,:)) - I_mean_c;
% end
for i = 1 : train_size
    Img_rmean(:,i) = double(Img2(:,i)) - I_mean_r;
end

%% Task 2:
% 2. Perform Principal Component Analysis (PCA) on the
% data matrix
%
% a) For the covariance matrix C = AAT, determine
% the top k-principal components, display the top
% k-eigenfaces on your screen (and also include in your
% report). k can be chosen at k=10 or k=15.

%Covariance matrix At.A for simple computation
Img_rmean_t = Img_rmean.';
covar_mtx = Img_rmean_t * Img_rmean;

[e_vec, e_val] = eig (covar_mtx);

%sort E-value
[E_val_sort_1, E_val_pos_1] = sort(e_val);
[E_val_sort_2, E_val_pos_2] = sort(E_val_sort_1(train_size,:),2,'descend');

%Re-order E-vector based on E-value
for i = 1 : train_size
    E_vec_sort(:,i) = e_vec(:,E_val_pos_2(1,i));
end

%Getting E-vector for A.AT)
E_vec_map = Img_rmean * E_vec_sort;
E_vec_map = real(E_vec_map);

k = 10;

figure();
for i = 1 : k
    Img_disp = reshape(E_vec_map(:,i), [100 100]);
    
    min1 = min(Img_disp,[],1);
    min2 = min(min1,[],2);
    max1 = max(Img_disp,[],1);
    max2 = max(max1,[],2);
    
    subplot(2,5,i);
    imshow(Img_disp,[min2 max2]);
    
    E_vec_top(:,i) = E_vec_map(:,i);
    
end

%% b) Project each training image onto the space spanned
% by the top k-eigenfaces.

for i = 1 : train_size
    for j = 1 : size(E_vec_top,2)
        
        Img_project(i,j) = Img_rmean_t(i,:) * E_vec_top(:,j);
        
    end
end

E_vec_map_t = E_vec_map';

for i = 1 : train_size
    for j = 1 : size(E_vec_map,2)
        
        Img_project_all(i,j) = Img_rmean_t(i,:) * E_vec_map(:,j);
        %Img_project_all(i,j) = E_vec_map(j,:) * Img_rmean_t(:,i);
        
    end
end



%% c) Repeat this for each test image, projecting onto the
% space spanned by the top k-eigenfaces. Use this projection
% and Euclidian distance metric, perform a nearest-neighbour
% search over all the 135 faces. Find out which three images
% are the most similar to the test faces. Display these top
% 3 faces next to your test image on screen

for i = 1 : test_size
    filename_test = [testpath test_filenames(i).name];   % filename in the list
    Img_test(:,:,i) = imread(filename_test);
end

Img_test_1 = imresize (Img_test , [100 100]);
%A = Img_test_1(:,:,1);
Img_test_2 = reshape (Img_test_1, [10000 test_size]);

%mean
I_test_mean_r = mean(Img_test_2,2);

for i = 1 : test_size
    Img_test_rmean(:,i) = double(Img_test_2(:,i)) - I_test_mean_r;
end

Img_test_rmean_t = Img_test_rmean';

for i = 1 : test_size
    for j = 1 : size(E_vec_top,2)
        
        Img_test_project(i,j) = Img_test_rmean_t(i,:) * E_vec_top(:,j);
        
    end
end

%finding euclidian dist b/n test & train projections
equi_dist = pdist2(Img_test_project , Img_project);

%sort dist
[equi_sort , equi_loc] = sort (equi_dist,2);

%display 3 similar images

i = 0;
for j = 1 : 2
    figure();
    for k = 1 : 5
        i = i + 1;
        Im_test =reshape( Img_test_2(:,i) , [100 100]);
        Im_test_1 =reshape( Img2(:,equi_loc(i,1)) , [100 100]);
        Im_test_2 =reshape( Img2(:,equi_loc(i,2)) , [100 100]);
        Im_test_3 =reshape( Img2(:,equi_loc(i,3)) , [100 100]);
        
        subplot(4,5,k); imshow(Im_test,[]),title('Test Image');
        subplot(4,5,k+5); imshow(Im_test_1,[]),title('Match 1');
        subplot(4,5,k+10); imshow(Im_test_2,[]),title('Match 2');
        subplot(4,5,k+15); imshow(Im_test_3,[]),title('Match 3');
    end
end

%% Task 3:
%Test the face recogniser on your own face image.
% a) Read in one of your own frontal face images.
% Convert it to grey-scale, manually crop out the facial
% region out, and resize as a new image file:
% "my_face_cropped.jpg”. Then run your face recogniser on
% this new image file. Display the top 3 faces in the training
% folder that are most similar to your own faces.

%face_my_test = imread('subject16.normal.png');
face_my_test = imread('face1.jpg');
face_my_test = rgb2gray(face_my_test);
face_my_test_1 = imresize (face_my_test , [100 100]);
%A = Img_test_1(:,:,1);
face_my_test_2 = reshape (face_my_test_1, [10000 1]);

face_my_test_2 = double(face_my_test_2) - I_mean_r;
face_my_test_2_t = face_my_test_2';

%PCA & check
%for i = 1 : 1
for j = 1 : size(E_vec_top,2)
    
    Img_face_project(j) = double(face_my_test_2_t) * E_vec_top(:,j);
    
end
%end

%finding euclidian dist b/n test & train projections

equi_face_dist = pdist2(Img_face_project , Img_project);


%sort dist
[equi_face_sort , equi_face_loc] = sort (equi_face_dist,2);

%display 3 similar images
for i = 1 : 1
    Im_test =reshape( face_my_test_2_t , [100 100]);
    Im_test_1 =reshape( Img2(:,equi_face_loc(i,1)) , [100 100]);
    Im_test_2 =reshape( Img2(:,equi_face_loc(i,2)) , [100 100]);
    Im_test_3 =reshape( Img2(:,equi_face_loc(i,3)) , [100 100]);
    figure();
    subplot(3,3,2); imshow(face_my_test_1,[]),title('Test Image');
    subplot(3,3,4); imshow(Im_test_1,[]),title('Match Image 1');
    subplot(3,3,5); imshow(Im_test_2,[]),title('Match Image 2');
    subplot(3,3,6); imshow(Im_test_3,[]),title('Match Image 3');
end


% b) You may also wish to repeat this experiment by pre-adding
% your own faces into the training dataset. Of course, make
% sure your test face image is a different one - that is not
% included in the training set.

%% Q1. For your own face image, how much of the energy is captured by the
% first k principal components (in percentage %)? Plot a curve of the energy
% percentages as a function of k, for k = 2, 3, 4, 5, 6 , 7, 8, 9,10. 
%     The energy is defined as the sum of coefficients projected onto 
%     the eigenface space.

% Q2. How many principal components are necessary to obtain a visually 
% recognisable face reconstruction? That is, if you take an image and 
%     project it onto the space spanned by the first k principal components,
%     how big does k have to be before the images look recognisable? You may
%     try this for k=2, 4, 6, 8, 10, … and display the reconstruction 
%         results.

figure();
energy = zeros(10,1);
s = zeros(10,1);
for j = 1 : 10
    %I_mean_e_face = mean(E_vec_top,2);
    project_train = zeros(10000,1);
    for i = 1 : j
        projcet_w_v = (Img_test_project(1,i) * E_vec_top(:,i));
        project_train = project_train + projcet_w_v;
        %Energy
        energy(j) = energy(j) + (Img_test_project(1,i) ^ 2);
        s(i) = svds(energy);
    end
    project_train_main = project_train + I_mean_r;
    rest1 = reshape(project_train_main, [100 100]);
    
    if (rem(j,2) == 0)
        str= ['k=' num2str(j)];
        subplot(1,5,j/2);
        imshow(rest1,[]);title(str);
    end
end

energy_all = Img_project_all.^ 2;
energy_all_1 = sum(energy_all);
energy_all_2 = sum(energy_all_1);
    
energy1 = energy /  max(energy);
energy2 = (energy / energy_all_2) * 100;

figure();
plot(energy2),xlabel('K-Value'),ylabel('% energy'),title('Energy Graph');

%% Notes
% Before doing anything, you must make sure that all face images
% must be geometrically aligned. Specifically, you need to
% take into account of the differences in position of the
% face in each image. A simple way is, before you do any
% training or testing, you should manually crop the face
% region out, define a standard window size, resize the
% face image, make sure the face region are all aligned -
% e.g. eyes, noses, mouths are roughly at the same positions
% in an image - save the results into disk, so you don't
% have to do the above pre-processing more than once.
%
% • In doing eigen-decomposition, always remember to
% subtract the mean face and, when reconstructing images
% based on the first k-principal components, add the mean
% face back in at the end.
%
% • You can Matlab's eigs() or svds() functions to implement PCA.
% Other than this, you should not use Matlab's inbuilt eigenface
% function if there is one.
%
% • If you encounter some difficulty in solving the eigenvalues
% of the covariance matrix AAT, think about whether or not
% you can use a faster way to compute them using ATA. Read
% lecture notes if you do not know what I mean by this faster
% method.
