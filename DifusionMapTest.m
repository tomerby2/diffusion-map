close all;
clear;

images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');

N        = 1000;
lable2   = find(labels == 2, N);
s_images = images(:,lable2);

display_network(s_images(:,1:100)); % Show the first 100 images

%%
mDist = squareform( pdist(s_images') );
% figure; imagesc(M); colorbar;

epsilon = 1 * median(mDist(:));
K       = exp(-mDist.^2 / epsilon.^2);
A       = bsxfun(@rdivide, K, sum(K, 2));

[EigVec, EigVal] = eig(A);

%%
figure; scatter3(EigVec(:,2), EigVec(:,3), EigVec(:,4), 100, 1:N, 'Fill'); colorbar;

%%

% t=1;
% psi=(EigVal.^t)*(EigVec');
% figure;
% scatter3(EigVec(:,2),EigVec(:,3),EigVec(:,4));
% figure;
% scatter(EigVec(:,2),EigVec(:,3));
% 
% f = @(x,y) x*y
% f(4, 3)