close all;
clear;

images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');

N        = 1000;
lable2   = find(labels == 2, N);
s_images = images(:,lable2);

display_network(s_images(:,1:100)); % Show the first 100 images

%% finding eigenvectors
mDist = squareform( pdist(s_images') );
% figure; imagesc(M); colorbar;

epsilon = 1 * median(mDist(:));
K       = exp(-mDist.^2 / epsilon.^2);
A       = bsxfun(@rdivide, K, sum(K, 2));

[EigVec, EigVal] = eig(A);

%% plotting
figure; scatter3(EigVec(:,2), EigVec(:,3), EigVec(:,4), 100, 1:N, 'Fill'); colorbar;
title('diffusion map');
xlabel('\psi_2');
ylabel('\psi_3');
zlabel('\psi_4');

xImage = [EigVec(1,2) EigVec(1,2); EigVec(1,2) EigVec(1,2)];
yImage = [EigVec(1,3)+50 EigVec(1,3)+50; EigVec(1,3)-50 EigVec(1,3)-50];
zImage = [EigVec(1,4)-50 EigVec(1,4)+50; EigVec(1,4)-50 EigVec(1,4)+50];
figure; surf(xImage,yImage,zImage,'CData',vec2mat(s_images(:,1),28),...
     'FaceColor','texturemap');

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