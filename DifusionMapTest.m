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

%% plotting scatter 3D
figure; scatter3(EigVec(:,2), EigVec(:,3), EigVec(:,4), 100, 1:N, 'Fill'); colorbar;
title('diffusion map');
xlabel('\psi_2');
ylabel('\psi_3');
zlabel('\psi_4');


%% show example for psi2 & psi3

figure;
for i=1:N
    if( abs(EigVec(i,3))<0.005 || abs(EigVec(i,2))<0.005)
        xImage = [EigVec(i,2)-0.001 EigVec(i,2)-0.001; EigVec(i,2)+0.001 EigVec(i,2)+0.001];
        yImage = [EigVec(i,3)+0.001 EigVec(i,3)-0.001; EigVec(i,3)+0.001 EigVec(i,3)-0.001];
        zImage = [EigVec(i,4) EigVec(i,4); EigVec(i,4) EigVec(i,4)];
        surf(xImage,yImage,zImage,'CData',vec2mat(s_images(:,i),28),...
          'FaceColor','texturemap');
        hold on;
    end
end
 title('sampels of pictures for the first 2 eig-vecs');
xlabel('\psi_2');
ylabel('\psi_3');
zlabel('\psi_4');

