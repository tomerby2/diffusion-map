close all;
clc;
clear;
images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');
N=100;
lable2 = find(labels==2,N);
s_images=images(:,lable2);
display_network(s_images(:,1:100)); % Show the first 100 images
Xi=repelem(s_images,1,N);
Xj=repmat(s_images,1,N);
NORMS=vec2mat(sum((Xi-Xj).^2,1),N);
epsilon=1200;
K=exp(-NORMS/epsilon);
D=diag(K*ones(N,1));
A=(inv(D))*K;
[EigVec, EigVal] = eig(A);
t=1;
psi=(EigVal.^t)*(EigVec');
figure;
scatter3(EigVec(:,2),EigVec(:,3),EigVec(:,4));
figure;
scatter(EigVec(:,2),EigVec(:,3));

