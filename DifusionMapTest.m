close all;
clc;
clear;
images = loadMNISTImages('train-images.idx3-ubyte');
N=100;
s_images=images(:,1:N);
%[~,N]=size(images);
display_network(images(:,1:100)); % Show the first 100 images
Xi=repelem(s_images,1,N);
Xj=repmat(s_images,1,N);
NORMS=vec2mat(sum((Xi-Xj).^2,1),N);
epsilon=20;
K=exp(-NORMS/epsilon);
D=diag(K*ones(N,1));
A=(inv(D))*K;
[EigVec, EigVal] = eig(A);
t=1;
psi=(EigVal.^t)*(EigVec');
figure;
scatter(EigVec(:,2),EigVec(:,3));