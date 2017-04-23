close all;
clear;

images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');

N        = 1000;
% run on each digit
 for jj = 0 : 9
%for jj = 7
    lable    = find(labels == jj, N);
    s_images = images(:,lable);
    figure;
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
    title(['diffusion map of digit ' num2str(jj)]);
    xlabel('\psi_2');
    ylabel('\psi_3');
    zlabel('\psi_4');


    %% show example for psi2 & psi3

    figure;
    for ii = 1 : 150
            xImage = [EigVec(ii,2)-0.0015 EigVec(ii,2)-0.0015; EigVec(ii,2)+0.0015 EigVec(ii,2)+0.0015];
            yImage = [EigVec(ii,3)+0.0015 EigVec(ii,3)-0.0015; EigVec(ii,3)+0.0015 EigVec(ii,3)-0.0015];
            zImage = [EigVec(ii,4) EigVec(ii,4); EigVec(ii,4) EigVec(ii,4)];
            surf(xImage,yImage,zImage,'CData',vec2mat(s_images(:,ii),28),...
              'FaceColor','texturemap');
            hold on;
    end
    hold off;
     title(['sampels pictures of digit ' num2str(jj) ' for the first 2 eig-vecs']);
    xlabel('\psi_2');
    ylabel('\psi_3');
    zlabel('\psi_4');
end
