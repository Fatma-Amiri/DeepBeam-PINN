% ce code permet de faire le training

clc;
clear all;

load('training_data.mat');% permet de charger les données du training_data.mat: Minput et Moutpout



net = feedforwardnet([2, 2, 2]); % le réseau est composé de 3 couches cachées. chaque couche contient 10 neurones.
net.layerConnect = [0 0 0 0; 1 0 0 0; 0 1 0 0; 0 0 1 0]; % permetde connecter les différents couches du réseau


net.biasConnect = [1; 1; 1; 1]; % If net.biasConnect(i) is 1, then layer i has a bias
net.trainFcn = 'trainlm'; 

net= train(net, Minput, Moutput); 

% Minput est la matrice qui contient the training input data (phi0, phi1,
% phi2 et phi_data). phi_data sont les valeurs de l'axe des abscisses de la
% courbe Power=f(phi)
% Moutput est la matrice qui contient the training output data qui sont les
% valeurs de puissance de la courbe Power=f(phi)

save('training_beam.mat')