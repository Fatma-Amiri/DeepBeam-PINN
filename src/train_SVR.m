% train_SVR.m - Optimized SVR Baseline Training
load('training_data.mat'); % Load Minput, Moutput

% Subsample training set evenly to ensure fast convergence without RAM bottleneck
num_samples = size(Minput, 2);
sub_idx = round(linspace(1, num_samples, min(10000, num_samples)));
X_train = Minput(:, sub_idx)';
Y_train = Moutput(1, sub_idx)';

% Optimized SVR Parameters
boxConstraint = 131.9722;
kernelScale   = 1.000000528;
epsilon       = 0.00112;

fprintf('Training SVR with feature standardization...\n');

svr_model = fitrsvm(X_train, Y_train, ...
    'KernelFunction', 'rbf', ...
    'Standardize', true, ...        % CRITICAL: Normalizes input feature scales
    'BoxConstraint', boxConstraint, ...
    'KernelScale', kernelScale, ...
    'Epsilon', epsilon);

% Save retrained SVR model
save('svr_model.mat', 'svr_model');
disp('Optimized SVR Model trained and saved to svr_model.mat successfully.');

