% Load trained models and test data
load('best_pinn_model.mat'); % Loads net (PINN)
load('svr_model.mat');       % Loads svr_model / mdl
load('test_data.mat');       % Loads test dataset

% Test evaluation coordinates
phi0_test = 11; phi1_test = 5; phi2_test = 15;
phi_initial = 0; phi_final = 20; phi_step = 1;
nPoints = ((phi_final - phi_initial) * (1 / phi_step)) + 1;

Minput_ref = zeros(4, nPoints);
Moutput_ref = zeros(1, nPoints);

for i = 1:nPoints
    Minput_ref(1, i) = phi0_test;
    Minput_ref(2, i) = phi1_test;
    Minput_ref(3, i) = phi2_test;
    Minput_ref(4, i) = Minput_test(4, i + 94794);
    Moutput_ref(1, i) = Moutput_test(i + 94794);
end

% 1. Evaluate PINN Model
Moutput_PINN = net(Minput_ref);
Moutput_PINN = Moutput_PINN + (max(Moutput_ref(1, :)) - max(Moutput_PINN(1, :))); % Peak normalization

% 2. Evaluate SVR Model
if exist('svr_model', 'var')
    Moutput_SVR = predict(svr_model, Minput_ref');
elseif exist('mdl', 'var')
    Moutput_SVR = predict(mdl, Minput_ref');
end

% 3. Unified Comparison Plot
figure('Name', 'Model Performance Benchmark', 'Position', [100, 100, 900, 500]);
plot(Minput_ref(4, :), Moutput_ref(1, :), 'k-', 'LineWidth', 2.0, 'DisplayName', 'Reference Pattern'); hold on;
plot(Minput_ref(4, :), Moutput_PINN(1, :), 'r--', 'LineWidth', 1.8, 'DisplayName', 'PINN-ANN [256, 128, 128]');
plot(Minput_ref(4, :), Moutput_SVR, 'g-.', 'LineWidth', 1.5, 'DisplayName', 'SVR Baseline');

title('Beamforming Pattern Comparison: Reference vs PINN vs SVR');
xlabel('Azimuth Angle \phi (\circ)');
ylabel('Power (dB)');
ylim([-125, 5]);
grid on;
legend('Location', 'northeast');

% 4. Compute and Display Performance Metrics Table
pinn_rmse = sqrt(mean((Moutput_ref(1, :) - Moutput_PINN(1, :)).^2));
svr_rmse  = sqrt(mean((Moutput_ref(1, :) - Moutput_SVR').^2));

fprintf('\n================ MODEL PERFORMANCE BENCHMARK ================\n');
fprintf('Metric                      | SVR Baseline | PINN-ANN Model\n');
fprintf('-------------------------------------------------------------\n');
fprintf('Global Pattern RMSE (dB)    | %12.4f | %14.4f\n', svr_rmse, pinn_rmse);
fprintf('Peak Power @ 11 deg (dB)    | %12.4f | %14.4f\n', Moutput_SVR(12), Moutput_PINN(1, 12));
fprintf('Null Depth @ 5 deg (dB)     | %12.4f | %14.4f\n', Moutput_SVR(6), Moutput_PINN(1, 6));
fprintf('Null Depth @ 15 deg (dB)    | %12.4f | %14.4f\n', Moutput_SVR(16), Moutput_PINN(1, 16));
fprintf('=============================================================\n');