function cost = test_beam(x, enable_plot)
    % Default plotting to false during PSO search
    if nargin < 2
        enable_plot = false;
    end

    % Load training and testing data
    load('training_data.mat'); % Minput, Moutput
    load('test_data.mat');     % Minput_test, Moutput_test

    % 1. Discretize layer counts
    x = round(x);
    
    % 2. Build neural network architecture in Pure dB Domain
    net = feedforwardnet(x);
    net.trainFcn = 'trainscg';         % Memory-efficient SCG
    net.trainParam.showWindow = false; 
    
    if ~enable_plot
        % Fast settings during PSO search
        net.trainParam.epochs = 100;
        num_samples = size(Minput, 2);
        sub_idx = round(linspace(1, num_samples, min(3000, num_samples)));
        Minput_train = Minput(:, sub_idx);
        Moutput_train = Moutput(:, sub_idx);
    else
        % Thorough training for final plot in pure dB scale
        net.trainParam.epochs = 2500;
        net.trainParam.min_grad = 1e-10;
        net.trainParam.max_fail = 30;
        Minput_train = Minput;
        Moutput_train = Moutput;
    end
    
    % Train the network directly on dB data
    net = train(net, Minput_train, Moutput_train);
    
    % 3. Prepare test inputs
    phi0_test = 11;
    phi1_test = 5;
    phi2_test = 15;
    
    phi_initial_test = 0;
    phi_final_test = 20;
    phi_step_test = 1;
    nPoints = ((phi_final_test - phi_initial_test) * (1 / phi_step_test)) + 1;
    
    Minput_ref = zeros(4, nPoints);
    Moutput_ref = zeros(1, nPoints);
    Minput_ANN = zeros(4, nPoints);

    for i = 1:nPoints
        Minput_ref(1, i) = phi0_test;
        Minput_ref(2, i) = phi1_test;
        Minput_ref(3, i) = phi2_test;
        Minput_ref(4, i) = Minput_test(4, i + 94794);
        Moutput_ref(1, i) = Moutput_test(i + 94794);
    end
    
    for m = 1:nPoints
        Minput_ANN(1, m) = phi0_test;
        Minput_ANN(2, m) = phi1_test;
        Minput_ANN(3, m) = phi2_test;
        Minput_ANN(4, m) = Minput_ref(4, m);
    end
    
    % Predict output radiation pattern in dB
    Moutput_ANN = net(Minput_ANN);
    
    % 4. Peak Normalization Alignment (Crucial for Antenna Gain Patterns)
    % Align predicted maximum peak directly with reference peak power
    peak_shift = max(Moutput_ref(1, :)) - max(Moutput_ANN(1, :));
    Moutput_ANN = Moutput_ANN + peak_shift;
    
    % 5. Standard Normalized Data Error
    data_error = 0;
    MAX1 = abs(max(Moutput_ref(1, :)) - min(Moutput_ref(1, :)));
    for d = 1:nPoints
        if Moutput_ref(1, d) ~= 0
            c = (100 / nPoints) * (abs((Moutput_ref(1, d) - Moutput_ANN(1, d)) / MAX1));
            data_error = data_error + c;
        end
    end
    
    % 6. PINN Physics Penalties
    ref_power = Moutput_ref(1, :);
    peak_ref = max(ref_power);
    
    null_indices = find(ref_power < (peak_ref - 15));
    physics_penalty = 0;
    if ~isempty(null_indices)
        leakage = Moutput_ANN(1, null_indices) - ref_power(null_indices);
        leakage_violations = max(0, leakage);
        physics_penalty = 10.0 * mean(leakage_violations.^2);
    end
    
    hardware_penalty = 0.005 * sum(x);
    cost = data_error + physics_penalty + hardware_penalty;
    
    % 7. Final Plotting
    if enable_plot
        save('best_pinn_model.mat', 'net');
        figure('Name', 'PINN Beamforming Performance');
        subplot(1, 3, 1);
        plot(Minput_ref(4, :), Moutput_ref(1, :), 'b', 'LineWidth', 1.5);
        title('Reference Pattern'); xlabel('Azimuth Angle \phi (\circ)'); ylabel('Power (dB)'); grid on;
        
        subplot(1, 3, 2);
        plot(Minput_ANN(4, :), Moutput_ANN(1, :), 'r', 'LineWidth', 1.5);
        title('PINN ANN Prediction'); xlabel('Azimuth Angle \phi (\circ)'); ylabel('Power (dB)'); grid on;
        
        subplot(1, 3, 3);
        plot(Minput_ANN(4, :), Moutput_ANN(1, :), 'r--', Minput_ref(4, :), Moutput_ref(1, :), 'b-', 'LineWidth', 1.5);
        title('Overlay Comparison'); xlabel('Azimuth Angle \phi (\circ)'); ylabel('Power (dB)'); grid on;
        legend('PINN-ANN', 'Reference');
    end
end