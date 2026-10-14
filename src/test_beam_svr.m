function cost = test_beam_svr(x, Minput, Moutput, Minput_test, Moutput_test)
    % x = [BoxConstraint, KernelScale, Epsilon]

    boxConstraint = x(1);
    kernelScale = x(2);
    epsilon = x(3);

    % Train SVR with current hyperparameters
    svr_model = fitrsvm(Minput', Moutput', ...
        'KernelFunction', 'rbf', ...
        'BoxConstraint', boxConstraint, ...
        'KernelScale', kernelScale, ...
        'Epsilon', epsilon);

    % Prepare test data (same as your current logic)
    phi0_test = 11;
    phi1_test = 5;
    phi2_test = 15;

    phi_initial_test = 0;
    phi_final_test = 20;
    phi_step_test = 1;
    nPoints = ((phi_final_test - phi_initial_test) * (1 / phi_step_test)) + 1;

    for i = 1:nPoints
        Minput_ref(1, i) = phi0_test;
        Minput_ref(2, i) = phi1_test;
        Minput_ref(3, i) = phi2_test;
        Minput_ref(4, i) = Minput_test(4, i + 94794);
        Moutput_ref(1, i) = Moutput_test(i + 94794);
    end

    Minput_ANN = Minput_ref;

    % Predict
    Moutput_ANN = predict(svr_model, Minput_ANN');

    % Calculate error
    error = 0;
    MAX1 = abs(max(Moutput_ref(1, :)) - min(Moutput_ref(1, :)));
    for d = 1:nPoints
        if Moutput_ref(1, d) ~= 0
            c = (100 / nPoints) * abs((Moutput_ref(1, d) - Moutput_ANN(d)) / MAX1);
            error = error + c;
        end
    end

    cost = error;
end

