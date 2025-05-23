clc;
close all;
clear all;

% User-defined parameters
min_input = input('Enter lower bound of input: ');
max_input = input('Enter upper bound of input: ');
num_rules = input('Enter number of membership functions: ');
learning_rate = input('Enter learning rate for P matrix: ');
num_samples = input('Enter number of input samples: ');
noise_level = input('Enter noise level (standard deviation): ');

rng(42);  % For reproducibility

% Fuzzy system parameters
centers = linspace(min_input, max_input, num_rules);
width = (max_input - min_input) / num_rules;
max_error_order = input('max_error_order: ');
% Number of past errors to include in regressor

% Initialize ELS parameters
theta =16*ones(num_rules, 1);  % Initial parameters
theta_e=zeros(max_error_order,1);
theta=[theta;theta_e]
P_matrix = learning_rate * eye(num_rules + max_error_order);  % Initial covariance

% Data storage
predictions = zeros(num_samples, 1);
true_outputs = zeros(num_samples, 1);
noisy_outputs = zeros(num_samples, 1);
inputs = linspace(min_input, max_input, num_samples)';
trace_p = zeros(num_samples, 1);
mean_errors = [];
theta_history = zeros(num_samples, num_rules + max_error_order);

% Generate colored noise (Brownian)
white_noise = noise_level * randn(num_samples, 1);
bnoise = cumsum(white_noise);
bnoise = bnoise / max(abs(bnoise));  % Normalize
bnoise = bnoise(randperm(length(bnoise)));  % Shuffle

% Error memory buffer (fixed after initialization)
error_memory = zeros(max_error_order, 1);

% Phase 1: First loop - collect 5 errors (no error in regressor)
for i = 1:max_error_order
    x = inputs(i);
    y = 10 * (x^4) * cosh(x);
    y_noisy = y + bnoise(i);

    true_outputs(i) = y;
    noisy_outputs(i) = y_noisy;

    % Compute fuzzy rule activations
    mu = exp(-((x - centers)/width).^2)';
    b = mu / sum(mu);

    % Regressor only contains membership vector
    phi = [b; zeros(max_error_order, 1)];

    % Predict output
    y_pred = phi' * theta;
    e = y_noisy - y_pred;

    % Update parameters using ELS
    K = (P_matrix * phi) / (1 + phi' * P_matrix * phi);
    theta = theta + K * e;
    P_matrix = P_matrix - K * phi' * P_matrix;

    % Save error into memory
    error_memory(i) = e;

    % Store results
    predictions(i) = y_pred;
    theta_history(i, :) = theta';
    trace_p(i) = trace(P_matrix);
    mean_errors = [mean_errors; mean(true_outputs(1:i) - predictions(1:i))];

    fprintf('Init Step %d: x=%.4f, y=%.4f, y_noisy=%.4f, y_pred=%.4f, mean_error=%.4f\n', ...
        i, x, y, y_noisy, y_pred, mean_errors(end));
end

% Phase 2: Main loop - use fixed error memory in regressor
for i = max_error_order+1:num_samples
    x = inputs(i);
    y = 10 * (x^4) * cosh(x);
    y_noisy = y + bnoise(i);

    true_outputs(i) = y;
    noisy_outputs(i) = y_noisy;

    % Compute fuzzy rule activations
    mu = exp(-((x - centers)/width).^2)';
    b = mu / sum(mu);

    % Regressor includes both b and fixed error memory
    phi = [b; error_memory];

    % Predict output
    y_pred = phi' * theta;
    e = y_noisy - y_pred;

    % Update parameters using ELS
    K = (P_matrix * phi) / (1 + phi' * P_matrix * phi);
    theta = theta + K * e;
    P_matrix = P_matrix - K * phi' * P_matrix;

    % NOTE: error_memory is NOT updated anymore

    % Store results
    predictions(i) = y_pred;
    theta_history(i, :) = theta';
    trace_p(i) = trace(P_matrix);
    mean_errors = [mean_errors; mean(true_outputs(1:i) - predictions(1:i))];

    fprintf('Train Step %d: x=%.4f, y=%.4f, y_noisy=%.4f, y_pred=%.4f, mean_error=%.4f\n', ...
        i, x, y, y_noisy, y_pred, mean_errors(end));
end

% Plot results
figure;
plot(inputs, true_outputs, 'b', inputs, predictions, 'r--');
legend('True Output', 'Fuzzy Prediction');
title('System Approximation with Colored Noise');

figure;
plot(true_outputs - predictions);
title('Prediction Error (True - Estimated)');

figure;
plot(trace_p);
title('Trace of P Matrix (Uncertainty)');

figure;
pwelch(bnoise, [], [], [], 1);
title('Power Spectral Density of Colored Noise');

figure;
hold on;
for k = 1:(num_rules + max_error_order)
    plot(theta_history(:, k), 'DisplayName', sprintf('theta_%d', k));
end
hold off;
title('Evolution of Parameters (Theta)');
xlabel('Step');
ylabel('Theta Value');
legend('show');
grid on;

figure;
plot(mean_errors, 'LineWidth', 1.5);
title('Mean Error over Time');
xlabel('Step');
ylabel('Mean Error');
grid on;
