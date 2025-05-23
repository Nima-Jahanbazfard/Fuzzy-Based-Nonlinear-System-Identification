clc
clear all
close all

% Input the parameters
min_input = input('Enter lower bound of input: ');
max_input = input('Enter upper bound of input: ');
num_rules = input('Enter number of membership functions: ');
learning_rate = input('Enter learning rate for P matrix: ');
num_samples = input('Enter number of input samples: ');
noise_level = input('Enter noise level (standard deviation): ');



rng(42);

% Initialize Fuzzy System
centers = linspace(min_input, max_input, num_rules);
width = (max_input - min_input) / num_rules;
theta = 20 * eye(num_rules, 1);
P_matrix = learning_rate * eye(num_rules);

% Initialize Storage for Results
predictions = [];
errors = [];
true_outputs = [];
noisy_outputs = [];
inputs = linspace(min_input, max_input, num_samples)'; % Generate inputs in order
trace_p = [];
mean_errors = [];
theta_history = zeros(num_samples, num_rules); % Store all theta values

% Generate colored noise ( process for brown noise)
bnoise = zeros(num_samples, 1);
white_noise =noise_level* randn(num_samples,1);
bnoise=cumsum(white_noise);
bnoise=bnoise/max(abs(bnoise));
bnoise=bnoise(randperm(length(bnoise)));
% Online Training with Sequential Data
for i = 1:num_samples
    % Get current input (sequential)
    x = inputs(i);
    
    % Calculate true output
    y = 10 * (x^4) * cosh(x);
    
    % Add colored noise to the output
    y_noisy = y + bnoise(i);
    
    % Store true and noisy outputs
    true_outputs = [true_outputs; y];
    noisy_outputs = [noisy_outputs; y_noisy];
    
    % Calculate rule activations
    mu = exp(-((x - centers)/width).^2)';
    b = mu/sum(mu);
    
    % Calculate fuzzy output
    y_pred = b' * theta;
    
    % Update weights (RLS algorithm) - using noisy output
    K = P_matrix * b / (1 + b' * P_matrix * b);
    theta = theta + K * (y_noisy - y_pred);
    P_matrix = P_matrix - K * b' * P_matrix;
    trace_p = [trace_p; trace(P_matrix)];

    % Store all theta values
    theta_history(i, :) = theta';
    
    % Store results
    predictions = [predictions; y_pred];
    current_error = mean((true_outputs(1:i) - predictions(1:i)));
    mean_errors = [mean_errors; current_error];
    
    % Display progress
    fprintf('Step %d: x=%.4f, y_true=%.4f, y_noisy=%.4f, y_pred=%.4f\n , mean_errors=%.4f\n', i, x, y, y_noisy, y_pred,current_error);
end


% System approximation plot
figure;
plot(inputs, true_outputs, 'b', inputs, predictions, 'r--');
legend('True','Fuzzy');
title('System Approximation with Colored Noise');


% Error between true output and prediction
figure;
plot(true_outputs - predictions);
title('Error between True Output and Prediction');

% Trace of P matrix plot
figure;
plot(trace_p);
title('Trace of P Matrix');

% Plot power spectral density to show noise coloring
figure;
pwelch(bnoise, [], [], [], 1);
title('Power Spectral Density of Colored Noise');

% Plot all theta weights
figure;
hold on;
for k = 1:num_rules
    plot(theta_history(:, k), 'DisplayName', sprintf('theta_%d', k));
end
hold off;
title('Evolution of All Theta Weights');
xlabel('Step');
ylabel('Theta Value');
legend('show');
grid on;

%mean error plot
figure;
plot(mean_errors, 'LineWidth', 1.5);
title('Mean Error (True Output vs Prediction)');
xlabel('Time Step');
ylabel('Mean Error');
grid on;