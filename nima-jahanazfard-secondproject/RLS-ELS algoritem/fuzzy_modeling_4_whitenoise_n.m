clc
clear all
close all

% Input the parameters
min_input = input('Enter lower bound of input: ');
max_input = input('Enter upper bound of input: ');
num_rules = input('Enter number of rules: ');
learning_rate = input('Enter learning rate for P matrix: ');
num_samples = input('Enter number of input samples: ');
noise_level = input('Enter noise level (standard deviation): ');

%  random number for fixing noise
rng(42); 

% Initializing
centers = linspace(min_input, max_input, num_rules);
width = (max_input - min_input) / num_rules;
theta = 20*eye(num_rules, 1);
P_matrix = learning_rate * eye(num_rules);

%Storage of Results
predictions = [];
true_outputs = [];
noisy_outputs = [];
inputs = linspace(min_input, max_input, num_samples)'; 
trace_p = [];
mean_errors = []; 
theta_history = zeros(num_samples, num_rules); 

% Generate noise 
noise = noise_level * randn(num_samples, 1);
noise= noise(randperm(length(noise)));

% Training with Sequential Data
for i = 1:num_samples

    % making data
    x = inputs(i);
    y = 10 * (x^4) * cosh(x);
    y_noisy = y + noise(i);
    true_outputs = [true_outputs; y];
    noisy_outputs = [noisy_outputs; y_noisy];
    
    % making regressor vector with fuzzy rules
    mu = exp(-((x - centers)/width).^2)';
    b = mu/sum(mu);
    
    % Calculate fuzzy output
    y_pred = b' * theta;
    
    % Update weights 
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
    fprintf('Step %d: x=%.4f, y_true=%.4f, y_noisy=%.4f, y_pred=%.4f\n , mean_errors=%.4f\n', i, x, y, y_noisy, y_pred, current_error);
end

% plots

% main System and fuzzy modeling system
figure;
plot(inputs, true_outputs, inputs, predictions, 'r--');
legend('True','Fuzzy');
title('main System and fuzzy modeling system');

% Error between true output and prediction
figure;
plot(true_outputs - predictions);
title('Error between True Output and Prediction');

% Trace of P matrix 
figure;
plot(trace_p);
title('Trace of P Matrix');

% Plot all theta 
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

% Mean Error plot
figure;
plot(mean_errors, 'LineWidth', 1.5);
title('Mean Error (True Output vs Prediction)');
xlabel('Step');
ylabel('Mean Error');
grid on;