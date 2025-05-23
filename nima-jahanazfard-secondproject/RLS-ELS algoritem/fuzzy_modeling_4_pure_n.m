clc
clear all
close all

% User Inputs
min_input = input('Enter lower bound of input: ');       
max_input = input('Enter upper bound of input: ');       
num_rules = input('Enter number of rules: '); 
learning_rate = input('Enter learning rate for P matrix: '); 
num_samples = input('Enter number of input samples: ');   

% Initialize Fuzzy System 
centers = linspace(min_input, max_input, num_rules);     
width = (max_input - min_input) / num_rules;             
theta = 20 * eye(num_rules, 1);                        
P_matrix = learning_rate * eye(num_rules);              

% Storage 
predictions = [];        
errors = [];             
true_outputs = [];       
inputs = linspace(min_input, max_input, num_samples)';  
trace_p = [];            
mean_errors = []; 
theta_history = zeros(num_samples, num_rules); 

for i = 1:num_samples
    
    x = inputs(i);
    y = 10 * (x^4) * cosh(x);
    true_outputs = [true_outputs; y];
    
    % making regressor vector with fuzzy rules
    mu = exp(-((x - centers)/width).^2)';
    b = mu / sum(mu);
    
    % Fuzzy system prediction
    y_pred = b' * theta;
    
    % Weight Update
    K = P_matrix * b / (1 + b' * P_matrix * b);  
    theta = theta + K * (y - y_pred);        
    P_matrix = P_matrix - K * b' * P_matrix;     
    trace_p = [trace_p; trace(P_matrix)]; 

    % Store all theta values
    theta_history(i, :) = theta';

    % Store results
    predictions = [predictions; y_pred];
    errors = [errors; (y - y_pred)];
    current_mean_error = mean(errors(1:i));
    mean_errors = [mean_errors; current_mean_error];
    
    % Display progress
    fprintf('Step %d: x=%.4f, y_true=%.4f, y_pred=%.4f, Mean Error=%.4f\n', i, x, y, y_pred, current_mean_error);
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

% Mean Error plot
figure;
plot(mean_errors, 'LineWidth', 1.5);
title('Mean Absolute Error (True Output vs Prediction)');
xlabel('Step');
ylabel('Mean Error');
grid on;
