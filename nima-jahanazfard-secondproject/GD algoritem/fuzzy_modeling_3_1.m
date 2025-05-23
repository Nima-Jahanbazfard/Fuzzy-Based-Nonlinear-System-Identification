
clc
clear all
close all

%enter input threshold
xmin=input('enter down bound of input:');
xmax=input('enter up bound of input:');

% Generate sample data for y = 10x^4*cosh(x)
x = linspace(xmin, xmax, 100)';
y = (10.*(x.^4).*((exp(x)+exp(-1.*x))./2)) ;
% Training parameters
M = input('enter mont of fuzzy membership functions:');       
alpha = input('enter learning rate:');   
max_epochs = input('enter mont of epochs:');
epsilon = input('enter desirable precision:');
 

% Train fuzzy system
fuzzy_sys = fuzzy_modeling_3(x, y, M, alpha, max_epochs, epsilon);

% Test the system
y_pred = arrayfun(@(xi) fuzzy_sys.evaluate(xi), x);

% Plot results

figure;
plot(x, y, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Actual Function');
grid on
figure,plot(x, y_pred, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Fuzzy Approximation');
grid on
title('Fuzzy System Approximation of y = 10x^4 cosh(x)');
figure,plot(x, y_pred, 'r--',x, y, 'b-');
grid on
xlabel('x');
ylabel('y');
legend();
figure,plot(x, y-y_pred, 'g', 'LineWidth', 2, 'DisplayName', 'error');
grid on
title('error');
