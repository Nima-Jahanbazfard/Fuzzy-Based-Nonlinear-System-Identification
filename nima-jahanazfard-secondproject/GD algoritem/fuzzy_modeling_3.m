function fuzzy_system = fuzzy_modeling_3(input_data, output_data, M, alpha, max_epochs, epsilon)
    
    [num_samples, n] = size(input_data);
    
    
    % Centers of input membership functions
    x_bar = zeros(n, M);
    for i = 1:n
        x_bar(i,:) = linspace(min(input_data(:,i)), max(input_data(:,i)), M);
    end
    
    % Widths of input membership functions
    sigma = zeros(n, M);
    for i = 1:n
        sigma(i,:) = (max(input_data(:,i)) - min(input_data(:,i))) / M * 1.5; % Wider coverage
    end
    
    % Centers of output membership functions
    y_bar = linspace(min(output_data), max(output_data), M);
    
    % Training loop max_epoch=q according to wang book
    for epoch = 1:max_epochs
        total_error = 0;
        
        for p = 1:num_samples
            x0 = input_data(p,:)';  
            y0 = output_data(p);   
           
            % Calculate firing strengths (z^l)
            z = ones(1, M);
            for l = 1:M
                for i = 1:n
                    z(l) = z(l) * exp(-((x0(i) - x_bar(i,l)) / sigma(i,l))^2);
                end
            end
            
            % calculating output
            b = sum(z);
            a = sum(y_bar .* z);
            f = a / b;  % Final output
            
            % Error calculation 
            e = 0.5 * (f - y0)^2;
            total_error = total_error + e;
            
            
            % Update y_bar
            for l = 1:M
                y_bar(l) = y_bar(l) - alpha * (f - y0) * (1 / b) * z(l);
            end
            
            % Update x_bar and sigma 
            for l = 1:M
                for i = 1:n
                    
                    % Update x_bar 
                    x_bar(i,l) = x_bar(i,l) - alpha * ...
                    ((f - y0) * (y_bar(l) - f) * z(l) / b) * ...
                                2*(x0(i) - x_bar(i,l)) / (sigma(i,l)^2);
                    
                    % Update sigma 
                    sigma(i,l) = sigma(i,l) - alpha *... 
                    ((f - y0) * (y_bar(l) - f) * z(l) / b) * ...
                                2*(x0(i) - x_bar(i,l))^2 / (sigma(i,l)^3);
                end
            end
        end
        
        fprintf('Epoch %d: Error = %.4f\n', epoch, total_error);
        
        % Early stopping if error is below threshold
        if total_error < epsilon
            break;
        end
    end
    
    % Store final parameters
    fuzzy_system = struct();
    fuzzy_system.M = M;
    fuzzy_system.x_bar = x_bar;
    fuzzy_system.sigma = sigma;
    fuzzy_system.y_bar = y_bar;
    
    % Evaluation function
    fuzzy_system.evaluate = @(x) evaluate_fuzzy_system(x, x_bar, sigma, y_bar);
end

function y = evaluate_fuzzy_system(x, x_bar, sigma, y_bar)
    % Evaluate fuzzy system for input x
    [n, M] = size(x_bar);
    z = ones(1, M);
    
    % Calculate firing strengths
    for l = 1:M
        for i = 1:n
            z(l) = z(l) * exp(-((x(i) - x_bar(i,l)) / sigma(i,l))^2);
        end
    end
    
    % Compute output
    b = sum(z);
    a = sum(y_bar .* z);
    y = a / b;
end


