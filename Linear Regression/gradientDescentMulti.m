function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
ncols = size(X)(1,2);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    sum_errors = zeros(ncols,1);
    predict = X*theta;
    error = predict - y;
    
    for j = 1:ncols
      sum_error = error .* X(:,j);
      sum_errors(j,1) = sum(sum_error);
    end
    
    theta = theta - ((alpha*(1/m)) .* sum_errors);
    theta;

    % ============================================================

    % Save the cost J in every iteration    
    cc = computeCostMulti(X, y, theta);
    J_history(iter) = cc;

end

end
