function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
ncols = size(X)(1,2);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

predict = sigmoid(X*theta);
error = -y .* (log(predict)) - (1 - y) .* (log(1-predict));
J = (1/m) * sum(error);


for j = 1:ncols
   
  error = (predict - y) .* X(:,j);
  grad(j,1) = (1/m) * sum(error);
 
end

J;
grad';

% =============================================================

end
