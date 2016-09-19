function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
nrows = size(theta)(1,1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

predict = sigmoid(X*theta);
error = -y .* (log(predict)) - (1 - y) .* (log(1-predict));

% not regularizing theta0 term
reg = (lambda/(2*m)) * sum((theta(2:nrows,1).^ 2));
J = ((1/m) * sum(error)) + reg;

% gardient without regulatization for theta0 term
er = (predict - y) .* X(:,1);
grad(1,1) = ((1/m) * sum(er));

for j = 2:nrows
   
  er = (predict - y) .* X(:,j);
  grad(j,1) = ((1/m) * sum(er)) + ((lambda/m) * theta(j,1));
  
end

J;
grad;

% =============================================================

end
