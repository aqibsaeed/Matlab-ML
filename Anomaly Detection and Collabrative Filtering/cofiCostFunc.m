function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));
nm = size(R,1);
nu = size(R,2);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%
    
  for i = 1:nu
      u_rated = find(R(:,i) == 1);
      u_rating = Y(u_rated,i);
      u_theta = Theta(i,:);
      prediction = X(u_rated,:) * u_theta';
      diff = (prediction - u_rating);
      h = (1/2) * sum((diff).^2);
      regularization = (lambda/2) * (sum(u_theta .^ 2));
      h += regularization;

      J += h;
      Theta_grad(i,:) = (diff' * X(u_rated,:)) + (lambda * u_theta);
   end
    
   for j = 1:nm
      rated_idx = find(R(j,:) == 1);
      rating = Y(j,rated_idx);
      rated_theta = Theta(rated_idx,:);
      diff = (X(j,:) * rated_theta') - rating;
      regularization = (lambda/2) * (sum((X(j,:).^2)));
      J += regularization;
      
      X_grad(j,:) = (diff * rated_theta) + (lambda * X(j,:));
   end 
    
    
    
    
% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
