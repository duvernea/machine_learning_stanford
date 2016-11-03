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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


% compute hypothesis
h = sigmoid(X*theta);
% compute number of training set samples
m=size (X,1);
% compute regularization
%theta, without theta0
thetasize = size(theta,1);
reg = lambda/(2*m) * sum(theta(2:thetasize).^2);
% compute cost function
J = 1/m.*(-y'* log (h) - (1 - y)'*log(1-h))+reg;
% gradient function
reg_gradient_term = [0; lambda/m*(theta(2:thetasize))];
grad = 1/m.*X'*(h-y)+reg_gradient_term;



% =============================================================

end
