function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
%%%%%%%%%%%%%% PART 1 **********
col_ones = [ones(size(X,1),1)];
% Add ones to first column of X
X=[col_ones, X]; 

%hidden layer
z2 = (Theta1 * X')';
a2 = sigmoid(z2);
col_ones_hidden = [ones(size(a2,1),1)];
a2 = [col_ones_hidden, a2];
%output layer
z3 = (Theta2* a2')';
a3 = sigmoid(z3);

% h = hypothesis output based on Theta
[max_values, h] = max(a3, [], 2);

% Note y==10 is the representation for y equal to 0

cost = 0;
for result=1:num_labels
	y_temp = y==result;
	test = -y_temp'*log(a3(:,result))-(1-y_temp)'*log(1-a3(:,result));
	cost += test;
end

J = 1/m*cost;

% remove first column from Theta 1 and Theta 2
Theta1_onesremoved = Theta1(:,2:size(Theta1,2));
Theta2_onesremoved = Theta2(:,2:size(Theta2,2));

Jreg = lambda/(2*m)*(sum((Theta1_onesremoved.^2)(:))+sum((Theta2_onesremoved.^2)(:)));

% Normalized Cost
J = J+Jreg;

%%%%%%%%%%%%%% PART 2 **********

for t=1:m
	% Take one training example
	% Add ones to first column of X
	train_example=X(t,:); 

	%hidden layer
	z2 = (Theta1 * train_example')';
	a2 = sigmoid(z2);
	a2 = [1, a2];
	%output layer
	z3 = (Theta2* a2')';
	a3 = sigmoid(z3);

	temp = zeros(1,num_labels);
	temp (y(t)) = 1;

	delta3 = a3-temp;
	gprime = a2.*(1-a2);
	delta2= (delta3*Theta2).*gprime;

	delta2 = delta2(2:end);

	Theta2_grad += delta3'*a2;
	Theta1_grad += delta2'*train_example;

end

Theta1_grad /= m;
Theta2_grad /= m;

% regularization
% Compute regularization terms
Theta1_reg = Theta1;
Theta1_reg(:,1) = 0;
Theta1_reg *= lambda/m;
Theta2_reg = Theta2;
Theta2_reg(:,1) = 0;
Theta2_reg *= lambda/m;
Theta1_grad += Theta1_reg;
Theta2_grad += Theta2_reg;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
