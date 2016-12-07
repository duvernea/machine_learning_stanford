function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% test values array for C and sigma
test_values = [.01, .03, .1, .3, 1, 3, 10, 30];

% Set prediction error  starting point above
prediction_error_final = 1000;


for C = test_values
	for sigma = test_values
		% Train model using training set and particular value of C and sigma:
		model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
		% Make predictions with this model
		predictions = svmPredict(model, Xval);
		prediction_error = mean(double(predictions ~= yval));
		if (prediction_error < prediction_error_final)
			%printf("Found a new best case");
			prediction_error_final = prediction_error;
			C_best = C;
			sigma_best = sigma;
		end
		%printf(strcat("Prediction error: C: ", num2str(C), " Sigma: ", num2str(sigma), " = ", num2str(prediction_error)));
	end
end

C= C_best;
sigma = sigma_best;

% =========================================================================

end
