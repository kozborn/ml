function J = costFunction(X, y, theta)

% X is the "design matrix" containing out training example
% y is the class labels

m = size(X,1);
predictions = X * theta;
sqrErrors = (predictions-y).^2;
J = (1/(2*m)) * sum(sqrErrors);


