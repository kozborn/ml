% X =
%   1   2   3
%   1   4   6

% Y =
%   1
%   0


% theta =
%   1
%   1
%   1


function J = logisticCostFunction(X, y, theta)
% X is the 'design matrix' containing our training example
% y is the class labels
X 
y
theta
m = size(X,1)
h = X*theta
g = 1  ./ (1 + exp(-h))
J = 1/m * ((-y') * log(g) - (1 - y)' * log( 1 - g))
