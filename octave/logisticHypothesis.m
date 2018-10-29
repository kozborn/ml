function g = logisticHypothesis(X, theta)
  h = X*theta;
  g = 1  ./ (1 + exp(-h));

