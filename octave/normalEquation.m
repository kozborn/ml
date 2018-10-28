function thetas = normalEquation(X,y)

  % θ = (X' * X) ^−1 * X'* y

  thetas = pinv(X' * X) * X' * y



