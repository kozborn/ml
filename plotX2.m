function p = plotX2(X)
  xmax = max(X(:, 1))
  ymax = max(X(:, 2))

  plot(X(:,1), X(:, 2), '+', 'color', 'r', 'markersize', 10,  xmax + 50, ymax+100)