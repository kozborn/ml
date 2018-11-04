function plotMultiset(X, y)

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%
load coursera-features-set1.txt
load multiclass-coursera-results-set.txt

x = coursera_features_set1;
y = multiclass_coursera_results_set;

thetas1 = [-11.1299, 0.239069, -0.132743 ];
thetas2 = [-11.5316 -0.12481 0.236158];
thetas3 = [3.99663 -0.0342845 -0.0256397];

one = find(y == 1); twos = find(y == 2); threes=find(y==3);
plot(x(one, 1), x(one, 2), 'k+', 'LineWidth', 2, 'MarkerSize', 7);
plot(x(twos, 1), x(twos, 2), 'ko', 'MarkerFaceColor', 'r', 'MarkerSize', 7);
plot(x(threes, 1), x(threes, 2), 'k*', 'MarkerFaceColor', 'g', 'MarkerSize', 7);

plot_x = [min(x(:,2))-2,  max(x(:,2))+2]
plot_y = (-1./thetas1(3)).*(thetas1(2).*plot_x + thetas1(1));

plot(plot_x, plot_y, 'linewidth', 1)
plot_y = (-1./thetas2(3)).*(thetas2(2).*plot_x + thetas2(1));
plot(plot_x, plot_y, 'color', 'r', 'linewidth', 1)
plot_y = (-1./thetas3(3)).*(thetas3(2).*plot_x + thetas3(1));
plot(plot_x, plot_y, 'color', 'g', 'linewidth', 1)
% =========================================================================



hold off;

end