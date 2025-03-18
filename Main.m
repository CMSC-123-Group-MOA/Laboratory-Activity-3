% Load data from the .data file
data = dlmread('iris_training.data', ',', 0, 0);  % Read only numeric data

% Extract features (first 4 columns)
X = data(:, 1:4);

% Read labels separately (since they are strings)
fid = fopen('iris_training.data', 'r');  
labels = textscan(fid, '%*f %*f %*f %*f %s', 'Delimiter', ',');  
fclose(fid);

y = labels{1};

class_labels = {'Iris-setosa', 'Iris-versicolor', 'Iris-virginica'};
[~, y] = ismember(y, class_labels);
% Display first few rows to verify
disp(X);
disp(y);


input_layer = 4 % 4 features, sepal length&width, petal length&width
hidden_layer = 10 % arbitrary amount
num_labels = 3 % 3 classifications, Setosa, Veriscolour, Virginica

Theta1 = randInitializeWeights(input_layer, hidden_layer);
Theta2 = randInitializeWeights(hidden_layer, num_labels);

nn_params = [Theta1(:); Theta2(:)];

J = nnCostFunction(nn_params, input_layer, hidden_layer, num_labels, X, y, 1);
disp(J);
