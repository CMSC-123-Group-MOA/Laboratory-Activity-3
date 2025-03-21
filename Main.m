% Load data
data = dlmread('iris_training.data', ',', 0, 0);  % Read only numeric data

% Extract features (first 4 columns, excluding the classification on t he 5th column)
X = data(:, 1:4); %This is the training data



% Read labels separately (since they are strings)
fid = fopen('iris_training.data', 'r');
labels = textscan(fid, '%*f %*f %*f %*f %s', 'Delimiter', ',');
fclose(fid);

y = labels{1};

class_labels = {'Iris-setosa', 'Iris-versicolor', 'Iris-virginica'};
[~, y] = ismember(y, class_labels); % this is the outcome for each of the training data
% Display first few rows to verify
% disp(X);
% disp(y);


input_layer = 4 % 4 features, sepal length&width, petal length&width
hidden_layer = 10 % arbitrary amount
num_labels = 3 % 3 classifications, Setosa, Veriscolour, Virginica



