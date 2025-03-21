function pops = generatePopulation(num_pops)
%GENERATEPOPULATION generates a variable number of populations (dependent on num_pops)
%and outputs it as cell array of weight matrices

input_layer = 4 % 4 features, sepal length&width, petal length&width
hidden_layer = 10 % arbitrary amount
num_labels = 3 % 3 classifications, Setosa, Veriscolour, Virginica

pops = cell(num_pops);

for i = 1:num_pops
  %Random generation of weights
  Theta1 = randInitializeWeights(input_layer, hidden_layer); %Theta1 are the weights between input layer and hidden layer
  Theta2 = randInitializeWeights(hidden_layer, num_labels); %Theta2 are the weights between hidden layer and output layer
  nn_params = [Theta1(:); Theta2(:)]; % combine them into one (for use later for the cost function)
  pops[i] = nn_params; % full pops and stuff ye
end

end