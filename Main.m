input_layer = 4 % 3 features, sepal length&width, petal length&width
hidden_layer = 10 % arbitrary amount
output_layer = 3 % 3 classifications, Setosa, Veriscolour, Virginica

Theta1 = randInitializeWeights(4, 5);
Theta2 = randInitializeWeights(5, 3);

nn_params = [Theta1(:); Theta2(:)];

printf("Theta 1\n");
disp(Theta1);

printf("\nTheta 2\n");
disp(Theta2);

printf("\nnn_params\n")
disp(nn_params);
