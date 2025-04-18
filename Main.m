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

% Some Constants
input_layer = 4 % 4 features, sepal length&width, petal length&width
hidden_layer = 120 % arbitrary amount
num_labels = 3 % 3 classifications, Setosa, Veriscolour, Virginica

MAX_GENERATIONS = 1000 % Maximum Generations to go through
TOTAL_POPULATION = 50 % Total Population
TOURNAMENT_SIZE = TOTAL_POPULATION .* 0.5% Tournament size, Lower means more diversity but slower convergence
                    % but higher means that less diversity but faster convergence
MUTATION_CHANCE = 0.01 % Mutation chance. 

% GENERATE POPULATION
pops = generatePopulation(TOTAL_POPULATION, input_layer, hidden_layer, num_labels);

optimal_weights = [];

minFitness = ones(MAX_GENERATIONS, 1);

% FITNESS EVALUATION
for g = 1: MAX_GENERATIONS
    new_pops = cell(TOTAL_POPULATION, 1);
    fitness = ones(TOTAL_POPULATION, 1);
    for i = 1: TOTAL_POPULATION
        nn_params = pops{i};
        fitness(i) = nnCostFunction(nn_params, input_layer, hidden_layer, num_labels, X, y, 1);
    end

    % sort the population according to fitness, in ascending order
    [sorted_fitness, sorted_indices] = sort(fitness, 'ascend');
    pops = pops(sorted_indices);

    % get the current minimum and set that as the optimal weight
    optimal_weights = pops{1};
    minFitness(g) = fitness(1);

    printf("Generation %d Min Fitness: %d\n", g, fitness(1))

    % stop if fitness at that minimum is already 0
    if (fitness(1) == 0)
        break;
    endif

    % crossover using tournament selection
    for j = 1: 2: TOTAL_POPULATION
        p1 = tournament_selection(pops, fitness, TOURNAMENT_SIZE);
        p2 = tournament_selection(pops, fitness, TOURNAMENT_SIZE);
        [c1, c2] = crossover(p1, p2);
        
        
        % apply mutation
        c1 = mutation(c1, MUTATION_CHANCE);
        c2 = mutation(c2, MUTATION_CHANCE);

        % get child fitness
        fitc1 = nnCostFunction(c1, input_layer, hidden_layer, num_labels, X, y, 1);
        fitc2 = nnCostFunction(c2, input_layer, hidden_layer, num_labels, X, y, 1);
        % printf("Child 1 Fitness: %d Child 2 Fitness: %d Lowest Fitness: %d\n", fitc1, fitc2, fitness(TOTAL_POPULATION))

        %insert children to new pops
        if (fitc1 < fitness(TOTAL_POPULATION))
            new_pops(j) = c1;
        else
            new_pops(j) = pops(j);
        endif
        if (j+1 <= TOTAL_POPULATION)
            if (fitc2 < fitness(TOTAL_POPULATION))
                new_pops(j+1) = c2;
            else
                new_pops(j+1) = pops(j+1);
            endif
        endif
    end
    pops = new_pops;
end

p = plot (minFitness);
xlabel ("Generation");
ylabel ("Min Fitness");
title ("Genetic Algorithm");
waitfor(p);

Theta1 = reshape(optimal_weights(1:hidden_layer * (input_layer + 1)), ...
                 hidden_layer, (input_layer + 1));

Theta2 = reshape(optimal_weights((1 + (hidden_layer * (input_layer + 1))):end), ...
                 num_labels, (hidden_layer + 1));

data = dlmread('iris_testing.data', ',', 0, 0);  % Read only numeric data
testing_data = data(:, 1:4); %This is the testing data

fid = fopen('iris_testing.data', 'r');
labels = textscan(fid, '%*f %*f %*f %*f %s', 'Delimiter', ',');
fclose(fid);

testing_labels = labels{1};

class_labels = {'Iris-setosa', 'Iris-versicolor', 'Iris-virginica'};
[~, testing_labels] = ismember(testing_labels, class_labels); % this is the outcome for each of the training data

result = predict(Theta1, Theta2, testing_data);
disp("Predicted Results: ");
disp(result);
disp("Actual Results: ");
disp(testing_labels);
training_acc = mean(double(result == testing_labels(1:length(result), 1))) * 100;
fprintf('Training Accuracy: %.2f%%\n', training_acc);
% J = nnCostFunction(nn_params, input_layer, hidden_layer, num_labels, X, y, 1); %Cost computation of with weights Theta1 and Theta2
% disp(J);
