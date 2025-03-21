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
hidden_layer = 10 % arbitrary amount
num_labels = 3 % 3 classifications, Setosa, Veriscolour, Virginica

MAX_GENERATIONS = 100 % Maximum Generations to go through
TOTAL_POPULATION = 100 % Total Population
TOURNAMENT_SIZE = 2 % Tournament size, Lower means more diversity but slower convergence
                    % but higher means that less diversity but faster convergence
MUTATION_CHANCE = 0.25 % Mutation chance. 

% GENERATE POPULATION
pops = generatePopulation(TOTAL_POPULATION);

optimal_weights = [];


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
    optimal_weights = pops{i};
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

        %insert children to new pops
        new_pops(j) = c1;
        if (j+1 <= TOTAL_POPULATION)
            new_pops(j+1) = c2;
        endif
    end
    pops = new_pops;
end




% J = nnCostFunction(nn_params, input_layer, hidden_layer, num_labels, X, y, 1); %Cost computation of with weights Theta1 and Theta2
% disp(J);
