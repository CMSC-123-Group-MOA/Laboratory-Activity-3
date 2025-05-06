% Load data
data = dlmread('cancer_training.data', ',', 0, 0);  % Read only numeric data

% Extract features 
X = data(:, 2:10); %This is the training data
y = data(:, 11:11) %This is traning labels
% Display first few rows to verify
% disp(X);
% disp(y);

% Some Constants
input_layer = 9 
hidden_layer = 120 % arbitrary amount
num_labels = 2 % 2 classifications 2 or 4

MAX_GENERATIONS = 100 % Maximum Generations to go through
TOTAL_POPULATION = 100 % Total Population

% GENERATE POPULATION
pops_position = generatePopulation(TOTAL_POPULATION, input_layer, hidden_layer, num_labels);
pops_velocity= generatePopulation(TOTAL_POPULATION, input_layer, hidden_layer, num_labels);
best_personal = pops_position;
best_global = pops_position{1};

optimal_weights = [];
minFitness = ones(MAX_GENERATIONS, 1);

fid1 = fopen('results.txt', 'w');

% Inital fitness evaluation
for i = 1: TOTAL_POPULATION
    nn_params = pops_position{i};
    best_personal_fitness = nnCostFunction(best_personal{i}, input_layer, hidden_layer, num_labels, X, y, 1);
    best_global_fitness = nnCostFunction(best_global, input_layer, hidden_layer, num_labels, X, y, 1);

    if (best_personal_fitness < best_global_fitness)
        best_global = best_personal{i};
    endif

    printf("particle %d - Personal Fitness: %d\n", i, best_personal_fitness);
end

%disp(pops_velocity);
%disp(new_velocity(pops_velocity, pops_position, best_personal, best_global, 1, 2, 2))

current_fitness = nnCostFunction(pops_position{1} + pops_velocity{1}, input_layer, hidden_layer, num_labels, X, y, 1);
disp(current_fitness);
% FITNESS EVALUATION
for g = 1: MAX_GENERATIONS
    pops_position = updatePosition(pops_position, pops_velocity);

    for i = 1: TOTAL_POPULATION
        current_fitness = nnCostFunction(pops_position{i}, input_layer, hidden_layer, num_labels, X, y, 1);
        best_personal_fitness = nnCostFunction(best_personal{i}, input_layer, hidden_layer, num_labels, X, y, 1);
        global_fitness = nnCostFunction(best_global, input_layer, hidden_layer, num_labels, X, y, 1);

        if (current_fitness < best_personal_fitness)
            best_personal{i} = pops_position{i};
        endif 

        if (current_fitness < global_fitness)
            best_global = pops_position{i};
        endif 
    end

    pops_velocity = updateVelocity(pops_velocity, pops_position, best_personal, best_global, 1, 2, 2);
    global_fitness = nnCostFunction(best_global, input_layer, hidden_layer, num_labels, X, y, 1);

    printf("Generation %d Global Fitness: %d\n", g, global_fitness);

    % sort the population according to fitness, in ascending order
    % [sorted_fitness, sorted_indices] = sort(fitness, 'ascend');
    % pops = pops(sorted_indices);

    % get the current minimum and set that as the optimal weight
    % optimal_weights = pops{1};
    % minFitness(g) = fitness(1);

    % printf("Generation %d Min Fitness: %d\n", g, fitness(1))

    % stop if fitness at that minimum is already 0
    % if (fitness(1) == 0)
    %     break;
    % endif

end

%xlabel ("Generation");
%p = plot (minFitness);
%ylabel ("Min Fitness");
%title ("Genetic Algorithm");
%waitfor(p);

Theta1 = reshape(optimal_weights(1:hidden_layer * (input_layer + 1)), ...
                 hidden_layer, (input_layer + 1));

Theta2 = reshape(optimal_weights((1 + (hidden_layer * (input_layer + 1))):end), ...
                 num_labels, (hidden_layer + 1));

data = dlmread('cancer_testing.data', ',', 0, 0);  % Read only numeric data
testing_data = data(:, 2:10); %This is the testing data
testing_labels = data(:, 11:11); %This is the testing data

result = predict(Theta1, Theta2, testing_data);
disp("Predicted Results: ");
%disp(result);
disp("Actual Results: ");
%disp(testing_labels);
training_acc = mean(double(result == testing_labels(1:length(result), 1))) * 100;
fprintf('Training Accuracy: %.2f%%\n', training_acc);
% J = nnCostFunction(nn_params, input_layer, hidden_layer, num_labels, X, y, 1); %Cost computation of with weights Theta1 and Theta2
% disp(J);

fid1 = fopen('logs.txt', 'a');

%fprintf(fid1, 'input layers: %d, hidden layers: %d, num labels: %d, generations: %d, population: %d, tournament: %d, mutation: %d, accuracy: %.2f\n', input_layer, hidden_layer, num_labels, MAX_GENERATIONS, TOTAL_POPULATION, TOURNAMENT_SIZE, MUTATION_CHANCE, training_acc);

fclose(fid1);

