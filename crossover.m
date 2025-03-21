function [child1, child2] = crossover(nn_params_1, nn_params_2)
%CROSSOVER just does a crossover

mask = randi([0,1], length(nn_params_1), 1); % mask vector column full of 0s and 1s
child1 = mask .* nn_params_1 + (1 - mask) .* nn_params_2; % apply mask to get child1
mask = randi([0,1], length(nn_params_1), 1); % mask vector column full of 0s and 1s
child2 = mask .* nn_params_1 + (1 - mask) .* nn_params_2; % apply mask to get child1

end