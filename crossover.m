function child = crossover(nn_params_1, nn_params_2)
%CROSSOVER just does a crossover

mask = randi([0,1], size(nn_params_1)) % mask vector full of 0s and 1s

child = mask .* nn_params_1 + (1 - mask) .* nn_params_2

end