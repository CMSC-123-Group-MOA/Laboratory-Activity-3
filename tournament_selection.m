function selected = tournament_selection(population, fitness, tournament_size)
%TOURNAMENT_SELECTION This is a tournament selection function that selects
%based on a tournament style format given a tournament size.

total_pops = length(population);
tournament_indices = randperm(total_pops, tournament_size);
[~, best_idx] = min(fitness(tournament_indices));
selected = population{tournament_indices(best_idx)};

end