function mutated = mutation(pop, mutation_chance)
%MUTATION mutates the given pop given a mutation chance.

mutated = pop;
for i = 1: length(pop)
    if rand() < mutation_chance
        mutated(i) = rand() * 2 * 0.12 - 0.12;
    end
end

end