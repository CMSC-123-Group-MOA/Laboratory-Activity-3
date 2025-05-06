function velocity = updateVelocity(current_velocity, position, personal_best, global_best, w, c1, c2)
  % Inputs:
  % current_velocity, position, personal_best: 1xN cell arrays of 1xD vectors
  % global_best: 1xD vector (not a cell)
  % w: inertia weight
  % c1, c2: cognitive and social coefficients

  N = numel(position); % Number of particles

  for i = 1:N
    % Generate random vectors of the same size as the particle
    D = numel(position{i});
    r1 = rand(1, D);
    r2 = rand(1, D);

    % Velocity update rule (element-wise operations)
    inertia    = w  * current_velocity{i};
    cognitive  = c1 * r1 * (personal_best{i} - position{i});
    social     = c2 * r2 * (global_best - position{i});

    velocity{i} = inertia + cognitive + social;
  end
end

