function new_position = updatePosition(position, velocity)
  % Inputs:
  % position, velocity: 1xN cell arrays of 1xD vectors
  % Output:
  % new_position: 1xN cell array of updated positions

  N = numel(position);
  new_position = cell(1, N);

  for i = 1:N
    new_position{i} = position{i} + velocity{i};
  end
end

