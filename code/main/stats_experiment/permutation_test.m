function p = permutation_test(group1, group2, nPermutations)
% Performs a permutation test between two groups and returns the p-value
% Highly optimized version for maximum speed
%
% Parameters:
%   group1 - First group of values
%   group2 - Second group of values
%   nPermutations - Number of permutations to run
%
% Returns:
%   p - p-value from the permutation test

obsDiff = mean(group1) - mean(group2);
n1 = length(group1);
n2 = length(group2);
n = n1 + n2;

% PERFORMANCE: Direct computation of mean differences
combined = [group1; group2];
permCount = 0;

% PERFORMANCE: Randomized permutation indices
for j = 1:nPermutations
    permutedIdx = randperm(n);
    idx1 = permutedIdx(1:n1);
    idx2 = permutedIdx(n1+1:end);
    
    % PERFORMANCE: Vectorized mean calculation
    permDiff = mean(combined(idx1)) - mean(combined(idx2));
    
    % Count permutations with difference >= observed
    if abs(permDiff) >= abs(obsDiff)
        permCount = permCount + 1;
    end
end

p = permCount / nPermutations;

% Ensure p-value is not zero (important for later model fitting)
if p == 0
    p = 1 / (2 * nPermutations);
end
end