% Function to generate training data for p-value correction model
function trainingData = generateTrainingData(nPValues, sampleSize, skewnessValues, scaleParam, nPermutations)
    % nPValues: Number of p-values to generate per parameter value
    % sampleSize: Size of each sample (n)
    % skewnessValues: Array of skewness values for Gamma distribution
    % scaleParam: Scale parameter for the gamma distribution (fixed)
    % nPermutations: Number of permutations for permutation test
    
    % Create progress tracking file
    progressUpdateInterval = 100; % Update progress every N iterations
    
    % Initialize storage for training data
    nSkewnessValues = length(skewnessValues);
    totalPairs = nPValues * nSkewnessValues;
    
    % Initialize cell arrays to store results from each skewness parameter
    biasedPValsBySkewness = cell(nSkewnessValues, 1);
    unbiasedPValsBySkewness = cell(nSkewnessValues, 1);
    
    % Parallelize across skewness values
    parfor s = 1:nSkewnessValues
        skewnessValue = skewnessValues(s);
        
        % Convert skewness to shape parameter: Skewness = 2/sqrt(shape), so shape = 4/skewness^2
        % Calculate shape directly inside the loop to avoid broadcast issues
        shapeParam = 4 / (skewnessValue^2);
        
        % Generate p-values for biased distribution (Gamma)
        biasedPValsForSkewness = zeros(nPValues, 1);
        
        % Generate p-values for unbiased distribution (Uniform)
        unbiasedPValsForSkewness = zeros(nPValues, 1);
        
        % Use regular for loop for inner iterations since we've parallelized the outer loop
        for i = 1:nPValues
            % Group 1: all zeros
            group1 = zeros(sampleSize, 1);
            
            % Group 2 for Gamma: Generate and shift to mean=0
            meanGamma = shapeParam * scaleParam;
            gammaData = gamrnd(shapeParam, scaleParam, sampleSize, 1) - meanGamma;
            
            % Group 2 for Uniform: Generate centered at 0 with half-width = scaleParam
            uniformData = unifrnd(-0.5, 0.5, sampleSize, 1);
            
            % Compute p-values using permutation test
            biasedPValsForSkewness(i) = permutationMeanTest(group1, gammaData, nPermutations);
            unbiasedPValsForSkewness(i) = permutationMeanTest(group1, uniformData, nPermutations);
            
            % Log progress every N iterations
            if mod(i, progressUpdateInterval) == 0
                fprintf('  Worker %d: Skewness %.2f (shape %.2f) - Completed %d/%d iterations (%.1f%%)\n', ...
                    labindex, skewnessValue, shapeParam, i, nPValues, i/nPValues*100);
            end
        end
        
        % Sort both sets of p-values
        [biasedPValsSorted, ~] = sort(biasedPValsForSkewness);
        [unbiasedPValsSorted, ~] = sort(unbiasedPValsForSkewness);
        
        % Store the sorted p-values for this skewness
        biasedPValsBySkewness{s} = biasedPValsSorted;
        unbiasedPValsBySkewness{s} = unbiasedPValsSorted;
        
        % Log progress
        fprintf('COMPLETED skewness value %.2f (shape %.2f) (%d of %d)\n', ...
            skewnessValue, shapeParam, s, length(skewnessValues));
        
        % Signal completion by writing to a temporary file
        tmpFile = sprintf('param_done_%d.txt', s);
        fid = fopen(tmpFile, 'w');
        fprintf(fid, '1');
        fclose(fid);
    end
    
    % Combine results from all skewness values
    biasedPValues = zeros(totalPairs, 1);
    unbiasedPValues = zeros(totalPairs, 1);
    skewnessParams = zeros(totalPairs, 1);
    shapeParamsAll = zeros(totalPairs, 1);
    
    pairIdx = 1;
    for s = 1:nSkewnessValues
        % Calculate shape parameter again for storage
        skewnessValue = skewnessValues(s);
        shapeParam = 4 / (skewnessValue^2);
        
        for i = 1:nPValues
            biasedPValues(pairIdx) = biasedPValsBySkewness{s}(i);
            unbiasedPValues(pairIdx) = unbiasedPValsBySkewness{s}(i);
            skewnessParams(pairIdx) = skewnessValue;
            shapeParamsAll(pairIdx) = shapeParam;
            pairIdx = pairIdx + 1;
        end
    end
    
    % Combine into a table for easier handling
    trainingData = table(biasedPValues, unbiasedPValues, skewnessParams, shapeParamsAll, ...
                       'VariableNames', {'BiasedPValue', 'UnbiasedPValue', 'SkewnessParam', 'ShapeParam'});
end

function p = permutationMeanTest(group1, group2, nPermutations)
    % Calculate the observed difference of means.
    obsDiff = mean(group1) - mean(group2);
    
    % Combine both groups.
    combined = [group1; group2];
    n = length(group1);  % assumes equal group sizes
    
    % Preallocate an array for permutation differences.
    permDiff = zeros(nPermutations, 1);
    
    for j = 1:nPermutations
        % Randomly permute the combined data.
        permutedIdx = randperm(length(combined));
        
        % Divide the permuted data into two groups.
        permGroup1 = combined(permutedIdx(1:n));
        permGroup2 = combined(permutedIdx(n+1:end));
        
        % Compute the difference of means for this permutation.
        permDiff(j) = mean(permGroup1) - mean(permGroup2);
    end
    
    % The two-tailed p-value is the fraction of permuted differences (in absolute value)
    % that are at least as large as the observed absolute difference.
    p = mean(abs(permDiff) >= abs(obsDiff));
end