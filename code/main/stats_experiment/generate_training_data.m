function trainingData = generate_training_data(maxIterations, sampleSize, nPermutations)
% Generates training data for a fixed number of iterations rather than targeting
% a specific number of p-values. Optimized for high-performance systems.
%
% Parameters:
%   maxIterations - Maximum number of iterations to run
%   sampleSize - Sample size per group
%   nPermutations - Number of permutations for permutation test
%
% Returns:
%   trainingData - Table containing all generated p-values and parameters

% Define skewness bins for stratification in the realistic Bowley's range
skewnessBins = [-0.95, -0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95];
numBins = length(skewnessBins) - 1;

% Set a reasonable capacity estimate for arrays - this is just an estimate
estimatedSamples = maxIterations * 50; % Assuming ~50 valid samples per iteration
initialCapacity = estimatedSamples * 1.2; % Add 20% buffer

% Initialize storage with pre-allocated arrays
biasedPValues = zeros(initialCapacity, 1);
unbiasedPValues = zeros(initialCapacity, 1);
measuredMeans = zeros(initialCapacity, 1);
measuredVariances = zeros(initialCapacity, 1);
measuredSkewness = zeros(initialCapacity, 1);
measuredKurtosis = zeros(initialCapacity, 1);
measuredBowleySkewness = zeros(initialCapacity, 1);
shapeParams = zeros(initialCapacity, 1);
distributionTypes = cell(initialCapacity, 1);

% Initialize bin counts
currentCountPerBin = zeros(numBins, 1);
totalCollected = 0;

% Start tracking time
startTime = tic;
lastPrintTime = tic;

% Start parallel pool with maximum workers
if isempty(gcp('nocreate'))
    c = parcluster('local');
    c.NumWorkers = 24; % Use all 24 processors
    saveProfile(c);
    poolobj = parpool('local', 23); % Use 23 workers, leaving 1 for the main process
    
    % Set thread pool size for maximum performance
    poolobj.SpecifyObjectSharingBehavior = false;
    maxNumCompThreads(1); % Limit the main thread to 1 computational thread
end

% Print header for progress tracking
fprintf('=============================================================\n');
fprintf('ITERATION-LIMITED DATA GENERATION (MAXIMUM %d ITERATIONS)\n', maxIterations);
fprintf('=============================================================\n');
fprintf('Max iterations: %d\n', maxIterations);
fprintf('Sample size per group: %d\n', sampleSize);
fprintf('Permutations per test: %d\n', nPermutations);
fprintf('=============================================================\n\n');

% Storage for tracking progress
progressData = struct();
progressData.iterations = [];
progressData.samplesCollected = [];
progressData.binCounts = [];
progressData.elapsedTime = [];

% PERFORMANCE: Use batch processing to minimize overhead
iterations = 0;
batchSize = 500; % Process more samples per batch for better throughput

% Keep track of total permutation tests for reporting
totalPermTests = 0;

while iterations < maxIterations
    iterations = iterations + 1;
    
    % PERFORMANCE: Use larger batches for better parallel efficiency
    batchBiased = zeros(batchSize, 1);
    batchUnbiased = zeros(batchSize, 1);
    batchMeans = zeros(batchSize, 1);
    batchVariances = zeros(batchSize, 1);
    batchSkewness = zeros(batchSize, 1);
    batchKurtosis = zeros(batchSize, 1);
    batchBowleySkewness = zeros(batchSize, 1);
    batchShapeParams = zeros(batchSize, 1);
    batchDistTypes = cell(batchSize, 1);
    
    % PERFORMANCE: Pre-allocate uniform data for all samples at once
    uniformData = rand(sampleSize, batchSize) - 0.5; % Range [-0.5, 0.5]
    
    % PERFORMANCE: Generate gamma shape parameters for the whole batch
    % Using a wider range to get a broader variety of skewness values
    gammaShapeParams = exp(unifrnd(log(0.1), log(20), batchSize, 1));
    
    % PERFORMANCE: Process the batch in parallel with optimized workload
    parfor i = 1:batchSize
        % PERFORMANCE: Create group1 as zeros array
        group1 = zeros(sampleSize, 1);
        
        % Use gamma distribution for all samples
        shapeParam = gammaShapeParams(i);
        scaleParam = 1.0;
        
        % Generate gamma data centered at mean
        meanGamma = shapeParam * scaleParam;
        group2 = gamrnd(shapeParam, scaleParam, sampleSize, 1) - meanGamma;
        
        % Store parameter and distribution type
        batchShapeParams(i) = shapeParam;
        batchDistTypes{i} = 'gamma';
        
        % Calculate all moments
        moments = calculate_moments(group2);
        batchMeans(i) = moments.mean;
        batchVariances(i) = moments.variance;
        batchSkewness(i) = moments.skewness;
        batchKurtosis(i) = moments.kurtosis;
        batchBowleySkewness(i) = moments.bowleySkewness;
        
        % PERFORMANCE: Optimized permutation test
        batchBiased(i) = permutation_test(group1, group2, nPermutations);
        batchUnbiased(i) = permutation_test(group1, uniformData(:, i), nPermutations);
    end
    
    % Update total permutation test count
    totalPermTests = totalPermTests + 2 * batchSize;
    
    % Assign each sample to appropriate skewness bin
    samplesAddedThisIteration = 0;
    
    for i = 1:batchSize
        % Find which bin this sample belongs to
        binIdx = find(batchBowleySkewness(i) >= skewnessBins(1:end-1) & ...
                     batchBowleySkewness(i) < skewnessBins(2:end), 1);
        
        % Add the sample if it falls in a valid bin
        if ~isempty(binIdx)
            % Check if we need to resize arrays (safety check)
            if totalCollected + 1 > length(biasedPValues)
                % Increase capacity by 50%
                newCapacity = round(length(biasedPValues) * 1.5);
                biasedPValues(end+1:newCapacity) = 0;
                unbiasedPValues(end+1:newCapacity) = 0;
                measuredMeans(end+1:newCapacity) = 0;
                measuredVariances(end+1:newCapacity) = 0;
                measuredSkewness(end+1:newCapacity) = 0;
                measuredKurtosis(end+1:newCapacity) = 0;
                measuredBowleySkewness(end+1:newCapacity) = 0;
                shapeParams(end+1:newCapacity) = 0;
                distributionTypes(end+1:newCapacity) = {''};
            end
            
            % Increment counts
            totalCollected = totalCollected + 1;
            currentCountPerBin(binIdx) = currentCountPerBin(binIdx) + 1;
            samplesAddedThisIteration = samplesAddedThisIteration + 1;
            
            % Store the data
            biasedPValues(totalCollected) = batchBiased(i);
            unbiasedPValues(totalCollected) = batchUnbiased(i);
            measuredMeans(totalCollected) = batchMeans(i);
            measuredVariances(totalCollected) = batchVariances(i);
            measuredSkewness(totalCollected) = batchSkewness(i);
            measuredKurtosis(totalCollected) = batchKurtosis(i);
            measuredBowleySkewness(totalCollected) = batchBowleySkewness(i);
            shapeParams(totalCollected) = batchShapeParams(i);
            distributionTypes{totalCollected} = batchDistTypes{i};
        end
    end
    
    % Store progress data
    if mod(iterations, 5) == 0 || iterations == 1
        progressData.iterations(end+1) = iterations;
        progressData.samplesCollected(end+1) = totalCollected;
        progressData.binCounts(end+1,:) = currentCountPerBin';
        progressData.elapsedTime(end+1) = toc(startTime);
    end
    
    % Print progress report periodically
    currentTime = toc(startTime);
    if toc(lastPrintTime) >= 5 || iterations == maxIterations  % Every 5 seconds or final iteration
        lastPrintTime = tic;
        elapsedTime = toc(startTime);
        
        % Calculate processing speeds
        samplesPerSec = totalCollected / elapsedTime;
        permTestsPerSec = totalPermTests / elapsedTime;
        
        % Calculate progress percentage
        progressPct = 100 * iterations / maxIterations;
        
        % Print detailed progress report
        fprintf('\n--- Progress at %.1f seconds ---\n', elapsedTime);
        fprintf('Iteration %d/%d (%.1f%%): Collected %d p-values\n', ...
            iterations, maxIterations, progressPct, totalCollected);
        fprintf('Processing speed: %.1f samples/sec, %.1f permutation tests/sec\n', ...
            samplesPerSec, permTestsPerSec);
        
        % Print bin statistics
        fprintf('Bin counts: ');
        for b = 1:numBins
            fprintf('[%.1f-%.1f]: %d | ', ...
                skewnessBins(b), skewnessBins(b+1), currentCountPerBin(b));
            
            % Line break every 3 bins for readability
            if mod(b, 3) == 0
                fprintf('\n           ');
            end
        end
        fprintf('\n');
    end
end

% Trim arrays to actual size
biasedPValues = biasedPValues(1:totalCollected);
unbiasedPValues = unbiasedPValues(1:totalCollected);
measuredMeans = measuredMeans(1:totalCollected);
measuredVariances = measuredVariances(1:totalCollected);
measuredSkewness = measuredSkewness(1:totalCollected);
measuredKurtosis = measuredKurtosis(1:totalCollected);
measuredBowleySkewness = measuredBowleySkewness(1:totalCollected);
shapeParams = shapeParams(1:totalCollected);
distributionTypes = distributionTypes(1:totalCollected);

% Sort both sets of p-values independently
[biasedPValues, biasedIndices] = sort(biasedPValues);
[unbiasedPValues, unbiasedIndices] = sort(unbiasedPValues);

% At this point, you must decide which sorting to base other parameters on
% Let's use the biased p-values' sorting as the reference
measuredMeans = measuredMeans(biasedIndices);
measuredVariances = measuredVariances(biasedIndices);
measuredSkewness = measuredSkewness(biasedIndices);
measuredKurtosis = measuredKurtosis(biasedIndices);
measuredBowleySkewness = measuredBowleySkewness(biasedIndices);
shapeParams = shapeParams(biasedIndices);
distributionTypes = distributionTypes(biasedIndices);

% Create the final table with the sorted p-values and corresponding parameters
trainingData = table(biasedPValues, unbiasedPValues, ...
                     measuredMeans, measuredVariances, ...
                     measuredSkewness, measuredKurtosis, ...
                     measuredBowleySkewness, ...
                     shapeParams, distributionTypes, ...
                     'VariableNames', {'BiasedPValue', 'UnbiasedPValue', ...
                     'MeasuredMean', 'MeasuredVariance', ...
                     'MeasuredSkewness', 'MeasuredKurtosis', ...
                     'MeasuredBowleySkewness', ...
                     'ShapeParam', 'DistributionType'});

% Print final statistics
totalTime = toc(startTime);
fprintf('\n=============================================================\n');
fprintf('GENERATION COMPLETE\n');
fprintf('=============================================================\n');
fprintf('Total iterations:        %d\n', iterations);
fprintf('Total time:              %.2f seconds\n', totalTime);
fprintf('Total p-values collected: %d\n', totalCollected);
fprintf('Average p-values per iteration: %.1f\n', totalCollected/iterations);
fprintf('Average time per iteration: %.4f seconds\n', totalTime/iterations);
fprintf('Total permutation tests:  %d\n', totalPermTests);
fprintf('=============================================================\n');

% Display bin statistics
fprintf('\nBin Statistics:\n');
for i = 1:numBins
    fprintf('Bin [%.2f - %.2f]: %d samples (%.1f%%)\n', ...
        skewnessBins(i), skewnessBins(i+1), currentCountPerBin(i), ...
        100*currentCountPerBin(i)/totalCollected);
end

% Display distribution type statistics
distTypes = unique(distributionTypes);
fprintf('\nDistribution Type Statistics:\n');
for i = 1:length(distTypes)
    count = sum(strcmp(distributionTypes, distTypes{i}));
    fprintf('%s: %d samples (%.1f%%)\n', distTypes{i}, count, 100*count/totalCollected);
end

% Display summary statistics
fprintf('\nP-Value Statistics:\n');
fprintf('Mean biased p-value:    %.4f\n', mean(biasedPValues));
fprintf('Mean unbiased p-value:  %.4f\n', mean(unbiasedPValues));
fprintf('Mean measured mean:     %.4f\n', mean(measuredMeans));
fprintf('Mean measured variance: %.4f\n', mean(measuredVariances));
fprintf('Mean measured skewness: %.4f\n', mean(measuredSkewness));
fprintf('Mean measured kurtosis: %.4f\n', mean(measuredKurtosis));
fprintf('Mean measured Bowley skewness: %.4f\n', mean(measuredBowleySkewness));
fprintf('=============================================================\n');

% Continue with original visualization
%visualize_data_generation(progressData, trainingData, skewnessBins);

end