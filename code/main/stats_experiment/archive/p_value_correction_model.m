% Optimized script for p-value correction model
% Simplified to focus on Leave-One-Parameter-Out Cross-Validation
clear; close all;
rng(42); % Set random seed for reproducibility

%% Setup parameters
nPValues = 500;           % Number of p-values per parameter value
sampleSize = 50;         % Sample size per group
nPermutations = 1000;      % Number of permutations for permutation test
numSkewnessValues = 24;   % Number of skewness values to test
skewnessValues = linspace(0.1, 3.0, numSkewnessValues);
fixedScale = 3.0;         % Fixed scale parameter for all distributions
maxIterations = 5000;
generateData = 0;

% Start parallel pool if not already started
if isempty(gcp('nocreate'))
    c = parcluster('local');
    c.NumWorkers = min(24, feature('numcores'));
    saveProfile(c);
    parpool('local', min(23, feature('numcores')-1));
end


function p = PermutationTest(group1, group2, nPermutations)
% Performs a permutation test between two groups and returns the p-value
% Highly optimized version for maximum speed
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

function trainingData = generateTrainingData(maxIterations, sampleSize, nPermutations)
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
measuredSkewnessValues = zeros(initialCapacity, 1);
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
    batchSkewness = zeros(batchSize, 1);
    batchShapeParams = zeros(batchSize, 1);
    batchDistTypes = cell(batchSize, 1);
    
    % PERFORMANCE: Pre-allocate uniform data for all samples at once
    uniformData = rand(sampleSize, batchSize) - 0.5; % Range [-0.5, 0.5]
    
    % PERFORMANCE: Generate distribution parameters for the whole batch first
    distChoices = randi(4, batchSize, 1); % 1=gamma, 2=beta, 3=normal mixture, 4=weibull
    
    % PERFORMANCE: Prepare parameter arrays for each distribution
    gammaShapeParams = exp(unifrnd(log(0.5), log(10), batchSize, 1));
    
    betaParamsA = zeros(batchSize, 1);
    betaParamsB = zeros(batchSize, 1);
    for i = 1:batchSize
        if rand < 0.5
            % Negative skewness (a > b)
            betaParamsA(i) = unifrnd(2, 10);
            betaParamsB(i) = unifrnd(0.5, 2);
        else
            % Positive skewness (a < b)
            betaParamsA(i) = unifrnd(0.5, 2);
            betaParamsB(i) = unifrnd(2, 10);
        end
    end
    
    mixtureMu1 = zeros(batchSize, 1);
    mixtureMu2 = unifrnd(-3, 3, batchSize, 1);
    mixtureSigma1 = ones(batchSize, 1);
    mixtureSigma2 = unifrnd(0.5, 2, batchSize, 1);
    mixtureP = unifrnd(0.1, 0.9, batchSize, 1);
    
    weibullShapes = unifrnd(0.5, 5, batchSize, 1);
    
    % PERFORMANCE: Process the batch in parallel with optimized workload
    parfor i = 1:batchSize
        % PERFORMANCE: Create group1 as zeros array
        group1 = zeros(sampleSize, 1);
        
        % Use pre-generated parameters based on distribution choice
        distChoice = distChoices(i);
        
        % Generate data based on distribution choice
        switch distChoice
            case 1 % Gamma - good for positive skewness
                shapeParam = gammaShapeParams(i);
                scaleParam = 1.0;
                
                % Generate gamma data centered at mean
                meanGamma = shapeParam * scaleParam;
                group2 = gamrnd(shapeParam, scaleParam, sampleSize, 1) - meanGamma;
                
                % Store parameter and distribution type
                batchShapeParams(i) = shapeParam;
                batchDistTypes{i} = 'gamma';
                
            case 2 % Beta - can produce both positive and negative skewness
                % Use pre-generated beta parameters
                a = betaParamsA(i);
                b = betaParamsB(i);
                
                % Generate beta data centered at mean
                meanBeta = a / (a + b);
                group2 = betarnd(a, b, sampleSize, 1) - meanBeta;
                
                % Store parameter and distribution type
                batchShapeParams(i) = a / b; % Store a/b ratio as the shape parameter
                batchDistTypes{i} = 'beta';
                
            case 3 % Mixture of two normals - can create various skewness levels
                % Use pre-generated mixture parameters
                mu1 = mixtureMu1(i);
                mu2 = mixtureMu2(i);
                sigma1 = mixtureSigma1(i);
                sigma2 = mixtureSigma2(i);
                p = mixtureP(i);
                
                % PERFORMANCE: Vectorized mixture generation
                isGroup1 = rand(sampleSize, 1) < p;
                normData1 = normrnd(mu1, sigma1, sampleSize, 1);
                normData2 = normrnd(mu2, sigma2, sampleSize, 1);
                data = isGroup1 .* normData1 + (~isGroup1) .* normData2;
                
                % Center the data
                group2 = data - mean(data);
                
                % Store parameter and distribution type
                batchShapeParams(i) = p; % Store mix proportion as the shape parameter
                batchDistTypes{i} = 'normix';
                
            case 4 % Weibull - good for various skewness levels
                % Use pre-generated Weibull shape parameter
                shape = weibullShapes(i);
                
                % Generate Weibull data centered at mean
                scale = 1;
                meanWeibull = scale * gamma(1 + 1/shape);
                group2 = wblrnd(scale, shape, sampleSize, 1) - meanWeibull;
                
                % Store parameter and distribution type
                batchShapeParams(i) = shape;
                batchDistTypes{i} = 'weibull';
        end
        
        % PERFORMANCE: Optimized Bowley's skewness calculation
        sortedData = sort(group2);
        n = length(sortedData);
        q1Idx = floor(n * 0.25) + 1;
        q2Idx = floor(n * 0.50) + 1;
        q3Idx = floor(n * 0.75) + 1;
        
        q1 = sortedData(q1Idx);
        q2 = sortedData(q2Idx);
        q3 = sortedData(q3Idx);
        
        % Calculate Bowley's Quartile Skewness
        if (q3 - q1) == 0
            batchSkewness(i) = 0;
        else
            batchSkewness(i) = (q3 - 2*q2 + q1) / (q3 - q1);
        end
        
        % Ensure skewness is in valid range
        batchSkewness(i) = min(max(batchSkewness(i), -1), 1);
        
        % PERFORMANCE: Optimized permutation test
        batchBiased(i) = PermutationTest(group1, group2, nPermutations);
        batchUnbiased(i) = PermutationTest(group1, uniformData(:, i), nPermutations);
    end
    
    % Update total permutation test count
    totalPermTests = totalPermTests + 2 * batchSize;
    
    % Assign each sample to appropriate skewness bin
    samplesAddedThisIteration = 0;
    
    for i = 1:batchSize
        % Find which bin this sample belongs to
        binIdx = find(batchSkewness(i) >= skewnessBins(1:end-1) & ...
                     batchSkewness(i) < skewnessBins(2:end), 1);
        
        % Add the sample if it falls in a valid bin
        if ~isempty(binIdx)
            % Check if we need to resize arrays (safety check)
            if totalCollected + 1 > length(biasedPValues)
                % Increase capacity by 50%
                newCapacity = round(length(biasedPValues) * 1.5);
                biasedPValues(end+1:newCapacity) = 0;
                unbiasedPValues(end+1:newCapacity) = 0;
                measuredSkewnessValues(end+1:newCapacity) = 0;
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
            measuredSkewnessValues(totalCollected) = batchSkewness(i);
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
measuredSkewnessValues = measuredSkewnessValues(1:totalCollected);
shapeParams = shapeParams(1:totalCollected);
distributionTypes = distributionTypes(1:totalCollected);

% Sort both sets of p-values independently
[biasedPValues, biasedIndices] = sort(biasedPValues);
[unbiasedPValues, unbiasedIndices] = sort(unbiasedPValues);

% At this point, you must decide which sorting to base other parameters on
% Let's use the biased p-values' sorting as the reference
measuredSkewnessValues = measuredSkewnessValues(biasedIndices);
shapeParams = shapeParams(biasedIndices);
distributionTypes = distributionTypes(biasedIndices);

% Create the final table with the sorted p-values and corresponding parameters
trainingData = table(biasedPValues, unbiasedPValues, measuredSkewnessValues, ...
                     shapeParams, distributionTypes, ...
                     'VariableNames', {'BiasedPValue', 'UnbiasedPValue', ...
                     'MeasuredSkewness', 'ShapeParam', 'DistributionType'});

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
fprintf('Mean measured skewness: %.4f\n', mean(measuredSkewnessValues));
fprintf('=============================================================\n');

% Note: The sorting of p-values is now done before creating the trainingData table
% No additional analysis is needed here, as requested

% Continue with original visualization
visualizeDataGeneration(progressData, trainingData, skewnessBins);

end

function visualizeDataGeneration(progressData, trainingData, skewnessBins)
% Creates a comprehensive visualization of the data generation process
% and final dataset characteristics

% Create a figure with multiple subplots
figure('Position', [50, 50, 1400, 900], 'Color', 'white');

% 1. Bin filling progress over iterations
subplot(2, 3, 1);
plot(progressData.iterations, progressData.samplesCollected, 'b-', 'LineWidth', 2);
hold on;
xlabel('Iterations');
ylabel('Total Samples Collected');
title('Data Collection Progress');
grid on;

% 2. Time efficiency
subplot(2, 3, 2);
plot(progressData.iterations, progressData.samplesCollected ./ progressData.elapsedTime, 'r-', 'LineWidth', 2);
xlabel('Iterations');
ylabel('Samples per Second');
title('Collection Efficiency');
grid on;

% 3. Bin filling progress
subplot(2, 3, 3);
binCenters = (skewnessBins(1:end-1) + skewnessBins(2:end)) / 2;
plot(progressData.iterations, progressData.binCounts, 'LineWidth', 1.5);
xlabel('Iterations');
ylabel('Samples per Bin');
title('Bin Filling Progress');
grid on;

% Create a custom legend for the bin plot
legend(arrayfun(@(x,y) sprintf('%.2f-%.2f', x, y), skewnessBins(1:end-1), skewnessBins(2:end), 'UniformOutput', false), ...
    'Location', 'eastoutside', 'FontSize', 8);

% 4. Final distribution of skewness values
subplot(2, 3, 4);
histogram(trainingData.MeasuredSkewness, skewnessBins, 'FaceColor', [0.4, 0.6, 0.8], 'FaceAlpha', 0.7);
xlabel('Bowley''s Skewness');
ylabel('Frequency');
title('Final Distribution of Skewness Values');
grid on;

% 5. Relationship between biased and unbiased p-values, colored by skewness
subplot(2, 3, 5);
scatter(trainingData.BiasedPValue, trainingData.UnbiasedPValue, 10, trainingData.MeasuredSkewness, 'filled', 'MarkerFaceAlpha', 0.5);
hold on;
plot([0, 1], [0, 1], 'r--', 'LineWidth', 1.5);
colormap(jet);
c = colorbar;
c.Label.String = 'Skewness';
xlabel('Biased P-value');
ylabel('Unbiased P-value');
title('P-value Relationship by Skewness');
grid on;
axis([0, 1, 0, 1]);

% 6. Distribution of points by distribution type
subplot(2, 3, 6);
distTypes = unique(trainingData.DistributionType);
numDistTypes = length(distTypes);
colors = jet(numDistTypes);
hold on;

for i = 1:numDistTypes
    idx = strcmp(trainingData.DistributionType, distTypes{i});
    scatter(trainingData.MeasuredSkewness(idx), trainingData.BiasedPValue(idx), 20, colors(i,:), 'filled', 'MarkerFaceAlpha', 0.5);
end

xlabel('Measured Skewness');
ylabel('Biased P-value');
title('Distribution of P-values by Distribution Type');
legend(distTypes, 'Location', 'best');
grid on;

% Add overall title
sgtitle('Data Generation Characteristics', 'FontSize', 16, 'FontWeight', 'bold');

% Save the figure
saveas(gcf, 'data_generation_characteristics.png');
end            

%% Generate all data
fprintf('Generating data for all skewness values...\n');
tic;
if generateData == 0 % Changed to use equality comparison operator
    fprintf('Loading previously generated data...\n');
    load('pvalue_correction_full_data.mat');
else
    allData = generateTrainingData(maxIterations, sampleSize, nPermutations);
    elapsedTime = toc;
    fprintf('Data generation complete in %.2f seconds.\n', elapsedTime);
    % Save the full dataset
    save('pvalue_correction_full_data.mat', 'allData', 'skewnessValues');
end
%% Display summary statistics
fprintf('\nTotal number of paired p-values: %d\n', height(allData));
fprintf('Number of significant biased p-values (p < 0.05): %d (%.2f%%)\n', ...
    sum(allData.BiasedPValue < 0.05), 100*sum(allData.BiasedPValue < 0.05)/height(allData));
fprintf('Number of significant unbiased p-values (p < 0.05): %d (%.2f%%)\n', ...
    sum(allData.UnbiasedPValue < 0.05), 100*sum(allData.UnbiasedPValue < 0.05)/height(allData));

%% Perform Leave-One-Parameter-Out Cross-Validation with new generateTrainingData
fprintf('\nPerforming Leave-One-Parameter-Out Cross-Validation with new data generation\n');

% Initialize variables for cross-validation
numFolds = 20;  % Number of folds to use
fold_data = cell(numFolds, 1);  % Cell array to store data for each fold
rmse_poly_lopo = zeros(numFolds, 1);  % Store RMSE for each fold

% Generate stratified folds based on skewness values
% Instead of holding out specific skewness values, we'll stratify the data by skewness
% and create folds that maintain the distribution of skewness values

% First, sort the data by measured skewness
[sortedSkewness, sortIdx] = sort(allData.MeasuredSkewness);
sortedData = allData(sortIdx, :);

% Create indices for each fold using stratified sampling
foldIndices = cell(numFolds, 1);
numSamples = height(sortedData);
samplesPerFold = floor(numSamples / numFolds);

% Assign indices to folds in a stratified manner
for i = 1:numFolds
    if i < numFolds
        startIdx = (i-1) * samplesPerFold + 1;
        endIdx = i * samplesPerFold;
        foldIndices{i} = startIdx:endIdx;
    else
        % Last fold gets remaining samples
        startIdx = (i-1) * samplesPerFold + 1;
        foldIndices{i} = startIdx:numSamples;
    end
end

% Helper function to get predictions for a specific fold
function [X_test, y_test, y_pred] = getPredictionsForFold(allData, testIndices, final_mdl_poly)
    % Create boolean mask for test data
    testMask = false(height(allData), 1);
    testMask(testIndices) = true;
    
    % Get test and training data
    testData = allData(testMask, :);
    trainData = allData(~testMask, :);
    
    % Prepare input features
    X_train = [trainData.BiasedPValue, trainData.MeasuredSkewness];
    y_train = trainData.UnbiasedPValue;
    X_test = [testData.BiasedPValue, testData.MeasuredSkewness];
    y_test = testData.UnbiasedPValue;
    
    % Create polynomial features
    X_train_poly = [X_train, X_train(:,1).^2, X_train(:,1).^3, ...
                  X_train(:,2).^2, X_train(:,1).*X_train(:,2)];
    X_test_poly = [X_test, X_test(:,1).^2, X_test(:,1).^3, ...
                 X_test(:,2).^2, X_test(:,1).*X_test(:,2)];
    
    % Train and predict
    if nargin < 3
        mdl = fitlm(X_train_poly, y_train);
        y_pred = predict(mdl, X_test_poly);
    else
        y_pred = predict(final_mdl_poly, X_test_poly);
    end
end

% Perform cross-validation
run_cv = 1;
if run_cv == 1
    for fold = 1:numFolds
        fprintf('Processing fold %d/%d...\n', fold, numFolds);
        
        % Get test indices for this fold
        testIndices = foldIndices{fold};
        testIdx = sortIdx(testIndices);  % Map back to original indices
        
        % Use the getPredictionsForFold helper function
        [X_test, y_test, y_pred_poly] = getPredictionsForFold(allData, testIdx);
        
        % Calculate skewness range in test set
        testData = allData(testIdx, :);
        minSkew = min(testData.MeasuredSkewness);
        maxSkew = max(testData.MeasuredSkewness);
        meanSkew = mean(testData.MeasuredSkewness);
        fprintf('Test set skewness range: %.4f to %.4f (mean: %.4f)\n', ...
            minSkew, maxSkew, meanSkew);
        
        fprintf('Data split: %d samples for training, %d samples for testing\n', ...
            height(allData) - length(testIdx), length(testIdx));
        
        rmse_poly_lopo(fold) = sqrt(mean((y_pred_poly - y_test).^2));
        
        fprintf('RMSE for fold %d: %.6f\n', fold, rmse_poly_lopo(fold));
        
        % Store fold data for later analysis
        fold_data{fold} = struct();
        fold_data{fold}.TestIndices = testIdx;
        fold_data{fold}.TestSkewness = testData.MeasuredSkewness;
        fold_data{fold}.RMSE = rmse_poly_lopo(fold);
        fold_data{fold}.Predictions = y_pred_poly;
        fold_data{fold}.TrueValues = y_test;
    end
    
    %% Display cross-validation results
    fprintf('\nCross-Validation results (RMSE):\n');
    fprintf('Polynomial model: %.6f ± %.6f\n', mean(rmse_poly_lopo), std(rmse_poly_lopo));

    %% Train final model using all data
    fprintf('Training final model with all data...\n');
    
    % Prepare all data
    X_all = [allData.BiasedPValue, allData.MeasuredSkewness];
    y_all = allData.UnbiasedPValue;
    
    % Prepare polynomial features
    X_all_poly = [X_all, X_all(:,1).^2, X_all(:,1).^3, ...
                X_all(:,2).^2, X_all(:,1).*X_all(:,2)];
    
    % Train final model
    final_mdl_poly = fitlm(X_all_poly, y_all);
    
    % Create figure to visualize CV results
    figure('Position', [100, 100, 1200, 600]);
    
    % Plot 1: RMSE by fold
    subplot(1, 2, 1);
    bar(1:numFolds, rmse_poly_lopo, 'FaceColor', [0.3, 0.6, 0.8]);
    hold on;
    plot([0, numFolds+1], [mean(rmse_poly_lopo), mean(rmse_poly_lopo)], 'r-', 'LineWidth', 2);
    xlabel('Fold');
    ylabel('RMSE');
    title('RMSE by Cross-Validation Fold');
    grid on;
    
    % Plot 2: Mean skewness vs RMSE for each fold
    subplot(1, 2, 2);
    foldMeanSkewness = zeros(numFolds, 1);
    for i = 1:numFolds
        foldMeanSkewness(i) = mean(fold_data{i}.TestSkewness);
    end
    
    scatter(foldMeanSkewness, rmse_poly_lopo, 100, 'filled', 'MarkerFaceColor', [0.3, 0.6, 0.8]);
    xlabel('Mean Skewness in Test Fold');
    ylabel('RMSE');
    title('RMSE vs. Mean Skewness in Folds');
    grid on;
    
    % Add correlation coefficient to the plot
    [rho, pval] = corr(foldMeanSkewness, rmse_poly_lopo);
    text(0.05, 0.95, sprintf('r = %.4f (p = %.4f)', rho, pval), ...
        'Units', 'normalized', 'FontSize', 12);
    
    saveas(gcf, 'cross_validation_results.png');
    
    %% Analyze best and worst performing folds
    [~, best_fold_idx] = min(rmse_poly_lopo);
    [~, worst_fold_idx] = max(rmse_poly_lopo);
    
    fprintf('\nBest performing fold: Fold %d (RMSE = %.6f, Mean Skewness = %.4f)\n', ...
            best_fold_idx, rmse_poly_lopo(best_fold_idx), foldMeanSkewness(best_fold_idx));
    fprintf('Worst performing fold: Fold %d (RMSE = %.6f, Mean Skewness = %.4f)\n', ...
            worst_fold_idx, rmse_poly_lopo(worst_fold_idx), foldMeanSkewness(worst_fold_idx));
    
    % Create detailed plots for best and worst folds
    figure('Position', [100, 100, 1200, 800]);
    
    % Get predictions for best fold using the helper function
    [~, best_true, best_pred] = getPredictionsForFold(allData, fold_data{best_fold_idx}.TestIndices, final_mdl_poly);
    
    % Get predictions for worst fold using the helper function
    [~, worst_true, worst_pred] = getPredictionsForFold(allData, fold_data{worst_fold_idx}.TestIndices, final_mdl_poly);
    
    % Plot best fold results
    subplot(2, 2, 1);
    scatter(best_true, best_pred, 20, 'filled', 'MarkerFaceAlpha', 0.5);
    hold on;
    plot([0, 1], [0, 1], 'r--', 'LineWidth', 1.5);
    title(sprintf('Best Fold (Fold %d): Predicted vs. True', best_fold_idx));
    xlabel('True P-Value');
    ylabel('Predicted P-Value');
    grid on;
    axis square;
    axis([0 1 0 1]);
    
    % Plot best fold error histogram
    subplot(2, 2, 2);
    best_errors = best_pred - best_true;
    histogram(best_errors, 50);
    title(sprintf('Best Fold Error Distribution (RMSE = %.6f)', rmse_poly_lopo(best_fold_idx)));
    xlabel('Error (Predicted - True)');
    ylabel('Frequency');
    grid on;
    
    % Plot worst fold results
    subplot(2, 2, 3);
    scatter(worst_true, worst_pred, 20, 'filled', 'MarkerFaceAlpha', 0.5);
    hold on;
    plot([0, 1], [0, 1], 'r--', 'LineWidth', 1.5);
    title(sprintf('Worst Fold (Fold %d): Predicted vs. True', worst_fold_idx));
    xlabel('True P-Value');
    ylabel('Predicted P-Value');
    grid on;
    axis square;
    axis([0 1 0 1]);
    
    % Plot worst fold error histogram
    subplot(2, 2, 4);
    worst_errors = worst_pred - worst_true;
    histogram(worst_errors, 50);
    title(sprintf('Worst Fold Error Distribution (RMSE = %.6f)', rmse_poly_lopo(worst_fold_idx)));
    xlabel('Error (Predicted - True)');
    ylabel('Frequency');
    grid on;
    
    sgtitle('Best vs. Worst Fold Comparison: Polynomial Model');
    saveas(gcf, 'best_worst_fold_comparison.png');
    
    %% Print the model coefficients
    fprintf('\n============= MODEL COEFFICIENTS =============\n');
    coeffs = final_mdl_poly.Coefficients.Estimate;
    fprintf('Polynomial: UnbiasedPValue = %.4f + (%.4f * BiasedPValue) + (%.4f * MeasuredSkewness) + ...\n', ...
        coeffs(1), coeffs(2), coeffs(3));
    
    % Display full polynomial equation
    featureNames = {'Intercept', 'BiasedPValue', 'MeasuredSkewness', 'BiasedPValue^2', 'BiasedPValue^3', 'MeasuredSkewness^2', 'BiasedPValue*MeasuredSkewness'};
    fprintf('\nFull polynomial equation:\n');
    fprintf('UnbiasedPValue = %.6f', coeffs(1));
    for i = 2:length(coeffs)
        if coeffs(i) >= 0
            fprintf(' + %.6f * %s', coeffs(i), featureNames{i});
        else
            fprintf(' - %.6f * %s', abs(coeffs(i)), featureNames{i});
        end
    end
    fprintf('\n');
    
    % Save the final model
    save('pvalue_correction_poly_model.mat', 'final_mdl_poly');
    fprintf('Final polynomial model saved to pvalue_correction_poly_model.mat\n');
else
    % Prepare all data
    X_all = [allData.BiasedPValue, allData.MeasuredSkewness];
    y_all = allData.UnbiasedPValue;
    
    % Prepare polynomial features
    X_all_poly = [X_all, X_all(:,1).^2, X_all(:,1).^3, ...
                X_all(:,2).^2, X_all(:,1).*X_all(:,2)];
    
    % Train final model
    final_mdl_poly = fitlm(X_all_poly, y_all);
end
%% Function to calculate Bowley's Quartile Skewness
function sk = bowleysSkewness(data)
    % Calculate quartiles
    q1 = quantile(data, 0.25);
    q2 = quantile(data, 0.50); % median
    q3 = quantile(data, 0.75);
    
    % Calculate Bowley's Quartile Skewness
    sk = (q3 - 2*q2 + q1) / (q3 - q1);
end


%% Testing code with multiple distributions
fprintf('\n============= TESTING WITH MULTIPLE DISTRIBUTIONS =============\n');

% Setup parameters for the test
nPValuesTest = 5000;        % Number of p-values per distribution parameter
sampleSizeTest = 100;       % Sample size per group
nPermutationsTest = 1000;   % Number of permutations

% Define the distribution parameters to test
chiSquaredDfValues = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64];
weibullShapeValues = [0.5, 1, 1.5, 2, 3, 5, 10];  % Shape parameter
lognormalSigmaValues = [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2];  % Sigma parameter

fprintf('Testing with %d Chi-Squared, %d Weibull, and %d Log-normal distribution parameters\n', ...
    length(chiSquaredDfValues), length(weibullShapeValues), length(lognormalSigmaValues));

% Create a structure to store results
testResults = struct();
testResults.ChiSquare = struct('BiasedPValues', [], 'UnbiasedPValues', [], ...
                            'PredictedPValues', [], 'RMSE', [], ...
                            'MeasuredSkewness', [], 'Parameter', []);
testResults.Weibull = struct('BiasedPValues', [], 'UnbiasedPValues', [], ...
                            'PredictedPValues', [], 'RMSE', [], ...
                            'MeasuredSkewness', [], 'Parameter', []);
testResults.LogNormal = struct('BiasedPValues', [], 'UnbiasedPValues', [], ...
                            'PredictedPValues', [], 'RMSE', [], ...
                            'MeasuredSkewness', [], 'Parameter', []);

% Create a parallel pool if not already started
if isempty(gcp('nocreate'))
    c = parcluster('local');
    c.NumWorkers = min(24, feature('numcores'));
    saveProfile(c);
    parpool('local', min(23, feature('numcores')-1));
end

%% Process Chi-Squared distributions
fprintf('\nProcessing Chi-Squared distributions...\n');

% Initialize cell array for parallel results
chi_results = cell(length(chiSquaredDfValues), 1);

% Create a waitbar for overall progress
h = waitbar(0, 'Processing Chi-Squared distributions...', 'Name', 'Progress');

% Create a parallel.pool.DataQueue to track progress
dataQueue = parallel.pool.DataQueue;

% Set up a listener to update the waitbar
numParams = length(chiSquaredDfValues);
afterEach(dataQueue, @(~) updateChiWaitbar(h, numParams));

% Process each degree of freedom in parallel
parfor d = 1:length(chiSquaredDfValues)
    df = chiSquaredDfValues(d);
    fprintf('Processing Chi-Squared df=%d...\n', df);
    
    % Initialize arrays to collect data for this df
    biasedPVals = zeros(nPValuesTest, 1);
    unbiasedPVals = zeros(nPValuesTest, 1);
    measuredSkewness = zeros(nPValuesTest, 1);
    
    % Pre-generate all uniform data for unbiased p-values
    uniformData = zeros(sampleSizeTest, nPValuesTest);
    for i = 1:nPValuesTest
        uniformData(:, i) = unifrnd(-0.5, 0.5, sampleSizeTest, 1);
    end
    
    % Group 1 is all zeros
    group1 = zeros(sampleSizeTest, 1);
    
    % Use fixed df for all samples
    random_df = df;
    
    % Generate data and process all p-values
    for i = 1:nPValuesTest
        % Generate chi-squared data with the random_df
        group2 = chi2rnd(random_df, sampleSizeTest, 1) - random_df;
        
        % Calculate Bowley's Quartile Skewness for this sample
        measuredSkewness(i) = bowleysSkewness(group2);
        
        % Perform permutation test for biased p-value
        biasedPVals(i) = PermutationTest(group1, group2, nPermutationsTest);
        
        % Permutation test for unbiased p-value (uniform distribution)
        unbiasedPVals(i) = PermutationTest(group1, uniformData(:, i), nPermutationsTest);
    end
    
    % Sort the p-values independently
    biasedPVals = sort(biasedPVals);
    unbiasedPVals = sort(unbiasedPVals);
    measuredSkewness = sort(measuredSkewness);
    
    % Make a local copy of the model for parallel use
    local_model = final_mdl_poly;  % Assuming this model exists in the workspace
    
    % Prepare feature vector for prediction using MEASURED skewness
    X_test = [biasedPVals, measuredSkewness];
    X_test_poly = [X_test, X_test(:,1).^2, X_test(:,1).^3, ...
                X_test(:,2).^2, X_test(:,1).*X_test(:,2)];
    
    % Use the local model copy
    predictedPVals = predict(local_model, X_test_poly);
    
    % Calculate local RMSE
    local_errors = predictedPVals - unbiasedPVals;
    local_RMSE = sqrt(mean(local_errors.^2));
    
    % Create a structure to hold results for this df
    local_results = struct();
    local_results.BiasedPValues = biasedPVals;
    local_results.UnbiasedPValues = unbiasedPVals;
    local_results.PredictedPValues = predictedPVals;
    local_results.MeasuredSkewness = measuredSkewness;
    local_results.Parameter = repmat(df, length(biasedPVals), 1);
    local_results.RMSE = local_RMSE;
    
    % Store data in a cell array to be combined after parallel execution
    chi_results{d} = local_results;
    
    fprintf('RMSE for Chi-Squared df=%d: %.6f\n', df, local_RMSE);
    
    % Send update to the dataQueue to update progress bar
    send(dataQueue, 1);
end

% Close the waitbar after all iterations are complete
close(h);

% Helper function to update the waitbar
function updateChiWaitbar(h, numParams)
    persistent count
    if isempty(count)
        count = 0;
    end
    count = count + 1;
    waitbar(count/numParams, h, sprintf('Processing Chi-Squared: %d/%d complete (%.1f%%)', count, numParams, (count/numParams)*100));
end

% Combine all Chi-Squared results after parallel execution
chiSquaredResults = struct();
chiSquaredResults.BiasedPValues = [];
chiSquaredResults.UnbiasedPValues = [];
chiSquaredResults.PredictedPValues = [];
chiSquaredResults.MeasuredSkewness = [];
chiSquaredResults.Parameter = [];
chiSquaredResults.RMSE_ByParam = zeros(length(chiSquaredDfValues), 1);

for d = 1:length(chiSquaredDfValues)
    chiSquaredResults.BiasedPValues = [chiSquaredResults.BiasedPValues; chi_results{d}.BiasedPValues];
    chiSquaredResults.UnbiasedPValues = [chiSquaredResults.UnbiasedPValues; chi_results{d}.UnbiasedPValues];
    chiSquaredResults.PredictedPValues = [chiSquaredResults.PredictedPValues; chi_results{d}.PredictedPValues];
    chiSquaredResults.MeasuredSkewness = [chiSquaredResults.MeasuredSkewness; chi_results{d}.MeasuredSkewness];
    chiSquaredResults.Parameter = [chiSquaredResults.Parameter; chi_results{d}.Parameter];
    chiSquaredResults.RMSE_ByParam(d) = chi_results{d}.RMSE;
end

% Calculate overall RMSE
chi_errors = chiSquaredResults.PredictedPValues - chiSquaredResults.UnbiasedPValues;
chiSquaredResults.RMSE_Overall = sqrt(mean(chi_errors.^2));

% Store in the main results structure
testResults.ChiSquare = chiSquaredResults;

%% Process Weibull distributions
fprintf('\nProcessing Weibull distributions...\n');

% Initialize cell array for parallel results
weibull_results = cell(length(weibullShapeValues), 1);

% Create a waitbar for overall progress
h = waitbar(0, 'Processing Weibull distributions...', 'Name', 'Progress');

% Create a parallel.pool.DataQueue to track progress
dataQueue = parallel.pool.DataQueue;

% Set up a listener to update the waitbar
numParams = length(weibullShapeValues);
afterEach(dataQueue, @(~) updateWeibullWaitbar(h, numParams));

% Process each shape parameter in parallel
parfor p = 1:length(weibullShapeValues)
    shape = weibullShapeValues(p);
    fprintf('Processing Weibull shape=%.2f...\n', shape);
    
    % Initialize arrays to collect data for this shape parameter
    biasedPVals = zeros(nPValuesTest, 1);
    unbiasedPVals = zeros(nPValuesTest, 1);
    measuredSkewness = zeros(nPValuesTest, 1);
    
    % Pre-generate all uniform data for unbiased p-values
    uniformData = zeros(sampleSizeTest, nPValuesTest);
    for i = 1:nPValuesTest
        uniformData(:, i) = unifrnd(-0.5, 0.5, sampleSizeTest, 1);
    end
    
    % Group 1 is all zeros
    group1 = zeros(sampleSizeTest, 1);
    
    % Generate data and process all p-values
    for i = 1:nPValuesTest
        % Generate Weibull data with the specified shape parameter (scale=1)
        % Center the distribution by subtracting the mean
        weibullMean = gamma(1 + 1/shape);
        group2 = wblrnd(1, shape, sampleSizeTest, 1) - weibullMean;
        
        % Calculate Bowley's Quartile Skewness for this sample
        measuredSkewness(i) = bowleysSkewness(group2);
        
        % Perform permutation test for biased p-value
        biasedPVals(i) = PermutationTest(group1, group2, nPermutationsTest);
        
        % Permutation test for unbiased p-value (uniform distribution)
        unbiasedPVals(i) = PermutationTest(group1, uniformData(:, i), nPermutationsTest);
    end
    
    % Sort the p-values independently
    biasedPVals = sort(biasedPVals);
    unbiasedPVals = sort(unbiasedPVals);
    measuredSkewness = sort(measuredSkewness);
    
    % Make a local copy of the model for parallel use
    local_model = final_mdl_poly;  % Assuming this model exists in the workspace
    
    % Prepare feature vector for prediction using MEASURED skewness
    X_test = [biasedPVals, measuredSkewness];
    X_test_poly = [X_test, X_test(:,1).^2, X_test(:,1).^3, ...
                X_test(:,2).^2, X_test(:,1).*X_test(:,2)];
    
    % Use the local model copy
    predictedPVals = predict(local_model, X_test_poly);
    
    % Calculate local RMSE
    local_errors = predictedPVals - unbiasedPVals;
    local_RMSE = sqrt(mean(local_errors.^2));
    
    % Create a structure to hold results for this shape parameter
    local_results = struct();
    local_results.BiasedPValues = biasedPVals;
    local_results.UnbiasedPValues = unbiasedPVals;
    local_results.PredictedPValues = predictedPVals;
    local_results.MeasuredSkewness = measuredSkewness;
    local_results.Parameter = repmat(shape, length(biasedPVals), 1);
    local_results.RMSE = local_RMSE;
    
    % Store data in a cell array to be combined after parallel execution
    weibull_results{p} = local_results;
    
    fprintf('RMSE for Weibull shape=%.2f: %.6f\n', shape, local_RMSE);
    
    % Send update to the dataQueue to update progress bar
    send(dataQueue, 1);
end

% Close the waitbar after all iterations are complete
close(h);

% Helper function to update the waitbar
function updateWeibullWaitbar(h, numParams)
    persistent count
    if isempty(count)
        count = 0;
    end
    count = count + 1;
    waitbar(count/numParams, h, sprintf('Processing Weibull: %d/%d complete (%.1f%%)', count, numParams, (count/numParams)*100));
end

% Combine all Weibull results after parallel execution
weibullResults = struct();
weibullResults.BiasedPValues = [];
weibullResults.UnbiasedPValues = [];
weibullResults.PredictedPValues = [];
weibullResults.MeasuredSkewness = [];
weibullResults.Parameter = [];
weibullResults.RMSE_ByParam = zeros(length(weibullShapeValues), 1);

for p = 1:length(weibullShapeValues)
    weibullResults.BiasedPValues = [weibullResults.BiasedPValues; weibull_results{p}.BiasedPValues];
    weibullResults.UnbiasedPValues = [weibullResults.UnbiasedPValues; weibull_results{p}.UnbiasedPValues];
    weibullResults.PredictedPValues = [weibullResults.PredictedPValues; weibull_results{p}.PredictedPValues];
    weibullResults.MeasuredSkewness = [weibullResults.MeasuredSkewness; weibull_results{p}.MeasuredSkewness];
    weibullResults.Parameter = [weibullResults.Parameter; weibull_results{p}.Parameter];
    weibullResults.RMSE_ByParam(p) = weibull_results{p}.RMSE;
end

% Calculate overall RMSE
weibull_errors = weibullResults.PredictedPValues - weibullResults.UnbiasedPValues;
weibullResults.RMSE_Overall = sqrt(mean(weibull_errors.^2));

% Store in the main results structure
testResults.Weibull = weibullResults;

%% Process Log-normal distributions
fprintf('\nProcessing Log-normal distributions...\n');

% Initialize cell array for parallel results
lognormal_results = cell(length(lognormalSigmaValues), 1);

% Create a waitbar for overall progress
h = waitbar(0, 'Processing Log-normal distributions...', 'Name', 'Progress');

% Create a parallel.pool.DataQueue to track progress
dataQueue = parallel.pool.DataQueue;

% Set up a listener to update the waitbar
numParams = length(lognormalSigmaValues);
afterEach(dataQueue, @(~) updateLognormalWaitbar(h, numParams));

% Process each sigma parameter in parallel
parfor p = 1:length(lognormalSigmaValues)
    sigma = lognormalSigmaValues(p);
    fprintf('Processing Log-normal sigma=%.2f...\n', sigma);
    
    % Initialize arrays to collect data for this sigma parameter
    biasedPVals = zeros(nPValuesTest, 1);
    unbiasedPVals = zeros(nPValuesTest, 1);
    measuredSkewness = zeros(nPValuesTest, 1);
    
    % Pre-generate all uniform data for unbiased p-values
    uniformData = zeros(sampleSizeTest, nPValuesTest);
    for i = 1:nPValuesTest
        uniformData(:, i) = unifrnd(-0.5, 0.5, sampleSizeTest, 1);
    end
    
    % Group 1 is all zeros
    group1 = zeros(sampleSizeTest, 1);
    
    % Generate data and process all p-values
    for i = 1:nPValuesTest
        % Generate Log-normal data with the specified sigma parameter
        % Setting mu=-sigma^2/2 gives a distribution with mean=1
        mu = -sigma^2/2;
        group2 = lognrnd(mu, sigma, sampleSizeTest, 1) - 1; % Center around 0
        
        % Calculate Bowley's Quartile Skewness for this sample
        measuredSkewness(i) = bowleysSkewness(group2);
        
        % Perform permutation test for biased p-value
        biasedPVals(i) = PermutationTest(group1, group2, nPermutationsTest);
        
        % Permutation test for unbiased p-value (uniform distribution)
        unbiasedPVals(i) = PermutationTest(group1, uniformData(:, i), nPermutationsTest);
    end
    
    % Sort the p-values independently
    biasedPVals = sort(biasedPVals);
    unbiasedPVals = sort(unbiasedPVals);
    measuredSkewness = sort(measuredSkewness);
    
    % Make a local copy of the model for parallel use
    local_model = final_mdl_poly;  % Assuming this model exists in the workspace
    
    % Prepare feature vector for prediction using MEASURED skewness
    X_test = [biasedPVals, measuredSkewness];
    X_test_poly = [X_test, X_test(:,1).^2, X_test(:,1).^3, ...
                X_test(:,2).^2, X_test(:,1).*X_test(:,2)];
    
    % Use the local model copy
    predictedPVals = predict(local_model, X_test_poly);
    
    % Calculate local RMSE
    local_errors = predictedPVals - unbiasedPVals;
    local_RMSE = sqrt(mean(local_errors.^2));
    
    % Create a structure to hold results for this sigma parameter
    local_results = struct();
    local_results.BiasedPValues = biasedPVals;
    local_results.UnbiasedPValues = unbiasedPVals;
    local_results.PredictedPValues = predictedPVals;
    local_results.MeasuredSkewness = measuredSkewness;
    local_results.Parameter = repmat(sigma, length(biasedPVals), 1);
    local_results.RMSE = local_RMSE;
    
    % Store data in a cell array to be combined after parallel execution
    lognormal_results{p} = local_results;
    
    fprintf('RMSE for Log-normal sigma=%.2f: %.6f\n', sigma, local_RMSE);
    
    % Send update to the dataQueue to update progress bar
    send(dataQueue, 1);
end

% Close the waitbar after all iterations are complete
close(h);

% Helper function to update the waitbar
function updateLognormalWaitbar(h, numParams)
    persistent count
    if isempty(count)
        count = 0;
    end
    count = count + 1;
    waitbar(count/numParams, h, sprintf('Processing Log-normal: %d/%d complete (%.1f%%)', count, numParams, (count/numParams)*100));
end

% Combine all Log-normal results after parallel execution
lognormalResults = struct();
lognormalResults.BiasedPValues = [];
lognormalResults.UnbiasedPValues = [];
lognormalResults.PredictedPValues = [];
lognormalResults.MeasuredSkewness = [];
lognormalResults.Parameter = [];
lognormalResults.RMSE_ByParam = zeros(length(lognormalSigmaValues), 1);

for p = 1:length(lognormalSigmaValues)
    lognormalResults.BiasedPValues = [lognormalResults.BiasedPValues; lognormal_results{p}.BiasedPValues];
    lognormalResults.UnbiasedPValues = [lognormalResults.UnbiasedPValues; lognormal_results{p}.UnbiasedPValues];
    lognormalResults.PredictedPValues = [lognormalResults.PredictedPValues; lognormal_results{p}.PredictedPValues];
    lognormalResults.MeasuredSkewness = [lognormalResults.MeasuredSkewness; lognormal_results{p}.MeasuredSkewness];
    lognormalResults.Parameter = [lognormalResults.Parameter; lognormal_results{p}.Parameter];
    lognormalResults.RMSE_ByParam(p) = lognormal_results{p}.RMSE;
end

% Calculate overall RMSE
lognormal_errors = lognormalResults.PredictedPValues - lognormalResults.UnbiasedPValues;
lognormalResults.RMSE_Overall = sqrt(mean(lognormal_errors.^2));

% Store in the main results structure
testResults.LogNormal = lognormalResults;

%% Visualize combined results
% Create a multi-panel figure for comparing all distributions
figure('Position', [50, 50, 1200, 1400]);
set(gcf, 'Color', 'white'); % Set white background
fontName = 'Arial';
fontSize = 12;
titleSize = 14;
mainTitleSize = 16;

% Row 1: Boxplots of Skewness for all distributions
subplot(3, 1, 1);

% Create data for boxplot
allSkew = [chiSquaredResults.MeasuredSkewness; 
           weibullResults.MeasuredSkewness; 
           lognormalResults.MeasuredSkewness];

% Create grouping vector
groupVec = zeros(size(allSkew));
totalParams = length(chiSquaredDfValues) + length(weibullShapeValues) + length(lognormalSigmaValues);
labelCell = cell(1, totalParams);

% Assign group numbers for Chi-Squared
paramIdx = 1;
for d = 1:length(chiSquaredDfValues)
    idx = find(chiSquaredResults.Parameter == chiSquaredDfValues(d));
    groupVec(idx) = paramIdx;
    labelCell{paramIdx} = sprintf('χ²(df=%d)', chiSquaredDfValues(d));
    paramIdx = paramIdx + 1;
end

% Assign group numbers for Weibull
weibullStart = length(chiSquaredResults.MeasuredSkewness) + 1;
for s = 1:length(weibullShapeValues)
    idx = weibullStart - 1 + find(weibullResults.Parameter == weibullShapeValues(s));
    groupVec(idx) = paramIdx;
    labelCell{paramIdx} = sprintf('Weib(s=%.1f)', weibullShapeValues(s));
    paramIdx = paramIdx + 1;
end

% Assign group numbers for Log-normal
lognormalStart = weibullStart + length(weibullResults.MeasuredSkewness);
for s = 1:length(lognormalSigmaValues)
    idx = lognormalStart - 1 + find(lognormalResults.Parameter == lognormalSigmaValues(s));
    groupVec(idx) = paramIdx;
    labelCell{paramIdx} = sprintf('LogN(σ=%.1f)', lognormalSigmaValues(s));
    paramIdx = paramIdx + 1;
end

% Create boxplot with proper format
boxplotHandle = boxplot(allSkew, groupVec, 'Labels', labelCell);
set(boxplotHandle, 'LineWidth', 1.2);
set(gca, 'XTickLabelRotation', 45); % Rotate x-axis labels for better readability

% Improve text formatting
title('Measured Bowley''s Skewness Across Distributions', 'FontSize', titleSize, 'FontName', fontName, 'FontWeight', 'bold');
ylabel('Skewness', 'FontSize', fontSize, 'FontName', fontName, 'FontWeight', 'bold');
set(gca, 'FontSize', fontSize-2, 'FontName', fontName, 'XGrid', 'on', 'YGrid', 'on', 'Box', 'on');

% Add distribution type color coding
hold on;
numChiParams = length(chiSquaredDfValues);
numWeibParams = length(weibullShapeValues);
numLogNParams = length(lognormalSigmaValues);

% Set box colors by distribution type
h = findobj(gca, 'Tag', 'Box');
for j = 1:length(h)
    if j <= numLogNParams
        patch(get(h(j), 'XData'), get(h(j), 'YData'), [0.8, 0.4, 0.2], 'FaceAlpha', 0.5);
    elseif j <= numLogNParams + numWeibParams
        patch(get(h(j), 'XData'), get(h(j), 'YData'), [0.2, 0.6, 0.8], 'FaceAlpha', 0.5);
    else
        patch(get(h(j), 'XData'), get(h(j), 'YData'), [0.4, 0.8, 0.4], 'FaceAlpha', 0.5);
    end
end

% Add legend for distribution types
legend({'Log-normal', 'Weibull', 'Chi-Squared'}, 'Location', 'NorthEast');

% Row 2: Scatter plot of predicted vs. true p-values for all distributions
subplot(3, 1, 2);

% Create a combined scatter plot for all distributions
hold on;
scatter(chiSquaredResults.UnbiasedPValues, chiSquaredResults.PredictedPValues, 15, 'filled', 'MarkerFaceColor', [0.4, 0.8, 0.4], 'MarkerFaceAlpha', 0.3);
scatter(weibullResults.UnbiasedPValues, weibullResults.PredictedPValues, 15, 'filled', 'MarkerFaceColor', [0.2, 0.6, 0.8], 'MarkerFaceAlpha', 0.3);
scatter(lognormalResults.UnbiasedPValues, lognormalResults.PredictedPValues, 15, 'filled', 'MarkerFaceColor', [0.8, 0.4, 0.2], 'MarkerFaceAlpha', 0.3);
plot([0, 1], [0, 1], 'k--', 'LineWidth', 2);

% Calculate combined RMSE
chiCount = length(chiSquaredResults.UnbiasedPValues);
weibCount = length(weibullResults.UnbiasedPValues);
logNCount = length(lognormalResults.UnbiasedPValues);
totalCount = chiCount + weibCount + logNCount;

combinedRMSE = sqrt((chiSquaredResults.RMSE_Overall^2 * chiCount + ...
                    weibullResults.RMSE_Overall^2 * weibCount + ...
                    lognormalResults.RMSE_Overall^2 * logNCount) / totalCount);

% Improve text formatting
title(sprintf('Model Performance Across Distributions (Overall RMSE: %.4f)', combinedRMSE), 'FontSize', titleSize, 'FontName', fontName, 'FontWeight', 'bold');
xlabel('True P-Value', 'FontSize', fontSize, 'FontName', fontName, 'FontWeight', 'bold');
ylabel('Predicted P-Value', 'FontSize', fontSize, 'FontName', fontName, 'FontWeight', 'bold');
set(gca, 'FontSize', fontSize-1, 'FontName', fontName, 'XGrid', 'on', 'YGrid', 'on', 'Box', 'on');
axis square;
axis([0 1 0 1]);

% Add legend
legend({'Chi-Squared', 'Weibull', 'Log-normal', 'Ideal'}, 'Location', 'SouthEast');

% Row 3: Bar chart of RMSE by distribution and parameter
subplot(3, 1, 3);

% Create data for grouped bar chart
numGroups = 3; % Chi-squared, Weibull, Log-normal
groupData = {chiSquaredResults.RMSE_ByParam, weibullResults.RMSE_ByParam, lognormalResults.RMSE_ByParam};
maxBars = max([length(chiSquaredDfValues), length(weibullShapeValues), length(lognormalSigmaValues)]);
barData = nan(maxBars, numGroups);

% Fill bar data matrix
barData(1:length(chiSquaredDfValues), 1) = chiSquaredResults.RMSE_ByParam;
barData(1:length(weibullShapeValues), 2) = weibullResults.RMSE_ByParam;
barData(1:length(lognormalSigmaValues), 3) = lognormalResults.RMSE_ByParam;

% Create grouped bar chart
barHandles = bar(barData, 'grouped');
set(barHandles(1), 'FaceColor', [0.4, 0.8, 0.4]); % Chi-squared
set(barHandles(2), 'FaceColor', [0.2, 0.6, 0.8]); % Weibull
set(barHandles(3), 'FaceColor', [0.8, 0.4, 0.2]); % Log-normal

% Improve text formatting
title('RMSE by Distribution and Parameter', 'FontSize', titleSize, 'FontName', fontName, 'FontWeight', 'bold');
ylabel('RMSE', 'FontSize', fontSize, 'FontName', fontName, 'FontWeight', 'bold');
set(gca, 'FontSize', fontSize-1, 'FontName', fontName, 'XGrid', 'on', 'YGrid', 'on', 'Box', 'on');

% Create x-axis labels
xLabels = cell(1, maxBars);
for i = 1:length(chiSquaredDfValues)
    if i <= length(chiSquaredDfValues)
        xLabels{i} = sprintf('Set %d', i);
    end
end
set(gca, 'XTickLabel', xLabels);

% Add overall legend
legend({'Chi-Squared', 'Weibull', 'Log-normal'}, 'Location', 'NorthEast');

% Add annotations for the parameters below the plot
annotation('textbox', [0.15, 0.02, 0.7, 0.05], 'String', ...
    sprintf('Chi-Squared (df): %s', mat2str(chiSquaredDfValues)), ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontName', fontName);
annotation('textbox', [0.15, 0.06, 0.7, 0.05], 'String', ...
    sprintf('Weibull (shape): %s', mat2str(weibullShapeValues)), ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontName', fontName);
annotation('textbox', [0.15, 0.10, 0.7, 0.05], 'String', ...
    sprintf('Log-normal (sigma): %s', mat2str(lognormalSigmaValues)), ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'center', 'FontName', fontName);

% Save the multi-panel plot
saveas(gcf, 'multi_distribution_comparison.png');

%% New plot with three panels showing all the distributions
figure('Position', [50, 50, 1500, 600]);
set(gcf, 'Color', 'white');

% Define distribution colors
chiColor = [0.4, 0.8, 0.4];
weibColor = [0.2, 0.6, 0.8];
logNColor = [0.8, 0.4, 0.2];

% Panel 1: Chi-Squared Distributions
subplot(1, 3, 1);
hold on;

% Generate x values for plotting
x = linspace(-5, 15, 1000);

% Plot each Chi-Squared distribution
for i = 1:length(chiSquaredDfValues)
    df = chiSquaredDfValues(i);
    % Calculate mean of chi-squared distribution
    mean_chi = df;
    
    % Generate y values (PDF) for each df
    y = chi2pdf(x + mean_chi, df);
    
    % Plot with varying transparency based on df value
    alpha = 0.3 + 0.7 * (i / length(chiSquaredDfValues));
    plot(x, y, 'Color', [chiColor, alpha], 'LineWidth', 1.5);
end

% Add vertical line at x=0
plot([0, 0], [0, 0.5], 'k--', 'LineWidth', 1.5);

% Improve formatting
title(sprintf('Chi-Squared Distributions\nRMSE: %.4f', chiSquaredResults.RMSE_Overall), 'FontSize', 14);
xlabel('Centered Value', 'FontSize', 12);
ylabel('Probability Density', 'FontSize', 12);
grid on;
xlim([-5, 15]);
ylim([0, 0.5]);

% Create a custom legend for df values
legend_entries = cell(1, length(chiSquaredDfValues));
for i = 1:length(chiSquaredDfValues)
    legend_entries{i} = sprintf('df = %d', chiSquaredDfValues(i));
end
legend(legend_entries, 'Location', 'NorthEast', 'FontSize', 8);

% Panel 2: Weibull Distributions
subplot(1, 3, 2);
hold on;

% Generate x values for plotting
x = linspace(-2, 5, 1000);

% Plot each Weibull distribution
for i = 1:length(weibullShapeValues)
    shape = weibullShapeValues(i);
    scale = 1; % Fixed scale parameter
    
    % Calculate mean of Weibull distribution
    mean_weibull = scale * gamma(1 + 1/shape);
    
    % Generate y values (PDF) for each shape parameter
    y_raw = wblpdf(x + mean_weibull, scale, shape);
    
    % Plot with varying transparency based on shape value
    alpha = 0.3 + 0.7 * (i / length(weibullShapeValues));
    plot(x, y_raw, 'Color', [weibColor, alpha], 'LineWidth', 1.5);
end

% Add vertical line at x=0
plot([0, 0], [0, 1.5], 'k--', 'LineWidth', 1.5);

% Improve formatting
title(sprintf('Weibull Distributions\nRMSE: %.4f', weibullResults.RMSE_Overall), 'FontSize', 14);
xlabel('Centered Value', 'FontSize', 12);
ylabel('Probability Density', 'FontSize', 12);
grid on;
xlim([-2, 5]);
ylim([0, 1.5]);

% Create a custom legend for shape values
legend_entries = cell(1, length(weibullShapeValues));
for i = 1:length(weibullShapeValues)
    legend_entries{i} = sprintf('shape = %.1f', weibullShapeValues(i));
end
legend(legend_entries, 'Location', 'NorthEast', 'FontSize', 8);

% Panel 3: Log-normal Distributions
subplot(1, 3, 3);
hold on;

% Generate x values for plotting
x = linspace(-2, 8, 1000);

% Plot each Log-normal distribution
for i = 1:length(lognormalSigmaValues)
    sigma = lognormalSigmaValues(i);
    mu = -sigma^2/2; % Setting mu so that the mean=1
    
    % Generate y values (PDF) for each sigma parameter
    y_raw = lognpdf(x + 1, mu, sigma); % Shift to center at 0
    
    % Plot with varying transparency based on sigma value
    alpha = 0.3 + 0.7 * (i / length(lognormalSigmaValues));
    plot(x, y_raw, 'Color', [logNColor, alpha], 'LineWidth', 1.5);
end

% Add vertical line at x=0
plot([0, 0], [0, 1.5], 'k--', 'LineWidth', 1.5);

% Improve formatting
title(sprintf('Log-normal Distributions\nRMSE: %.4f', lognormalResults.RMSE_Overall), 'FontSize', 14);
xlabel('Centered Value', 'FontSize', 12);
ylabel('Probability Density', 'FontSize', 12);
grid on;
xlim([-2, 8]);
ylim([0, 1.5]);

% Create a custom legend for sigma values
legend_entries = cell(1, length(lognormalSigmaValues));
for i = 1:length(lognormalSigmaValues)
    legend_entries{i} = sprintf('sigma = %.2f', lognormalSigmaValues(i));
end
legend(legend_entries, 'Location', 'NorthEast', 'FontSize', 8);

% Add an overall title
sgtitle('Distribution Shapes Used in the Analysis', 'FontSize', 16, 'FontWeight', 'bold');

% Save the distribution panel plot
saveas(gcf, 'distribution_shapes_comparison.png');

%% Summary statistics and distribution comparison
fprintf('\n============= SUMMARY OF RESULTS =============\n');
fprintf('Overall RMSE:\n');
fprintf('  Chi-Squared: %.6f\n', chiSquaredResults.RMSE_Overall);
fprintf('  Weibull:     %.6f\n', weibullResults.RMSE_Overall);
fprintf('  Log-normal:  %.6f\n', lognormalResults.RMSE_Overall);
fprintf('  Combined:    %.6f\n', combinedRMSE);

% Calculate skewness statistics
fprintf('\nSkewness Statistics:\n');
fprintf('  Chi-Squared - Mean: %.4f, Min: %.4f, Max: %.4f\n', ...
    mean(chiSquaredResults.MeasuredSkewness), ...
    min(chiSquaredResults.MeasuredSkewness), ...
    max(chiSquaredResults.MeasuredSkewness));
fprintf('  Weibull - Mean: %.4f, Min: %.4f, Max: %.4f\n', ...
    mean(weibullResults.MeasuredSkewness), ...
    min(weibullResults.MeasuredSkewness), ...
    max(weibullResults.MeasuredSkewness));
fprintf('  Log-normal - Mean: %.4f, Min: %.4f, Max: %.4f\n', ...
    mean(lognormalResults.MeasuredSkewness), ...
    min(lognormalResults.MeasuredSkewness), ...
    max(lognormalResults.MeasuredSkewness));

% Save the final results structure
save('multiple_distribution_results.mat', 'testResults', 'chiSquaredResults', 'weibullResults', 'lognormalResults');

fprintf('\nAnalysis complete. Results saved to multiple_distribution_results.mat\n');

%% Create a separate figure with distribution summary table
% Create a new figure for the summary table
figure('Position', [50, 50, 1000, 500]);  % Made wider to accommodate the additional column
set(gcf, 'Color', 'white');

% Create table data
distributionTypes = {};
parameterValues = {};
rmseValues = [];
skewnessValues = [];  % New array for skewness values

% Add Chi-Squared data
for i = 1:length(chiSquaredDfValues)
    distributionTypes{end+1} = 'Chi-Squared';
    parameterValues{end+1} = sprintf('df = %d', chiSquaredDfValues(i));
    rmseValues(end+1) = chiSquaredResults.RMSE_ByParam(i);
    
    % Calculate mean skewness for this parameter
    paramIndices = find(chiSquaredResults.Parameter == chiSquaredDfValues(i));
    meanSkewness = mean(chiSquaredResults.MeasuredSkewness(paramIndices));
    skewnessValues(end+1) = meanSkewness;
end

% Add Weibull data
for i = 1:length(weibullShapeValues)
    distributionTypes{end+1} = 'Weibull';
    parameterValues{end+1} = sprintf('shape = %.1f', weibullShapeValues(i));
    rmseValues(end+1) = weibullResults.RMSE_ByParam(i);
    
    % Calculate mean skewness for this parameter
    paramIndices = find(weibullResults.Parameter == weibullShapeValues(i));
    meanSkewness = mean(weibullResults.MeasuredSkewness(paramIndices));
    skewnessValues(end+1) = meanSkewness;
end

% Add Log-normal data
for i = 1:length(lognormalSigmaValues)
    distributionTypes{end+1} = 'Log-normal';
    parameterValues{end+1} = sprintf('sigma = %.2f', lognormalSigmaValues(i));
    rmseValues(end+1) = lognormalResults.RMSE_ByParam(i);
    
    % Calculate mean skewness for this parameter
    paramIndices = find(lognormalResults.Parameter == lognormalSigmaValues(i));
    meanSkewness = mean(lognormalResults.MeasuredSkewness(paramIndices));
    skewnessValues(end+1) = meanSkewness;
end

% Convert to column cell arrays
distributionTypes = distributionTypes';
parameterValues = parameterValues';
rmseValues = rmseValues';
skewnessValues = skewnessValues';  % Convert to column array

% Format skewness values to 4 decimal places
formattedSkewness = cellfun(@(x) sprintf('%.4f', x), num2cell(skewnessValues), 'UniformOutput', false);

% Create the table
tableData = [distributionTypes, parameterValues, formattedSkewness, num2cell(rmseValues)];
columnNames = {'Distribution Type', 'Parameter', 'Bowley''s Skewness', 'RMSE'};

% Create a uitable
t = uitable('Data', tableData, 'ColumnName', columnNames, ...
            'Position', [50 50 900 400], 'FontName', 'Arial', ...
            'ColumnWidth', {150, 150, 150, 150});
            
% Adjust the position of the table
t.Position = [50 50 900 400];

% Save the figure with the table
saveas(gcf, 'distribution_summary_table.png');

disp(t)