function results = process_distribution(final_mdl_poly, nPValuesTest, sampleSizeTest, nPermutationsTest, ...
                               paramValues, distributionType, waitbarTitle)
% Process any distribution type with parameters
% Parameters:
%   final_mdl_poly - Final polynomial model
%   nPValuesTest - Number of p-values per parameter
%   sampleSizeTest - Sample size per group
%   nPermutationsTest - Number of permutations
%   paramValues - Array of parameter values to test
%   distributionType - String indicating distribution type ('chi', 'weibull', 'lognormal')
%   waitbarTitle - Title for the waitbar

% Initialize cell array for parallel results
results_cell = cell(length(paramValues), 1);

% Create a waitbar
h = waitbar(0, waitbarTitle, 'Name', 'Progress');

% Create a parallel.pool.DataQueue to track progress
dataQueue = parallel.pool.DataQueue;

% Set up a listener to update the waitbar
count = 0;
numParams = length(paramValues);
afterEach(dataQueue, @(~) updateWaitbar(h, numParams, count));

% Process each parameter in parallel
parfor p = 1:length(paramValues)
    param = paramValues(p);
    fprintf('Processing %s param=%.2f...\n', distributionType, param);
    
    % Initialize arrays to collect data for this parameter
    biasedPVals = zeros(nPValuesTest, 1);
    unbiasedPVals = zeros(nPValuesTest, 1);
    measuredMeans = zeros(nPValuesTest, 1);
    measuredVariances = zeros(nPValuesTest, 1);
    measuredSkewness = zeros(nPValuesTest, 1);
    measuredKurtosis = zeros(nPValuesTest, 1);
    measuredBowleySkewness = zeros(nPValuesTest, 1);
    
    % Pre-generate all uniform data for unbiased p-values
    uniformData = unifrnd(-0.5, 0.5, sampleSizeTest, nPValuesTest);
    
    % Group 1 is all zeros
    group1 = zeros(sampleSizeTest, 1);
    
    % Generate data and process all p-values
    for i = 1:nPValuesTest
        % Generate appropriate distribution data based on distribution type
        switch lower(distributionType)
            case 'chi'
                group2 = chi2rnd(param, sampleSizeTest, 1) - param;
            case 'weibull'
                weibullMean = gamma(1 + 1/param);
                group2 = wblrnd(1, param, sampleSizeTest, 1) - weibullMean;
            case 'lognormal'
                mu = -param^2/2;
                group2 = lognrnd(mu, param, sampleSizeTest, 1) - 1;
            otherwise
                error('Unknown distribution type: %s', distributionType);
        end
        
        % Calculate all moments
        moments = calculate_moments(group2);
        measuredMeans(i) = moments.mean;
        measuredVariances(i) = moments.variance;
        measuredSkewness(i) = moments.skewness;
        measuredKurtosis(i) = moments.kurtosis;
        measuredBowleySkewness(i) = moments.bowleySkewness;
        
        % Perform permutation tests
        biasedPVals(i) = permutation_test(group1, group2, nPermutationsTest);
        unbiasedPVals(i) = permutation_test(group1, uniformData(:, i), nPermutationsTest);
    end
    
    % Sort the p-values independently
    biasedPVals = sort(biasedPVals);
    unbiasedPVals = sort(unbiasedPVals);
    
    % Sort the moments based on biased p-values for consistency
    [~, sortIdx] = sort(biasedPVals);
    measuredMeans = measuredMeans(sortIdx);
    measuredVariances = measuredVariances(sortIdx);
    measuredSkewness = measuredSkewness(sortIdx);
    measuredKurtosis = measuredKurtosis(sortIdx);
    measuredBowleySkewness = measuredBowleySkewness(sortIdx);
    
    % Prepare feature vector for prediction using all moments
    X_test = [biasedPVals, measuredMeans, measuredVariances, ...
              measuredSkewness, measuredKurtosis, measuredBowleySkewness];
    
    X_test_poly = [X_test, ...
                  X_test(:,1).^2, X_test(:,1).^3, ... % BiasedPValue terms
                  X_test(:,2).^2, X_test(:,1).*X_test(:,2), ... % Mean terms
                  X_test(:,3).^2, X_test(:,1).*X_test(:,3), ... % Variance terms
                  X_test(:,4).^2, X_test(:,1).*X_test(:,4), ... % Skewness terms
                  X_test(:,5).^2, X_test(:,1).*X_test(:,5), ... % Kurtosis terms
                  X_test(:,6).^2, X_test(:,1).*X_test(:,6)]; % BowleySkewness terms
    
    % Use the model to predict
    predictedPVals = predict(final_mdl_poly, X_test_poly);
    
    % Calculate local RMSE
    local_errors = predictedPVals - unbiasedPVals;
    local_RMSE = sqrt(mean(local_errors.^2));
    
    % Create a structure for results
    local_results = struct();
    local_results.BiasedPValues = biasedPVals;
    local_results.UnbiasedPValues = unbiasedPVals;
    local_results.PredictedPValues = predictedPVals;
    local_results.MeasuredMean = measuredMeans;
    local_results.MeasuredVariance = measuredVariances;
    local_results.MeasuredSkewness = measuredSkewness;
    local_results.MeasuredKurtosis = measuredKurtosis;
    local_results.MeasuredBowleySkewness = measuredBowleySkewness;
    local_results.Parameter = repmat(param, length(biasedPVals), 1);
    local_results.RMSE = local_RMSE;
    
    % Store data in cell array
    results_cell{p} = local_results;
    
    fprintf('RMSE for %s param=%.2f: %.6f\n', distributionType, param, local_RMSE);
    
    % Send update to the dataQueue
    send(dataQueue, p);
end

% Close the waitbar
close(h);

% Combine all results
results = struct();
results.BiasedPValues = [];
results.UnbiasedPValues = [];
results.PredictedPValues = [];
results.MeasuredMean = [];
results.MeasuredVariance = [];
results.MeasuredSkewness = [];
results.MeasuredKurtosis = [];
results.MeasuredBowleySkewness = [];
results.Parameter = [];
results.RMSE_ByParam = zeros(length(paramValues), 1);

for p = 1:length(paramValues)
    results.BiasedPValues = [results.BiasedPValues; results_cell{p}.BiasedPValues];
    results.UnbiasedPValues = [results.UnbiasedPValues; results_cell{p}.UnbiasedPValues];
    results.PredictedPValues = [results.PredictedPValues; results_cell{p}.PredictedPValues];
    results.MeasuredMean = [results.MeasuredMean; results_cell{p}.MeasuredMean];
    results.MeasuredVariance = [results.MeasuredVariance; results_cell{p}.MeasuredVariance];
    results.MeasuredSkewness = [results.MeasuredSkewness; results_cell{p}.MeasuredSkewness];
    results.MeasuredKurtosis = [results.MeasuredKurtosis; results_cell{p}.MeasuredKurtosis];
    results.MeasuredBowleySkewness = [results.MeasuredBowleySkewness; results_cell{p}.MeasuredBowleySkewness];
    results.Parameter = [results.Parameter; results_cell{p}.Parameter];
    results.RMSE_ByParam(p) = results_cell{p}.RMSE;
end

% Calculate overall RMSE
errors = results.PredictedPValues - results.UnbiasedPValues;
results.RMSE_Overall = sqrt(mean(errors.^2));
end

function updateWaitbar(h, numParams, countVal)
    persistent count
    if isempty(count)
        count = 0;
    end
    count = count + 1;
    waitbar(count/numParams, h, sprintf('Processing: %d/%d complete (%.1f%%)', ...
        count, numParams, (count/numParams)*100));
end