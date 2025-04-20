function testResults = test_distributions(final_mdl_poly, nPValuesTest, sampleSizeTest, nPermutationsTest, ...
    chiSquaredDfValues, weibullShapeValues, lognormalSigmaValues, varargin)
% Tests the model with multiple distributions using a more concise implementation
%
% Parameters:
%   final_mdl_poly - Final polynomial model
%   nPValuesTest - Number of p-values per distribution parameter
%   sampleSizeTest - Sample size per group
%   nPermutationsTest - Number of permutations
%   chiSquaredDfValues - Array of chi-squared df values to test
%   weibullShapeValues - Array of Weibull shape parameters to test
%   lognormalSigmaValues - Array of log-normal sigma parameters to test
%   varargin - Optional parameters:
%     'SavePath': Path to save results and visualizations (default: '.')
%     'Visualize': Whether to create visualizations (default: true)
%
% Returns:
%   testResults - Structure containing test results

% Parse optional inputs
p = inputParser;
p.addParameter('SavePath', '.', @ischar);
p.addParameter('Visualize', true, @islogical);
p.parse(varargin{:});
opts = p.Results;

fprintf('Testing with %d Chi-Squared, %d Weibull, and %d Log-normal distribution parameters\n', ...
    length(chiSquaredDfValues), length(weibullShapeValues), length(lognormalSigmaValues));

% Create a structure to store results
testResults = struct();

% Create a parallel pool if not already started
start_parallel_pool();

%% Process distributions using the unified processor
fprintf('\nProcessing Chi-Squared distributions...\n');
chiSquaredResults = process_distribution(final_mdl_poly, nPValuesTest, sampleSizeTest, nPermutationsTest, ...
    chiSquaredDfValues, 'chi', 'Processing Chi-Squared distributions...');
testResults.ChiSquare = chiSquaredResults;

fprintf('\nProcessing Weibull distributions...\n');
weibullResults = process_distribution(final_mdl_poly, nPValuesTest, sampleSizeTest, nPermutationsTest, ...
    weibullShapeValues, 'weibull', 'Processing Weibull distributions...');
testResults.Weibull = weibullResults;

fprintf('\nProcessing Log-normal distributions...\n');
lognormalResults = process_distribution(final_mdl_poly, nPValuesTest, sampleSizeTest, nPermutationsTest, ...
    lognormalSigmaValues, 'lognormal', 'Processing Log-normal distributions...');
testResults.LogNormal = lognormalResults;

%% Visualize combined results if requested
if opts.Visualize
    visualize_test_results(chiSquaredResults, weibullResults, lognormalResults, ...
        chiSquaredDfValues, weibullShapeValues, lognormalSigmaValues, opts.SavePath);
end

%% Summary statistics and distribution comparison
print_summary_statistics(chiSquaredResults, weibullResults, lognormalResults);

% Save the final results structure
save(fullfile(opts.SavePath, 'multiple_distribution_results.mat'), 'testResults');
fprintf('\nResults saved to %s\n', fullfile(opts.SavePath, 'multiple_distribution_results.mat'));

% Table creation section has been removed as requested
end

function print_summary_statistics(chiSquaredResults, weibullResults, lognormalResults)
% Prints summary statistics for all tested distributions

fprintf('\n============= SUMMARY OF RESULTS =============\n');
fprintf('Overall RMSE:\n');
fprintf('  Chi-Squared: %.6f\n', chiSquaredResults.RMSE_Overall);
fprintf('  Weibull:     %.6f\n', weibullResults.RMSE_Overall);
fprintf('  Log-normal:  %.6f\n', lognormalResults.RMSE_Overall);

% Calculate combined RMSE
chiCount = length(chiSquaredResults.UnbiasedPValues);
weibCount = length(weibullResults.UnbiasedPValues);
logNCount = length(lognormalResults.UnbiasedPValues);
totalCount = chiCount + weibCount + logNCount;

combinedRMSE = sqrt((chiSquaredResults.RMSE_Overall^2 * chiCount + ...
                    weibullResults.RMSE_Overall^2 * weibCount + ...
                    lognormalResults.RMSE_Overall^2 * logNCount) / totalCount);
                
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
end