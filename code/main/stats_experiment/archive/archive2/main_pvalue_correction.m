%% Main script for p-value correction model
% This script fits the model on the entire dataset without cross-validation

%% Configuration
config = struct(...
    'sampleSize', 50, ...         % Sample size per group
    'nPermutations', 1000, ...    % Number of permutations for permutation test
    'fixedScale', 3.0, ...        % Fixed scale parameter for all distributions
    'maxIterations', 10, ...      % Maximum number of iterations for data generation
    'generateData', 1, ...        % Set to 1 to generate new data, 0 to load existing data
    'outputPath', './results', ... % Output path for results and visualizations
    'nPValuesTest', 100, ...      % Number of p-values per distribution parameter for testing
    'sampleSizeTest', 100, ...    % Sample size per group for testing
    'nPermutationsTest', 1000, ... % Number of permutations for testing
    'randomSeed', 42 ...          % Random seed for reproducibility
);

% Define distribution parameters to test
testParams = struct(...
    'chiSquaredDfValues', [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64], ...
    'weibullShapeValues', [0.5, 1, 1.5, 2, 3, 5, 10], ...
    'lognormalSigmaValues', [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2] ...
);

%% Setup environment
rng(config.randomSeed); % Set random seed for reproducibility

% Create output directory if it doesn't exist
if ~exist(config.outputPath, 'dir')
    mkdir(config.outputPath);
    fprintf('Created output directory: %s\n', config.outputPath);
end

% Start parallel pool with optimal configuration
pool = start_parallel_pool('NumWorkers', [], 'ThreadsPerWorker', 1);

%% Generate or load training data
fprintf('=== Data Generation/Loading ===\n');
if config.generateData
    fprintf('Generating data for all skewness values with %d iterations...\n', config.maxIterations);
    tic;
    allData = generate_training_data(config.maxIterations, config.sampleSize, config.nPermutations);
    elapsedTime = toc;
    fprintf('Data generation complete in %.2f seconds.\n', elapsedTime);
    
    % Save the full dataset
    save(fullfile(config.outputPath, 'pvalue_correction_full_data.mat'), 'allData');
else
    fprintf('Loading previously generated data...\n');
    load(fullfile(config.outputPath, 'pvalue_correction_full_data.mat'));
end

%% Display summary statistics
fprintf('\n=== Dataset Summary ===\n');
fprintf('Total number of paired p-values: %d\n', height(allData));
fprintf('Number of significant biased p-values (p < 0.05): %d (%.2f%%)\n', ...
    sum(allData.BiasedPValue < 0.05), 100*sum(allData.BiasedPValue < 0.05)/height(allData));
fprintf('Number of significant unbiased p-values (p < 0.05): %d (%.2f%%)\n', ...
    sum(allData.UnbiasedPValue < 0.05), 100*sum(allData.UnbiasedPValue < 0.05)/height(allData));

%% Train final model on all data (without cross-validation)
fprintf('\n=== Training Final Model on All Data ===\n');

% Prepare all data
X_all = [allData.BiasedPValue, allData.MeasuredSkewness];
y_all = allData.UnbiasedPValue;

% Prepare polynomial features
X_all_poly = [X_all, X_all(:,1).^2, X_all(:,1).^3, ...
            X_all(:,2).^2, X_all(:,1).*X_all(:,2)];

% Train final model
final_mdl_poly = fitlm(X_all_poly, y_all);

% Print model coefficients
fprintf('\n============= MODEL COEFFICIENTS =============\n');
coeffs = final_mdl_poly.Coefficients.Estimate;
featureNames = {'Intercept', 'BiasedPValue', 'MeasuredSkewness', 'BiasedPValue^2', ...
                'BiasedPValue^3', 'MeasuredSkewness^2', 'BiasedPValue*MeasuredSkewness'};

% Display simple equation
fprintf('Polynomial: UnbiasedPValue = %.4f + (%.4f * BiasedPValue) + (%.4f * MeasuredSkewness) + ...\n', ...
    coeffs(1), coeffs(2), coeffs(3));

% Display full equation
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

% Calculate predictions and RMSE for the full dataset
y_pred = predict(final_mdl_poly, X_all_poly);
full_rmse = sqrt(mean((y_pred - y_all).^2));
fprintf('\nRMSE on full dataset: %.6f\n', full_rmse);

% Save the final model
save(fullfile(config.outputPath, 'pvalue_correction_poly_model.mat'), 'final_mdl_poly');
fprintf('Final polynomial model saved to %s\n', fullfile(config.outputPath, 'pvalue_correction_poly_model.mat'));

%% Test with multiple distributions
fprintf('\n=== Testing with Multiple Distributions ===\n');
testResults = test_distributions(final_mdl_poly, ...
    config.nPValuesTest, config.sampleSizeTest, config.nPermutationsTest, ...
    testParams.chiSquaredDfValues, testParams.weibullShapeValues, testParams.lognormalSigmaValues, ...
    'SavePath', config.outputPath, 'Visualize', true);

fprintf('\n=== Analysis complete ===\n');