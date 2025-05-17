%% Main script for p-value correction model
% This script fits the model on the entire dataset without cross-validation

config = struct(...
    'sampleSize', 50, ...
    'nPermutations', 1000, ...
    'fixedScale', 3.0, ...
    'maxIterations', 5000, ...
    'generateData', 0, ...
    'outputPath', './results', ...
    'nPValuesTest', 25000, ... % Number of p-values per distribution parameter for testing
    'sampleSizeTest', 100, ...
    'nPermutationsTest', 1000, ...
    'randomSeed', 42, ...
    'distributionMode', 'mixed' ... % Options: 'mixed', 'gamma', 'weibull'
);

% Define distribution parameters to test
testParams = struct(...
    'chiSquaredDfValues', [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48], ...
    'weibullShapeValues', [0.5, 1, 1.5, 2, 3, 5], ...
    'lognormalSigmaValues', [0.1, 0.25, 0.5, 0.75, 1, 1.5] ...
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
    fprintf('Generating data for all skewness values with %d iterations using "%s" distribution mode...\n', ...
        config.maxIterations, config.distributionMode);
    tic;
    allData = generate_training_data(config.maxIterations, config.sampleSize, ...
        config.nPermutations, config.distributionMode);
    elapsedTime = toc;
    fprintf('Data generation complete in %.2f seconds.\n', elapsedTime);
    
    % Save the full dataset with distribution mode in filename
    filename = sprintf('pvalue_correction_full_data_%s.mat', config.distributionMode);
    save(fullfile(config.outputPath, filename), 'allData');
else
    % Try to load dataset with distribution mode in filename
    filename = sprintf('pvalue_correction_full_data_%s.mat', config.distributionMode);
    if exist(fullfile(config.outputPath, filename), 'file')
        fprintf('Loading %s distribution mode data...\n', config.distributionMode);
        load(fullfile(config.outputPath, filename));
    else
        % Fall back to generic filename
        fprintf('Specific distribution mode file not found. Loading generic data file...\n');
        load(fullfile(config.outputPath, 'pvalue_correction_full_data.mat'));
    end
end

%% Display summary statistics
fprintf('\n=== Dataset Summary ===\n');
fprintf('Total number of paired p-values: %d\n', height(allData));
fprintf('Number of significant biased p-values (p < 0.05): %d (%.2f%%)\n', ...
    sum(allData.BiasedPValue < 0.05), 100*sum(allData.BiasedPValue < 0.05)/height(allData));
fprintf('Number of significant unbiased p-values (p < 0.05): %d (%.2f%%)\n', ...
    sum(allData.UnbiasedPValue < 0.05), 100*sum(allData.UnbiasedPValue < 0.05)/height(allData));

% Display distribution types if present
if ismember('DistributionType', allData.Properties.VariableNames)
    distTypes = unique(allData.DistributionType);
    fprintf('\nDistribution Types:\n');
    for i = 1:length(distTypes)
        count = sum(strcmp(allData.DistributionType, distTypes{i}));
        fprintf('%s: %d samples (%.1f%%)\n', distTypes{i}, count, 100*count/height(allData));
    end
end

%% Train final model on all data (without cross-validation)
fprintf('\n=== Training Final Model on All Data ===\n');

% Prepare all data using all four moments
X_all = [allData.BiasedPValue, allData.MeasuredMean, ...
         allData.MeasuredVariance, allData.MeasuredSkewness, ...
         allData.MeasuredKurtosis];
y_all = allData.UnbiasedPValue;

% Prepare polynomial features with all moments
X_all_poly = [X_all, ...
              X_all(:,1).^2, X_all(:,1).^3, ... % BiasedPValue terms
              X_all(:,2).^2, X_all(:,1).*X_all(:,2), ... % Mean terms
              X_all(:,3).^2, X_all(:,1).*X_all(:,3), ... % Variance terms
              X_all(:,4).^2, X_all(:,1).*X_all(:,4), ... % Skewness terms
              X_all(:,5).^2, X_all(:,1).*X_all(:,5)]; % Kurtosis terms

% Train final model
final_mdl_poly = fitlm(X_all_poly, y_all);

% Print model coefficients
fprintf('\n============= MODEL COEFFICIENTS =============\n');
coeffs = final_mdl_poly.Coefficients.Estimate;
featureNames = {'Intercept', 'BiasedPValue', 'MeasuredMean', 'MeasuredVariance', ...
                'MeasuredSkewness', 'MeasuredKurtosis', ...
                'BiasedPValue^2', 'BiasedPValue^3', ...
                'MeasuredMean^2', 'BiasedPValue*MeasuredMean', ...
                'MeasuredVariance^2', 'BiasedPValue*MeasuredVariance', ...
                'MeasuredSkewness^2', 'BiasedPValue*MeasuredSkewness', ...
                'MeasuredKurtosis^2', 'BiasedPValue*MeasuredKurtosis'};

% Display simple equation
fprintf('Polynomial: UnbiasedPValue = %.4f + (%.4f * BiasedPValue) + (%.4f * MeasuredMean) + (%.4f * MeasuredVariance) + (%.4f * MeasuredSkewness) + (%.4f * MeasuredKurtosis) + ...\n', ...
    coeffs(1), coeffs(2), coeffs(3), coeffs(4), coeffs(5), coeffs(6));

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

% Save the final model with distribution mode in filename
modelFilename = sprintf('pvalue_correction_poly_model_%s.mat', config.distributionMode);
save(fullfile(config.outputPath, modelFilename), 'final_mdl_poly');
fprintf('Final polynomial model saved to %s\n', fullfile(config.outputPath, modelFilename));

%% Test with multiple distributions
fprintf('\n=== Testing with Multiple Distributions ===\n');
testResults = test_distributions(final_mdl_poly, ...
    config.nPValuesTest, config.sampleSizeTest, config.nPermutationsTest, ...
    testParams.chiSquaredDfValues, testParams.weibullShapeValues, testParams.lognormalSigmaValues, ...
    'SavePath', config.outputPath, 'Visualize', true);

fprintf('\n=== Analysis complete ===\n');