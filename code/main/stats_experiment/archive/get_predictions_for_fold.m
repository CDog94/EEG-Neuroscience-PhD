function [X_test, y_test, y_pred, fold_data] = get_predictions_for_fold(allData, testIndices, final_mdl_poly, options)
% Gets predictions for a specific fold in cross-validation and optionally visualizes results
%
% Parameters:
%   allData - Table containing all generated data
%   testIndices - Indices of test data in the fold
%   final_mdl_poly - (Optional) Final model to use for prediction
%   options - (Optional) Struct with visualization options:
%     .visualize: Whether to create visualizations (default: false)
%     .foldNumber: Current fold number for labeling
%     .totalFolds: Total number of folds
%     .savePath: Path to save visualizations
%
% Returns:
%   X_test - Test features
%   y_test - True test values
%   y_pred - Predicted test values
%   fold_data - Struct with fold data (including RMSE)

% Set default options
if nargin < 4
    options = struct();
end
if ~isfield(options, 'visualize')
    options.visualize = false;
end

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
if nargin < 3 || isempty(final_mdl_poly)
    mdl = fitlm(X_train_poly, y_train);
    y_pred = predict(mdl, X_test_poly);
else
    mdl = final_mdl_poly; % Use provided model
    y_pred = predict(mdl, X_test_poly);
end

% Calculate RMSE and prepare fold data structure
rmse = sqrt(mean((y_pred - y_test).^2));

% Create fold data structure
fold_data = struct();
fold_data.TestIndices = testIndices;
fold_data.TestSkewness = testData.MeasuredSkewness;
fold_data.RMSE = rmse;
fold_data.Predictions = y_pred;
fold_data.TrueValues = y_test;
fold_data.TestData = testData;
fold_data.Model = mdl;

% Calculate skewness range and statistics
minSkew = min(testData.MeasuredSkewness);
maxSkew = max(testData.MeasuredSkewness);
meanSkew = mean(testData.MeasuredSkewness);

% Print fold information
if isfield(options, 'foldNumber') && isfield(options, 'totalFolds')
    fprintf('Processing fold %d/%d...\n', options.foldNumber, options.totalFolds);
end
fprintf('Test set skewness range: %.4f to %.4f (mean: %.4f)\n', ...
    minSkew, maxSkew, meanSkew);
fprintf('Data split: %d samples for training, %d samples for testing\n', ...
    height(allData) - length(testIndices), length(testIndices));
fprintf('RMSE for this fold: %.6f\n', rmse);

% Create visualization if requested
if options.visualize
    visualize_fold_results(fold_data, options);
end
end

function visualize_fold_results(fold_data, options)
% Create visualizations for fold results
%
% Parameters:
%   fold_data - Struct with fold data
%   options - Visualization options

% Create figure with appropriate title
if isfield(options, 'foldNumber')
    figTitle = sprintf('Fold %d Results', options.foldNumber);
else
    figTitle = 'Fold Results';
end

figure('Position', [100, 100, 1000, 800]);

% Plot 1: Predicted vs. True
subplot(2, 2, 1);
scatter(fold_data.TrueValues, fold_data.Predictions, 20, 'filled', 'MarkerFaceAlpha', 0.5);
hold on;
plot([0, 1], [0, 1], 'r--', 'LineWidth', 1.5);
title('Predicted vs. True P-Values');
xlabel('True P-Value');
ylabel('Predicted P-Value');
grid on;
axis square;
axis([0 1 0 1]);

% Plot 2: Error histogram
subplot(2, 2, 2);
errors = fold_data.Predictions - fold_data.TrueValues;
histogram(errors, 50);
title(sprintf('Error Distribution (RMSE = %.6f)', fold_data.RMSE));
xlabel('Error (Predicted - True)');
ylabel('Frequency');
grid on;

% Plot 3: Error vs. Skewness
subplot(2, 2, 3);
scatter(fold_data.TestSkewness, abs(errors), 20, 'filled', 'MarkerFaceAlpha', 0.5);
title('Absolute Error vs. Skewness');
xlabel('Measured Skewness');
ylabel('Absolute Error');
grid on;

% Plot 4: Error vs. Biased P-Value
subplot(2, 2, 4);
scatter(fold_data.TestData.BiasedPValue, abs(errors), 20, 'filled', 'MarkerFaceAlpha', 0.5);
title('Absolute Error vs. Biased P-Value');
xlabel('Biased P-Value');
ylabel('Absolute Error');
grid on;

% Overall title
sgtitle([figTitle, sprintf(' (RMSE: %.6f)', fold_data.RMSE)], 'FontSize', 14, 'FontWeight', 'bold');

% Save figure if path provided
if isfield(options, 'savePath') && ~isempty(options.savePath)
    if isfield(options, 'foldNumber')
        filename = sprintf('%s/fold_%d_results.png', options.savePath, options.foldNumber);
    else
        filename = sprintf('%s/fold_results.png', options.savePath);
    end
    saveas(gcf, filename);
    fprintf('Saved visualization to %s\n', filename);
end
end