function [fold_data, rmse_poly_lopo, final_mdl_poly] = run_cross_validation(allData, varargin)
% Performs cross-validation on the p-value correction model with optimized code
%
% Parameters:
%   allData - Table containing all generated data
%   varargin - Optional parameter pairs:
%     'NumFolds': Number of folds (default: 20)
%     'Visualize': Whether to visualize results (default: true)
%     'SavePath': Path to save visualizations (default: current directory)
%     'RunCV': Whether to run cross-validation (default: true)
%
% Returns:
%   fold_data - Cell array containing data for each fold
%   rmse_poly_lopo - Array of RMSE values for each fold
%   final_mdl_poly - Final polynomial model trained on all data

% Parse inputs
p = inputParser;
p.addParameter('NumFolds', 20, @(x) isnumeric(x) && x > 1);
p.addParameter('Visualize', true, @islogical);
p.addParameter('SavePath', '.', @ischar);
p.addParameter('RunCV', true, @islogical);
p.parse(varargin{:});

opts = p.Results;
numFolds = opts.NumFolds;

% Initialize variables for cross-validation
fold_data = cell(numFolds, 1);
rmse_poly_lopo = zeros(numFolds, 1);

% Create stratified folds based on skewness values
[sortedSkewness, sortIdx] = sort(allData.MeasuredSkewness);
numSamples = height(allData);
samplesPerFold = floor(numSamples / numFolds);

% Create indices for each fold using stratified sampling
foldIndices = cell(numFolds, 1);
for i = 1:numFolds
    if i < numFolds
        startIdx = (i-1) * samplesPerFold + 1;
        endIdx = i * samplesPerFold;
        foldIndices{i} = sortIdx(startIdx:endIdx);
    else
        % Last fold gets remaining samples
        startIdx = (i-1) * samplesPerFold + 1;
        foldIndices{i} = sortIdx(startIdx:numSamples);
    end
end

% Perform cross-validation
if opts.RunCV
    fprintf('\nPerforming %d-fold Cross-Validation\n', numFolds);
    fprintf('========================================\n');
    
    % Set up visualization options
    vizOptions = struct();
    vizOptions.visualize = opts.Visualize;
    vizOptions.savePath = opts.SavePath;
    vizOptions.totalFolds = numFolds;
    
    % Process each fold
    for fold = 1:numFolds
        vizOptions.foldNumber = fold;
        
        % Get predictions for this fold
        [~, ~, ~, fold_result] = get_predictions_for_fold(allData, foldIndices{fold}, [], vizOptions);
        
        % Store fold results
        fold_data{fold} = fold_result;
        rmse_poly_lopo(fold) = fold_result.RMSE;
    end
    
    % Display cross-validation results
    fprintf('\nCross-Validation results (RMSE):\n');
    fprintf('Polynomial model: %.6f ± %.6f\n', mean(rmse_poly_lopo), std(rmse_poly_lopo));

    % Create summary visualizations
    if opts.Visualize
        visualize_cv_summary(fold_data, rmse_poly_lopo, opts.SavePath);
    end
end

% Train final model using all data
fprintf('Training final model with all data...\n');

% Prepare all data
X_all = [allData.BiasedPValue, allData.MeasuredSkewness];
y_all = allData.UnbiasedPValue;

% Prepare polynomial features
X_all_poly = [X_all, X_all(:,1).^2, X_all(:,1).^3, ...
            X_all(:,2).^2, X_all(:,1).*X_all(:,2)];

% Train final model
final_mdl_poly = fitlm(X_all_poly, y_all);

% Print model coefficients
print_model_coefficients(final_mdl_poly);

% Save the final model
save('pvalue_correction_poly_model.mat', 'final_mdl_poly');
fprintf('Final polynomial model saved to pvalue_correction_poly_model.mat\n');
end

function print_model_coefficients(model)
% Prints the model coefficients in a readable format
fprintf('\n============= MODEL COEFFICIENTS =============\n');
coeffs = model.Coefficients.Estimate;
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
end

function visualize_cv_summary(fold_data, rmse_values, savePath)
% Creates summary visualizations for cross-validation results
numFolds = length(fold_data);

% Extract data from folds
foldMeanSkewness = zeros(numFolds, 1);
for i = 1:numFolds
    foldMeanSkewness(i) = mean(fold_data{i}.TestSkewness);
end

% Create figure for CV summary
figure('Position', [100, 100, 1200, 600]);

% Plot 1: RMSE by fold
subplot(1, 2, 1);
bar(1:numFolds, rmse_values, 'FaceColor', [0.3, 0.6, 0.8]);
hold on;
plot([0, numFolds+1], [mean(rmse_values), mean(rmse_values)], 'r-', 'LineWidth', 2);
xlabel('Fold');
ylabel('RMSE');
title('RMSE by Cross-Validation Fold');
grid on;

% Plot 2: Mean skewness vs RMSE
subplot(1, 2, 2);
scatter(foldMeanSkewness, rmse_values, 100, 'filled', 'MarkerFaceColor', [0.3, 0.6, 0.8]);
xlabel('Mean Skewness in Test Fold');
ylabel('RMSE');
title('RMSE vs. Mean Skewness in Folds');
grid on;

% Add correlation coefficient
[rho, pval] = corr(foldMeanSkewness, rmse_values);
text(0.05, 0.95, sprintf('r = %.4f (p = %.4f)', rho, pval), ...
    'Units', 'normalized', 'FontSize', 12);

% Save figure
saveas(gcf, fullfile(savePath, 'cross_validation_summary.png'));

% Analyze best and worst performing folds
[~, best_fold_idx] = min(rmse_values);
[~, worst_fold_idx] = max(rmse_values);

fprintf('\nBest performing fold: Fold %d (RMSE = %.6f, Mean Skewness = %.4f)\n', ...
        best_fold_idx, rmse_values(best_fold_idx), foldMeanSkewness(best_fold_idx));
fprintf('Worst performing fold: Fold %d (RMSE = %.6f, Mean Skewness = %.4f)\n', ...
        worst_fold_idx, rmse_values(worst_fold_idx), foldMeanSkewness(worst_fold_idx));

% Create best vs worst comparison
figure('Position', [100, 100, 1200, 800]);

% Compare predictions
subplot(2, 2, 1);
scatter(fold_data{best_fold_idx}.TrueValues, fold_data{best_fold_idx}.Predictions, 20, 'filled', 'MarkerFaceAlpha', 0.5);
hold on;
plot([0, 1], [0, 1], 'r--', 'LineWidth', 1.5);
title(sprintf('Best Fold (Fold %d): Predicted vs. True', best_fold_idx));
xlabel('True P-Value');
ylabel('Predicted P-Value');
grid on;
axis square;
axis([0 1 0 1]);

% Best fold error histogram
subplot(2, 2, 2);
best_errors = fold_data{best_fold_idx}.Predictions - fold_data{best_fold_idx}.TrueValues;
histogram(best_errors, 50);
title(sprintf('Best Fold Error Distribution (RMSE = %.6f)', rmse_values(best_fold_idx)));
xlabel('Error (Predicted - True)');
ylabel('Frequency');
grid on;

% Worst fold predictions
subplot(2, 2, 3);
scatter(fold_data{worst_fold_idx}.TrueValues, fold_data{worst_fold_idx}.Predictions, 20, 'filled', 'MarkerFaceAlpha', 0.5);
hold on;
plot([0, 1], [0, 1], 'r--', 'LineWidth', 1.5);
title(sprintf('Worst Fold (Fold %d): Predicted vs. True', worst_fold_idx));
xlabel('True P-Value');
ylabel('Predicted P-Value');
grid on;
axis square;
axis([0 1 0 1]);

% Worst fold error histogram
subplot(2, 2, 4);
worst_errors = fold_data{worst_fold_idx}.Predictions - fold_data{worst_fold_idx}.TrueValues;
histogram(worst_errors, 50);
title(sprintf('Worst Fold Error Distribution (RMSE = %.6f)', rmse_values(worst_fold_idx)));
xlabel('Error (Predicted - True)');
ylabel('Frequency');
grid on;

sgtitle('Best vs. Worst Fold Comparison: Polynomial Model', 'FontSize', 14);

% Save figure
saveas(gcf, fullfile(savePath, 'best_worst_fold_comparison.png'));
end