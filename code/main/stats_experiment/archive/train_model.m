% Script to train a p-value correction model for skewness with SVR added

%% Load the training data
% If you're running this after generating the data:
load('pvalue_correction_training_data.mat');
% Or use the generateTrainingData function directly for testing

%% Split data into training and test sets (80% train, 20% test)
rng(42); % Set random seed for reproducibility
numSamples = height(trainingData);
testFraction = 0.2;
testSize = round(testFraction * numSamples);

% Create random indices for test set
shuffledIndices = randperm(numSamples);
testIndices = shuffledIndices(1:testSize);
trainIndices = shuffledIndices(testSize+1:end);

% Split the data
testData = trainingData(testIndices, :);
trainData = trainingData(trainIndices, :);

fprintf('Data split: %d samples for training, %d samples for testing\n', ...
    height(trainData), height(testData));

%% Prepare for k-fold cross-validation on TRAINING data only
k = 4;  % 4-fold cross-validation as specified
cv = cvpartition(height(trainData), 'KFold', k);

% Initialize arrays to store results
rmse_linear = zeros(k, 1);
rmse_poly = zeros(k, 1);
rmse_svr = zeros(k, 1);  % Added for SVR

%% Create a waitbar for cross-validation
h_cv = waitbar(0, 'Starting cross-validation...', 'Name', 'Cross-Validation Progress');

%% Perform cross-validation
for fold = 1:k
    waitbar((fold-1)/k, h_cv, sprintf('Processing fold %d of %d...', fold, k));
    fprintf('Processing fold %d of %d...\n', fold, k);
    
    % Get train and validation indices for this fold
    foldTrainIdx = cv.training(fold);
    foldValIdx = cv.test(fold);
    
    foldTrainData = trainData(foldTrainIdx, :);
    foldValData = trainData(foldValIdx, :);
    
    % Prepare input features and target variable
    X_foldTrain = [foldTrainData.BiasedPValue, foldTrainData.SkewnessParam];
    y_foldTrain = foldTrainData.UnbiasedPValue;
    
    X_foldVal = [foldValData.BiasedPValue, foldValData.SkewnessParam];
    y_foldVal = foldValData.UnbiasedPValue;
    
    % Model 1: Linear Regression
    mdl_linear = fitlm(X_foldTrain, y_foldTrain);
    y_pred_linear = predict(mdl_linear, X_foldVal);
    rmse_linear(fold) = sqrt(mean((y_pred_linear - y_foldVal).^2));
    
    % Model 2: Polynomial Regression (include interaction and higher order terms)
    X_foldTrain_poly = [X_foldTrain, X_foldTrain(:,1).^2, X_foldTrain(:,1).^3, ...
                    X_foldTrain(:,2).^2, X_foldTrain(:,1).*X_foldTrain(:,2)];
    X_foldVal_poly = [X_foldVal, X_foldVal(:,1).^2, X_foldVal(:,1).^3, ...
                   X_foldVal(:,2).^2, X_foldVal(:,1).*X_foldVal(:,2)];
    
    mdl_poly = fitlm(X_foldTrain_poly, y_foldTrain);
    y_pred_poly = predict(mdl_poly, X_foldVal_poly);
    rmse_poly(fold) = sqrt(mean((y_pred_poly - y_foldVal).^2));
    
    % Model 3: Support Vector Regression (SVR)
    % Scale features for SVR
    X_foldTrain_scaled = normalize(X_foldTrain);
    X_foldVal_scaled = normalize(X_foldVal);
    
    % Train SVR model with epsilon-SVR and Gaussian kernel
    mdl_svr = fitrsvm(X_foldTrain_scaled, y_foldTrain, ...
                     'KernelFunction', 'gaussian', ...
                     'Standardize', true, ...
                     'Epsilon', 0.01, ...  % Epsilon parameter for epsilon-SVR
                     'KernelScale', 'auto');  % Let MATLAB determine optimal scale
    
    y_pred_svr = predict(mdl_svr, X_foldVal_scaled);
    rmse_svr(fold) = sqrt(mean((y_pred_svr - y_foldVal).^2));
    
    % Update waitbar with completion percentage
    waitbar(fold/k, h_cv, sprintf('Completed fold %d of %d', fold, k));
end

% Close the cross-validation waitbar
close(h_cv);

%% Display cross-validation results
fprintf('\nCross-validation results (RMSE):\n');
fprintf('Linear model:         %.6f ± %.6f\n', mean(rmse_linear), std(rmse_linear));
fprintf('Polynomial model:     %.6f ± %.6f\n', mean(rmse_poly), std(rmse_poly));
fprintf('SVR model:            %.6f ± %.6f\n', mean(rmse_svr), std(rmse_svr));

%% Determine best model type
model_means = [mean(rmse_linear), mean(rmse_poly), mean(rmse_svr)];
[~, best_model_idx] = min(model_means);
model_names = {'Linear', 'Polynomial', 'SVR'};
fprintf('\nBest model based on CV: %s\n', model_names{best_model_idx});

%% Train final model using all training data
fprintf('\nTraining final model with all training data...\n');

% Prepare training data
X_train = [trainData.BiasedPValue, trainData.SkewnessParam];
y_train = trainData.UnbiasedPValue;

% Prepare test data
X_test = [testData.BiasedPValue, testData.SkewnessParam];
y_test = testData.UnbiasedPValue;

% Scale data for SVR
X_train_scaled = normalize(X_train);
X_test_scaled = normalize(X_test);

% Prepare polynomial features if needed
X_train_poly = [X_train, X_train(:,1).^2, X_train(:,1).^3, ...
              X_train(:,2).^2, X_train(:,1).*X_train(:,2)];
X_test_poly = [X_test, X_test(:,1).^2, X_test(:,1).^3, ...
              X_test(:,2).^2, X_test(:,1).*X_test(:,2)];

% Train final models
final_mdl_linear = fitlm(X_train, y_train);
final_mdl_poly = fitlm(X_train_poly, y_train);

% Train final SVR model
fprintf('Training SVR model...\n');
final_mdl_svr = fitrsvm(X_train_scaled, y_train, ...
                       'KernelFunction', 'gaussian', ...
                       'Standardize', true, ...
                       'Epsilon', 0.01, ...
                       'KernelScale', 'auto');
fprintf('SVR model training complete.\n');

%% Print the model coefficients and equation
fprintf('\n============= MODEL COEFFICIENTS =============\n');

% Linear model coefficients
fprintf('\n--- Linear Model Equation ---\n');
fprintf('UnbiasedPValue = %.6f + (%.6f * BiasedPValue) + (%.6f * SkewnessParam)\n', ...
    final_mdl_linear.Coefficients.Estimate(1), ...
    final_mdl_linear.Coefficients.Estimate(2), ...
    final_mdl_linear.Coefficients.Estimate(3));

% Display detailed coefficient table
fprintf('\nLinear Model Coefficient Table:\n');
disp(final_mdl_linear.Coefficients);

% Polynomial model coefficients
fprintf('\n--- Polynomial Model Equation ---\n');
coeffs = final_mdl_poly.Coefficients.Estimate;
fprintf('UnbiasedPValue = %.6f + (%.6f * BiasedPValue) + (%.6f * SkewnessParam) + (%.6f * BiasedPValue²) + (%.6f * BiasedPValue³) + (%.6f * SkewnessParam²) + (%.6f * BiasedPValue*SkewnessParam)\n', ...
    coeffs(1), coeffs(2), coeffs(3), coeffs(4), coeffs(5), coeffs(6), coeffs(7));

% Display detailed coefficient table
fprintf('\nPolynomial Model Coefficient Table:\n');
disp(final_mdl_poly.Coefficients);

% Display variable names for polynomial model for clarity
fprintf('\nPolynomial Model Variable Names:\n');
disp(final_mdl_poly.CoefficientNames);

% SVR model information
fprintf('\n--- SVR Model Information ---\n');
fprintf('SVR Model Type: Epsilon-SVR with Gaussian Kernel\n');
fprintf('Number of Support Vectors: %d\n', final_mdl_svr.NumObservations);
fprintf('Epsilon: %.4f\n', final_mdl_svr.Epsilon);
fprintf('Kernel Scale: %.4f\n', final_mdl_svr.KernelParameters.Scale);

fprintf('==============================================\n\n');

% Select the best model based on cross-validation
switch best_model_idx
    case 1  % Linear
        correctionFcn = @(p, skewness) predictPValueCorrectionLinear(p, skewness, final_mdl_linear);
        best_model_name = 'Linear';
        final_mdl = final_mdl_linear;
        
        % Make predictions on test set
        testPredictions = predict(final_mdl_linear, X_test);
    case 2  % Polynomial
        correctionFcn = @(p, skewness) predictPValueCorrectionPoly(p, skewness, final_mdl_poly);
        best_model_name = 'Polynomial';
        final_mdl = final_mdl_poly;
        
        % Make predictions on test set
        testPredictions = predict(final_mdl_poly, X_test_poly);
    case 3  % SVR
        correctionFcn = @(p, skewness) predictPValueCorrectionSVR(p, skewness, final_mdl_svr);
        best_model_name = 'SVR';
        final_mdl = final_mdl_svr;
        
        % Make predictions on test set
        testPredictions = predict(final_mdl_svr, X_test_scaled);
end

% Save the final models
save('pvalue_correction_models.mat', 'final_mdl_linear', 'final_mdl_poly', 'final_mdl_svr', 'best_model_idx', 'best_model_name');
fprintf('Final models saved to pvalue_correction_models.mat\n');

%% Evaluate model on test set
fprintf('\nEvaluating final %s model on test set...\n', best_model_name);

% Calculate errors
test_errors = testPredictions - y_test;

% Calculate RMSE on test set
test_rmse = sqrt(mean(test_errors.^2));
fprintf('Test RMSE: %.6f\n', test_rmse);

% Calculate RMSE for significant p-values (p < 0.05)
sig_idx = testData.BiasedPValue < 0.05;
if any(sig_idx)
    sig_rmse = sqrt(mean((testPredictions(sig_idx) - y_test(sig_idx)).^2));
    fprintf('RMSE for p < 0.05: %.6f\n', sig_rmse);
else
    fprintf('No p-values < 0.05 in test set\n');
end

%% Visualize predictions vs ground truth on test set
fprintf('\nGenerating visualization plots for test set...\n');

% Sort data by ground truth for clearer visualization
[sorted_ground_truth, sort_idx] = sort(y_test);
sorted_predictions = testPredictions(sort_idx);
sorted_diff = test_errors(sort_idx);

% Create figure for visualization
figure('Name', ['P-Value Correction: ', best_model_name, ' Model Evaluation on Test Set'], 'Position', [100, 100, 1200, 800]);

% 1. Ground Truth vs Predictions
subplot(2, 3, 1);
plot(sorted_ground_truth, 'b-', 'LineWidth', 1.5);
hold on;
plot(sorted_predictions, 'r-', 'LineWidth', 1);
hold off;
title('Ground Truth vs Predicted Values (Sorted)');
xlabel('Sample Index (Sorted by Ground Truth)');
ylabel('P-Value');
legend('Ground Truth', 'Predictions');
grid on;

% 2. Ground Truth vs Predictions Scatter Plot
subplot(2, 3, 2);
scatter(y_test, testPredictions, 20, 'filled');
hold on;
plot([0, 1], [0, 1], 'k--', 'LineWidth', 1.5);  % Perfect prediction line
hold off;
title('Scatter Plot: Ground Truth vs Predictions');
xlabel('Ground Truth P-Value');
ylabel('Predicted P-Value');
grid on;
axis square;
axis([0 1 0 1]);

% 3. Prediction Error Histogram
subplot(2, 3, 3);
histogram(test_errors, 50);
title('Histogram of Prediction Errors');
xlabel('Prediction Error (Predicted - Ground Truth)');
ylabel('Frequency');
grid on;

% 4. Error by Ground Truth Value
subplot(2, 3, 4);
scatter(y_test, test_errors, 20, testData.SkewnessParam, 'filled');
hold on;
plot([0, 1], [0, 0], 'k--', 'LineWidth', 1.5);  % Zero error line
hold off;
title('Prediction Error by Ground Truth Value');
xlabel('Ground Truth P-Value');
ylabel('Prediction Error');
c = colorbar;
c.Label.String = 'Skewness Parameter';
grid on;

% 5. Model Comparison (RMSE by model type)
subplot(2, 3, 5);
bar([mean(rmse_linear), mean(rmse_poly), mean(rmse_svr)]);
hold on;
errorbar(1:3, [mean(rmse_linear), mean(rmse_poly), mean(rmse_svr)], ...
         [std(rmse_linear), std(rmse_poly), std(rmse_svr)], 'k.');
hold off;
title('Model Comparison (Cross-Validation)');
xlabel('Model Type');
ylabel('RMSE');
set(gca, 'XTickLabel', {'Linear', 'Polynomial', 'SVR'});
grid on;

% % 6. Important p-values focus (p < 0.1)
% subplot(2, 3, 6);
% important_idx = y_test < 0.1;
% scatter(y_test(important_idx), testPredictions(important_idx), 30, 'filled');
% hold on;
% plot([0, 1], [0, 1], 'k--', 'LineWidth', 1.5);  % Perfect prediction line
% hold off;
% title('Focus on Important P-Values (< 0.1)');
% xlabel('Ground Truth P-Value');
% ylabel('Predicted P-Value');
% grid on;
% axis square;
% axis([0 0.1 0 0.1]);

% Adjust layout
sgtitle(['P-Value Correction: ', best_model_name, ' Model Performance on Test Set'], 'FontSize', 14);
set(gcf, 'Color', 'white');

% Save the figure
saveas(gcf, 'pvalue_correction_model_test_evaluation.png');
fprintf('Visualization saved to pvalue_correction_model_test_evaluation.png\n');

%% Define prediction functions for each model type

% Function for linear model predictions
function corrected_p = predictPValueCorrectionLinear(p, skewness, mdl)
    X_pred = [p, skewness];
    corrected_p = predict(mdl, X_pred);
    % Ensure p-values are in valid range [0,1]
    corrected_p = max(0, min(1, corrected_p));
end

% Function for polynomial model predictions
function corrected_p = predictPValueCorrectionPoly(p, skewness, mdl)
    X_pred = [p, skewness, p^2, p^3, skewness^2, p*skewness];
    corrected_p = predict(mdl, X_pred);
    % Ensure p-values are in valid range [0,1]
    corrected_p = max(0, min(1, corrected_p));
end

% Function for SVR model predictions
function corrected_p = predictPValueCorrectionSVR(p, skewness, mdl)
    % Normalize input features similar to training
    X_pred = normalize([p, skewness]);
    corrected_p = predict(mdl, X_pred);
    % Ensure p-values are in valid range [0,1]
    corrected_p = max(0, min(1, corrected_p));
end

% Utility function for feature normalization
function X_scaled = normalize(X)
    % Simple Z-score normalization
    % Note: In production, you should use same mean/std from training data
    X_scaled = (X - mean(X)) ./ std(X);
    % Handle any NaN values that might appear if std is 0
    X_scaled(isnan(X_scaled)) = 0;
end