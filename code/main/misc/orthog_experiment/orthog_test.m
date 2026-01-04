% Script to compare regression betas under different collinearity levels
% Comparing: Univariate vs Multivariate models, Original vs Orthogonalized
% NOW WITH: t-tests for coefficients and F-tests for model significance
n = 1000;  % number of observations
true_beta = [2; -1.5; 3];  % true coefficients

%% SCENARIO 1: HIGH COLLINEARITY
fprintf('=== SCENARIO 1: HIGH COLLINEARITY ===\n');

% Create predictors with STRONG correlation
x1_high = randn(n, 1);
x2_high = 0.8*x1_high + sqrt(1-0.8^2)*randn(n, 1);  % r ≈ 0.8
x3_high = 0.6*x1_high + 0.5*x2_high + sqrt(1-0.6^2-0.5^2)*randn(n, 1);

X_high = [x1_high, x2_high, x3_high];
y_high = X_high * true_beta + 0.5*randn(n, 1);

% Apply Gram-Schmidt orthogonalization
X_high_orth = zeros(size(X_high));
X_high_orth(:,1) = X_high(:,1);

for k = 2:size(X_high, 2)
    X_high_orth(:,k) = X_high(:,k);
    for j = 1:k-1
        proj = (X_high(:,k)' * X_high_orth(:,j)) / (X_high_orth(:,j)' * X_high_orth(:,j));
        X_high_orth(:,k) = X_high_orth(:,k) - proj * X_high_orth(:,j);
    end
end

% Initialize storage for statistics
% Structure: [beta, tStat, pValue] for each predictor
stats_univ_orig_high = zeros(3, 3);
stats_univ_orth_high = zeros(3, 3);

% ORIGINAL PREDICTORS - Univariate Models
for i = 1:3
    mdl = fitlm(X_high(:,i), y_high);
    stats_univ_orig_high(i, 1) = mdl.Coefficients.Estimate(2);
    stats_univ_orig_high(i, 2) = mdl.Coefficients.tStat(2);
    stats_univ_orig_high(i, 3) = mdl.Coefficients.pValue(2);
end

% ORIGINAL PREDICTORS - Multivariate Model
mdl_multi_orig_high = fitlm(X_high, y_high);
stats_multi_orig_high = [mdl_multi_orig_high.Coefficients.Estimate(2:4), ...
                         mdl_multi_orig_high.Coefficients.tStat(2:4), ...
                         mdl_multi_orig_high.Coefficients.pValue(2:4)];
fstat_multi_orig_high = mdl_multi_orig_high.ModelFitVsNullModel.Fstat;
fpval_multi_orig_high = mdl_multi_orig_high.ModelFitVsNullModel.Pvalue;
rsq_multi_orig_high = mdl_multi_orig_high.Rsquared.Ordinary;

% ORTHOGONALIZED PREDICTORS - Univariate Models
for i = 1:3
    mdl = fitlm(X_high_orth(:,i), y_high);
    stats_univ_orth_high(i, 1) = mdl.Coefficients.Estimate(2);
    stats_univ_orth_high(i, 2) = mdl.Coefficients.tStat(2);
    stats_univ_orth_high(i, 3) = mdl.Coefficients.pValue(2);
end

% ORTHOGONALIZED PREDICTORS - Multivariate Model
mdl_multi_orth_high = fitlm(X_high_orth, y_high);
stats_multi_orth_high = [mdl_multi_orth_high.Coefficients.Estimate(2:4), ...
                         mdl_multi_orth_high.Coefficients.tStat(2:4), ...
                         mdl_multi_orth_high.Coefficients.pValue(2:4)];
fstat_multi_orth_high = mdl_multi_orth_high.ModelFitVsNullModel.Fstat;
fpval_multi_orth_high = mdl_multi_orth_high.ModelFitVsNullModel.Pvalue;
rsq_multi_orth_high = mdl_multi_orth_high.Rsquared.Ordinary;

% Display results - HIGH COLLINEARITY
fprintf('\nPredictor Correlations (X1-X2, X1-X3, X2-X3): [%.3f, %.3f, %.3f]\n', ...
    corr(X_high(:,1), X_high(:,2)), corr(X_high(:,1), X_high(:,3)), corr(X_high(:,2), X_high(:,3)));

fprintf('\n--- BETA COEFFICIENTS ---\n');
fprintf('%-30s X1: %7.3f  X2: %7.3f  X3: %7.3f\n', ...
    'Univariate Original:', stats_univ_orig_high(1,1), stats_univ_orig_high(2,1), stats_univ_orig_high(3,1));
fprintf('%-30s X1: %7.3f  X2: %7.3f  X3: %7.3f\n', ...
    'Multivariate Original:', stats_multi_orig_high(1,1), stats_multi_orig_high(2,1), stats_multi_orig_high(3,1));
fprintf('%-30s X1: %7.3f  X2: %7.3f  X3: %7.3f\n', ...
    'Univariate Orthogonalized:', stats_univ_orth_high(1,1), stats_univ_orth_high(2,1), stats_univ_orth_high(3,1));
fprintf('%-30s X1: %7.3f  X2: %7.3f  X3: %7.3f\n', ...
    'Multivariate Orthogonalized:', stats_multi_orth_high(1,1), stats_multi_orth_high(2,1), stats_multi_orth_high(3,1));

fprintf('\n--- T-STATISTICS (against zero) ---\n');
fprintf('%-30s X1: %7.2f  X2: %7.2f  X3: %7.2f\n', ...
    'Univariate Original:', stats_univ_orig_high(1,2), stats_univ_orig_high(2,2), stats_univ_orig_high(3,2));
fprintf('%-30s X1: %7.2f  X2: %7.2f  X3: %7.2f\n', ...
    'Multivariate Original:', stats_multi_orig_high(1,2), stats_multi_orig_high(2,2), stats_multi_orig_high(3,2));
fprintf('%-30s X1: %7.2f  X2: %7.2f  X3: %7.2f\n', ...
    'Univariate Orthogonalized:', stats_univ_orth_high(1,2), stats_univ_orth_high(2,2), stats_univ_orth_high(3,2));
fprintf('%-30s X1: %7.2f  X2: %7.2f  X3: %7.2f\n', ...
    'Multivariate Orthogonalized:', stats_multi_orth_high(1,2), stats_multi_orth_high(2,2), stats_multi_orth_high(3,2));

fprintf('\n--- P-VALUES ---\n');
fprintf('%-30s X1: %.2e  X2: %.2e  X3: %.2e\n', ...
    'Univariate Original:', stats_univ_orig_high(1,3), stats_univ_orig_high(2,3), stats_univ_orig_high(3,3));
fprintf('%-30s X1: %.2e  X2: %.2e  X3: %.2e\n', ...
    'Multivariate Original:', stats_multi_orig_high(1,3), stats_multi_orig_high(2,3), stats_multi_orig_high(3,3));
fprintf('%-30s X1: %.2e  X2: %.2e  X3: %.2e\n', ...
    'Univariate Orthogonalized:', stats_univ_orth_high(1,3), stats_univ_orth_high(2,3), stats_univ_orth_high(3,3));
fprintf('%-30s X1: %.2e  X2: %.2e  X3: %.2e\n', ...
    'Multivariate Orthogonalized:', stats_multi_orth_high(1,3), stats_multi_orth_high(2,3), stats_multi_orth_high(3,3));

fprintf('\n--- OMNIBUS F-TEST (Model vs Null) ---\n');
fprintf('%-30s F = %8.2f, p = %.2e, R² = %.4f\n', ...
    'Multivariate Original:', fstat_multi_orig_high, fpval_multi_orig_high, rsq_multi_orig_high);
fprintf('%-30s F = %8.2f, p = %.2e, R² = %.4f\n', ...
    'Multivariate Orthogonalized:', fstat_multi_orth_high, fpval_multi_orth_high, rsq_multi_orth_high);

%% SCENARIO 2: MODERATE COLLINEARITY
fprintf('\n\n=== SCENARIO 2: MODERATE COLLINEARITY ===\n');

% Create predictors with MODERATE correlation
x1_mod = randn(n, 1);
x2_mod = 0.4*x1_mod + sqrt(1-0.4^2)*randn(n, 1);  % r ≈ 0.4
x3_mod = 0.3*x1_mod + 0.3*x2_mod + sqrt(1-0.3^2-0.3^2)*randn(n, 1);

X_mod = [x1_mod, x2_mod, x3_mod];
y_mod = X_mod * true_beta + 0.5*randn(n, 1);

% Apply Gram-Schmidt orthogonalization
X_mod_orth = zeros(size(X_mod));
X_mod_orth(:,1) = X_mod(:,1);

for k = 2:size(X_mod, 2)
    X_mod_orth(:,k) = X_mod(:,k);
    for j = 1:k-1
        proj = (X_mod(:,k)' * X_mod_orth(:,j)) / (X_mod_orth(:,j)' * X_mod_orth(:,j));
        X_mod_orth(:,k) = X_mod_orth(:,k) - proj * X_mod_orth(:,j);
    end
end

% Initialize storage for statistics
stats_univ_orig_mod = zeros(3, 3);
stats_univ_orth_mod = zeros(3, 3);

% ORIGINAL PREDICTORS - Univariate Models
for i = 1:3
    mdl = fitlm(X_mod(:,i), y_mod);
    stats_univ_orig_mod(i, 1) = mdl.Coefficients.Estimate(2);
    stats_univ_orig_mod(i, 2) = mdl.Coefficients.tStat(2);
    stats_univ_orig_mod(i, 3) = mdl.Coefficients.pValue(2);
end

% ORIGINAL PREDICTORS - Multivariate Model
mdl_multi_orig_mod = fitlm(X_mod, y_mod);
stats_multi_orig_mod = [mdl_multi_orig_mod.Coefficients.Estimate(2:4), ...
                        mdl_multi_orig_mod.Coefficients.tStat(2:4), ...
                        mdl_multi_orig_mod.Coefficients.pValue(2:4)];
fstat_multi_orig_mod = mdl_multi_orig_mod.ModelFitVsNullModel.Fstat;
fpval_multi_orig_mod = mdl_multi_orig_mod.ModelFitVsNullModel.Pvalue;
rsq_multi_orig_mod = mdl_multi_orig_mod.Rsquared.Ordinary;

% ORTHOGONALIZED PREDICTORS - Univariate Models
for i = 1:3
    mdl = fitlm(X_mod_orth(:,i), y_mod);
    stats_univ_orth_mod(i, 1) = mdl.Coefficients.Estimate(2);
    stats_univ_orth_mod(i, 2) = mdl.Coefficients.tStat(2);
    stats_univ_orth_mod(i, 3) = mdl.Coefficients.pValue(2);
end

% ORTHOGONALIZED PREDICTORS - Multivariate Model
mdl_multi_orth_mod = fitlm(X_mod_orth, y_mod);
stats_multi_orth_mod = [mdl_multi_orth_mod.Coefficients.Estimate(2:4), ...
                        mdl_multi_orth_mod.Coefficients.tStat(2:4), ...
                        mdl_multi_orth_mod.Coefficients.pValue(2:4)];
fstat_multi_orth_mod = mdl_multi_orth_mod.ModelFitVsNullModel.Fstat;
fpval_multi_orth_mod = mdl_multi_orth_mod.ModelFitVsNullModel.Pvalue;
rsq_multi_orth_mod = mdl_multi_orth_mod.Rsquared.Ordinary;

% Display results - MODERATE COLLINEARITY
fprintf('\nPredictor Correlations (X1-X2, X1-X3, X2-X3): [%.3f, %.3f, %.3f]\n', ...
    corr(X_mod(:,1), X_mod(:,2)), corr(X_mod(:,1), X_mod(:,3)), corr(X_mod(:,2), X_mod(:,3)));

fprintf('\n--- BETA COEFFICIENTS ---\n');
fprintf('%-30s X1: %7.3f  X2: %7.3f  X3: %7.3f\n', ...
    'Univariate Original:', stats_univ_orig_mod(1,1), stats_univ_orig_mod(2,1), stats_univ_orig_mod(3,1));
fprintf('%-30s X1: %7.3f  X2: %7.3f  X3: %7.3f\n', ...
    'Multivariate Original:', stats_multi_orig_mod(1,1), stats_multi_orig_mod(2,1), stats_multi_orig_mod(3,1));
fprintf('%-30s X1: %7.3f  X2: %7.3f  X3: %7.3f\n', ...
    'Univariate Orthogonalized:', stats_univ_orth_mod(1,1), stats_univ_orth_mod(2,1), stats_univ_orth_mod(3,1));
fprintf('%-30s X1: %7.3f  X2: %7.3f  X3: %7.3f\n', ...
    'Multivariate Orthogonalized:', stats_multi_orth_mod(1,1), stats_multi_orth_mod(2,1), stats_multi_orth_mod(3,1));

fprintf('\n--- T-STATISTICS (against zero) ---\n');
fprintf('%-30s X1: %7.2f  X2: %7.2f  X3: %7.2f\n', ...
    'Univariate Original:', stats_univ_orig_mod(1,2), stats_univ_orig_mod(2,2), stats_univ_orig_mod(3,2));
fprintf('%-30s X1: %7.2f  X2: %7.2f  X3: %7.2f\n', ...
    'Multivariate Original:', stats_multi_orig_mod(1,2), stats_multi_orig_mod(2,2), stats_multi_orig_mod(3,2));
fprintf('%-30s X1: %7.2f  X2: %7.2f  X3: %7.2f\n', ...
    'Univariate Orthogonalized:', stats_univ_orth_mod(1,2), stats_univ_orth_mod(2,2), stats_univ_orth_mod(3,2));
fprintf('%-30s X1: %7.2f  X2: %7.2f  X3: %7.2f\n', ...
    'Multivariate Orthogonalized:', stats_multi_orth_mod(1,2), stats_multi_orth_mod(2,2), stats_multi_orth_mod(3,2));

fprintf('\n--- P-VALUES ---\n');
fprintf('%-30s X1: %.2e  X2: %.2e  X3: %.2e\n', ...
    'Univariate Original:', stats_univ_orig_mod(1,3), stats_univ_orig_mod(2,3), stats_univ_orig_mod(3,3));
fprintf('%-30s X1: %.2e  X2: %.2e  X3: %.2e\n', ...
    'Multivariate Original:', stats_multi_orig_mod(1,3), stats_multi_orig_mod(2,3), stats_multi_orig_mod(3,3));
fprintf('%-30s X1: %.2e  X2: %.2e  X3: %.2e\n', ...
    'Univariate Orthogonalized:', stats_univ_orth_mod(1,3), stats_univ_orth_mod(2,3), stats_univ_orth_mod(3,3));
fprintf('%-30s X1: %.2e  X2: %.2e  X3: %.2e\n', ...
    'Multivariate Orthogonalized:', stats_multi_orth_mod(1,3), stats_multi_orth_mod(2,3), stats_multi_orth_mod(3,3));

fprintf('\n--- OMNIBUS F-TEST (Model vs Null) ---\n');
fprintf('%-30s F = %8.2f, p = %.2e, R² = %.4f\n', ...
    'Multivariate Original:', fstat_multi_orig_mod, fpval_multi_orig_mod, rsq_multi_orig_mod);
fprintf('%-30s F = %8.2f, p = %.2e, R² = %.4f\n', ...
    'Multivariate Orthogonalized:', fstat_multi_orth_mod, fpval_multi_orth_mod, rsq_multi_orth_mod);

%% VISUALIZATION 1: ORIGINAL DATA STRUCTURE
fig1 = figure('Position', [50, 50, 1400, 900]);
sgtitle('Original Data: Predictor Relationships (Before Orthogonalization)', 'FontSize', 16, 'FontWeight', 'bold');

% HIGH COLLINEARITY - Correlation Matrix (Original)
subplot(2,4,1);
corr_matrix_high = corr(X_high);
imagesc(corr_matrix_high);
colorbar;
caxis([-1 1]);
colormap(gca, jet);
title('HIGH COL - Predictor Correlations', 'FontSize', 11, 'FontWeight', 'bold');
xlabel('Predictor', 'FontSize', 9);
ylabel('Predictor', 'FontSize', 9);
set(gca, 'XTick', 1:3, 'YTick', 1:3, 'XTickLabel', {'X1','X2','X3'}, 'YTickLabel', {'X1','X2','X3'});
for i = 1:3
    for j = 1:3
        text(j, i, sprintf('%.2f', corr_matrix_high(i,j)), ...
            'HorizontalAlignment', 'center', 'FontSize', 10, ...
            'FontWeight', 'bold', 'Color', 'w');
    end
end

% HIGH COLLINEARITY - Scatter Plots (Original)
for i = 1:3
    subplot(2,4,i+1);
    scatter(X_high(:,i), y_high, 20, 'filled', 'MarkerFaceAlpha', 0.5);
    hold on;
    p = polyfit(X_high(:,i), y_high, 1);
    x_line = linspace(min(X_high(:,i)), max(X_high(:,i)), 100);
    y_line = polyval(p, x_line);
    plot(x_line, y_line, 'r-', 'LineWidth', 2.5);
    hold off;
    title(sprintf('X%d vs Y (r=%.2f)', i, corr(X_high(:,i), y_high)), 'FontSize', 10, 'FontWeight', 'bold');
    xlabel(sprintf('X%d', i), 'FontSize', 9);
    ylabel('Y', 'FontSize', 9);
    grid on;
end

% MODERATE COLLINEARITY - Correlation Matrix (Original)
subplot(2,4,5);
corr_matrix_mod = corr(X_mod);
imagesc(corr_matrix_mod);
colorbar;
caxis([-1 1]);
colormap(gca, jet);
title('MOD COL - Predictor Correlations', 'FontSize', 11, 'FontWeight', 'bold');
xlabel('Predictor', 'FontSize', 9);
ylabel('Predictor', 'FontSize', 9);
set(gca, 'XTick', 1:3, 'YTick', 1:3, 'XTickLabel', {'X1','X2','X3'}, 'YTickLabel', {'X1','X2','X3'});
for i = 1:3
    for j = 1:3
        text(j, i, sprintf('%.2f', corr_matrix_mod(i,j)), ...
            'HorizontalAlignment', 'center', 'FontSize', 10, ...
            'FontWeight', 'bold', 'Color', 'w');
    end
end

% MODERATE COLLINEARITY - Scatter Plots (Original)
for i = 1:3
    subplot(2,4,i+5);
    scatter(X_mod(:,i), y_mod, 20, 'filled', 'MarkerFaceAlpha', 0.5);
    hold on;
    p = polyfit(X_mod(:,i), y_mod, 1);
    x_line = linspace(min(X_mod(:,i)), max(X_mod(:,i)), 100);
    y_line = polyval(p, x_line);
    plot(x_line, y_line, 'r-', 'LineWidth', 2.5);
    hold off;
    title(sprintf('X%d vs Y (r=%.2f)', i, corr(X_mod(:,i), y_mod)), 'FontSize', 10, 'FontWeight', 'bold');
    xlabel(sprintf('X%d', i), 'FontSize', 9);
    ylabel('Y', 'FontSize', 9);
    grid on;
end

%% VISUALIZATION 2: ORTHOGONALIZED DATA STRUCTURE
fig2 = figure('Position', [100, 100, 1400, 900]);
sgtitle('Orthogonalized Data: Predictor Relationships (After Orthogonalization)', 'FontSize', 16, 'FontWeight', 'bold');

% HIGH COLLINEARITY - Correlation Matrix (Orthogonalized)
subplot(2,4,1);
corr_matrix_high_orth = corr(X_high_orth);
imagesc(corr_matrix_high_orth);
colorbar;
caxis([-1 1]);
colormap(gca, jet);
title('HIGH COL - Predictor Correlations', 'FontSize', 11, 'FontWeight', 'bold');
xlabel('Predictor', 'FontSize', 9);
ylabel('Predictor', 'FontSize', 9);
set(gca, 'XTick', 1:3, 'YTick', 1:3, 'XTickLabel', {'X1','X2','X3'}, 'YTickLabel', {'X1','X2','X3'});
for i = 1:3
    for j = 1:3
        text(j, i, sprintf('%.2f', corr_matrix_high_orth(i,j)), ...
            'HorizontalAlignment', 'center', 'FontSize', 10, ...
            'FontWeight', 'bold', 'Color', 'w');
    end
end

% HIGH COLLINEARITY - Scatter Plots (Orthogonalized)
for i = 1:3
    subplot(2,4,i+1);
    scatter(X_high_orth(:,i), y_high, 20, 'filled', 'MarkerFaceAlpha', 0.5, 'MarkerFaceColor', [0.2 0.6 0.8]);
    hold on;
    p = polyfit(X_high_orth(:,i), y_high, 1);
    x_line = linspace(min(X_high_orth(:,i)), max(X_high_orth(:,i)), 100);
    y_line = polyval(p, x_line);
    plot(x_line, y_line, 'g-', 'LineWidth', 2.5);
    hold off;
    title(sprintf('Orth X%d vs Y (r=%.2f)', i, corr(X_high_orth(:,i), y_high)), 'FontSize', 10, 'FontWeight', 'bold');
    xlabel(sprintf('Orth X%d', i), 'FontSize', 9);
    ylabel('Y', 'FontSize', 9);
    grid on;
end

% MODERATE COLLINEARITY - Correlation Matrix (Orthogonalized)
subplot(2,4,5);
corr_matrix_mod_orth = corr(X_mod_orth);
imagesc(corr_matrix_mod_orth);
colorbar;
caxis([-1 1]);
colormap(gca, jet);
title('MOD COL - Predictor Correlations', 'FontSize', 11, 'FontWeight', 'bold');
xlabel('Predictor', 'FontSize', 9);
ylabel('Predictor', 'FontSize', 9);
set(gca, 'XTick', 1:3, 'YTick', 1:3, 'XTickLabel', {'X1','X2','X3'}, 'YTickLabel', {'X1','X2','X3'});
for i = 1:3
    for j = 1:3
        text(j, i, sprintf('%.2f', corr_matrix_mod_orth(i,j)), ...
            'HorizontalAlignment', 'center', 'FontSize', 10, ...
            'FontWeight', 'bold', 'Color', 'w');
    end
end

% MODERATE COLLINEARITY - Scatter Plots (Orthogonalized)
for i = 1:3
    subplot(2,4,i+5);
    scatter(X_mod_orth(:,i), y_mod, 20, 'filled', 'MarkerFaceAlpha', 0.5, 'MarkerFaceColor', [0.2 0.6 0.8]);
    hold on;
    p = polyfit(X_mod_orth(:,i), y_mod, 1);
    x_line = linspace(min(X_mod_orth(:,i)), max(X_mod_orth(:,i)), 100);
    y_line = polyval(p, x_line);
    plot(x_line, y_line, 'g-', 'LineWidth', 2.5);
    hold off;
    title(sprintf('Orth X%d vs Y (r=%.2f)', i, corr(X_mod_orth(:,i), y_mod)), 'FontSize', 10, 'FontWeight', 'bold');
    xlabel(sprintf('Orth X%d', i), 'FontSize', 9);
    ylabel('Y', 'FontSize', 9);
    grid on;
end

%% VISUALIZATION 3: BETA COMPARISON
fig3 = figure('Position', [150, 150, 1400, 900]);
sgtitle('Beta Coefficient Comparison: Original vs Orthogonalized', 'FontSize', 16, 'FontWeight', 'bold');

x_pos = 1:3;
bar_width = 0.18;
bar_positions = [-1.5, -0.5, 0.5, 1.5] * bar_width;

% HIGH COLLINEARITY - Betas
subplot(2,2,1);
hold on;
for i = 1:3
    bar(i + bar_positions(1), stats_univ_orig_high(i,1), bar_width, 'FaceColor', [0.8 0.2 0.2]);
    bar(i + bar_positions(2), stats_multi_orig_high(i,1), bar_width, 'FaceColor', [0.2 0.2 0.8]);
    bar(i + bar_positions(3), stats_univ_orth_high(i,1), bar_width, 'FaceColor', [1 0.6 0.6]);
    bar(i + bar_positions(4), stats_multi_orth_high(i,1), bar_width, 'FaceColor', [0.6 0.6 1]);
end
hold off;
ylabel('Beta Coefficient', 'FontSize', 11, 'FontWeight', 'bold');
title('HIGH COLLINEARITY', 'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'XTick', x_pos, 'XTickLabel', {'X1', 'X2', 'X3'});
legend({'Univ Orig', 'Multi Orig', 'Univ Orth', 'Multi Orth'}, 'Location', 'best');
grid on;
yline(0, 'k--', 'LineWidth', 1);
xlim([0.5 3.5]);

% HIGH COLLINEARITY - Differences
subplot(2,2,2);
diff_orig_high = stats_univ_orig_high(:,1) - stats_multi_orig_high(:,1);
diff_orth_high = stats_univ_orth_high(:,1) - stats_multi_orth_high(:,1);
hold on;
for i = 1:3
    bar(i - 0.15, diff_orig_high(i), 0.25, 'FaceColor', [0.8 0.4 0.2]);
    bar(i + 0.15, diff_orth_high(i), 0.25, 'FaceColor', [0.2 0.8 0.4]);
end
hold off;
ylabel('Beta Difference (Univ - Multi)', 'FontSize', 11, 'FontWeight', 'bold');
title('HIGH COLLINEARITY - Differences', 'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'XTick', x_pos, 'XTickLabel', {'X1', 'X2', 'X3'});
legend({'Original', 'Orthogonalized'}, 'Location', 'best');
grid on;
yline(0, 'k--', 'LineWidth', 1.5);
xlim([0.5 3.5]);

% MODERATE COLLINEARITY - Betas
subplot(2,2,3);
hold on;
for i = 1:3
    bar(i + bar_positions(1), stats_univ_orig_mod(i,1), bar_width, 'FaceColor', [0.8 0.2 0.2]);
    bar(i + bar_positions(2), stats_multi_orig_mod(i,1), bar_width, 'FaceColor', [0.2 0.2 0.8]);
    bar(i + bar_positions(3), stats_univ_orth_mod(i,1), bar_width, 'FaceColor', [1 0.6 0.6]);
    bar(i + bar_positions(4), stats_multi_orth_mod(i,1), bar_width, 'FaceColor', [0.6 0.6 1]);
end
hold off;
ylabel('Beta Coefficient', 'FontSize', 11, 'FontWeight', 'bold');
title('MODERATE COLLINEARITY', 'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'XTick', x_pos, 'XTickLabel', {'X1', 'X2', 'X3'});
legend({'Univ Orig', 'Multi Orig', 'Univ Orth', 'Multi Orth'}, 'Location', 'best');
grid on;
yline(0, 'k--', 'LineWidth', 1);
xlim([0.5 3.5]);

% MODERATE COLLINEARITY - Differences
subplot(2,2,4);
diff_orig_mod = stats_univ_orig_mod(:,1) - stats_multi_orig_mod(:,1);
diff_orth_mod = stats_univ_orth_mod(:,1) - stats_multi_orth_mod(:,1);
hold on;
for i = 1:3
    bar(i - 0.15, diff_orig_mod(i), 0.25, 'FaceColor', [0.8 0.4 0.2]);
    bar(i + 0.15, diff_orth_mod(i), 0.25, 'FaceColor', [0.2 0.8 0.4]);
end
hold off;
ylabel('Beta Difference (Univ - Multi)', 'FontSize', 11, 'FontWeight', 'bold');
title('MODERATE COLLINEARITY - Differences', 'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'XTick', x_pos, 'XTickLabel', {'X1', 'X2', 'X3'});
legend({'Original', 'Orthogonalized'}, 'Location', 'best');
grid on;
yline(0, 'k--', 'LineWidth', 1.5);
xlim([0.5 3.5]);

%% VISUALIZATION 4: T-STATISTICS COMPARISON
fig4 = figure('Position', [200, 200, 1400, 900]);
sgtitle('T-Statistics Comparison: Original vs Orthogonalized', 'FontSize', 16, 'FontWeight', 'bold');

% HIGH COLLINEARITY - T-stats
subplot(2,2,1);
hold on;
for i = 1:3
    bar(i + bar_positions(1), stats_univ_orig_high(i,2), bar_width, 'FaceColor', [0.8 0.2 0.2]);
    bar(i + bar_positions(2), stats_multi_orig_high(i,2), bar_width, 'FaceColor', [0.2 0.2 0.8]);
    bar(i + bar_positions(3), stats_univ_orth_high(i,2), bar_width, 'FaceColor', [1 0.6 0.6]);
    bar(i + bar_positions(4), stats_multi_orth_high(i,2), bar_width, 'FaceColor', [0.6 0.6 1]);
end
yline(1.96, 'k--', 'LineWidth', 1.5, 'Label', 'p=0.05');
yline(-1.96, 'k--', 'LineWidth', 1.5);
hold off;
ylabel('T-Statistic', 'FontSize', 11, 'FontWeight', 'bold');
title('HIGH COLLINEARITY - T-Statistics', 'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'XTick', 1:3, 'XTickLabel', {'X1', 'X2', 'X3'});
legend({'Univ Orig', 'Multi Orig', 'Univ Orth', 'Multi Orth'}, 'Location', 'best');
grid on;
xlim([0.5 3.5]);

% HIGH COLLINEARITY - T-stat Differences
subplot(2,2,2);
diff_tstat_orig_high = stats_univ_orig_high(:,2) - stats_multi_orig_high(:,2);
diff_tstat_orth_high = stats_univ_orth_high(:,2) - stats_multi_orth_high(:,2);
hold on;
for i = 1:3
    bar(i - 0.15, diff_tstat_orig_high(i), 0.25, 'FaceColor', [0.8 0.4 0.2]);
    bar(i + 0.15, diff_tstat_orth_high(i), 0.25, 'FaceColor', [0.2 0.8 0.4]);
end
hold off;
ylabel('T-Stat Difference (Univ - Multi)', 'FontSize', 11, 'FontWeight', 'bold');
title('HIGH COLLINEARITY - T-Stat Differences', 'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'XTick', x_pos, 'XTickLabel', {'X1', 'X2', 'X3'});
legend({'Original', 'Orthogonalized'}, 'Location', 'best');
grid on;
yline(0, 'k--', 'LineWidth', 1.5);
xlim([0.5 3.5]);

% MODERATE COLLINEARITY - T-stats
subplot(2,2,3);
hold on;
for i = 1:3
    bar(i + bar_positions(1), stats_univ_orig_mod(i,2), bar_width, 'FaceColor', [0.8 0.2 0.2]);
    bar(i + bar_positions(2), stats_multi_orig_mod(i,2), bar_width, 'FaceColor', [0.2 0.2 0.8]);
    bar(i + bar_positions(3), stats_univ_orth_mod(i,2), bar_width, 'FaceColor', [1 0.6 0.6]);
    bar(i + bar_positions(4), stats_multi_orth_mod(i,2), bar_width, 'FaceColor', [0.6 0.6 1]);
end
yline(1.96, 'k--', 'LineWidth', 1.5, 'Label', 'p=0.05');
yline(-1.96, 'k--', 'LineWidth', 1.5);
hold off;
ylabel('T-Statistic', 'FontSize', 11, 'FontWeight', 'bold');
title('MODERATE COLLINEARITY - T-Statistics', 'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'XTick', 1:3, 'XTickLabel', {'X1', 'X2', 'X3'});
legend({'Univ Orig', 'Multi Orig', 'Univ Orth', 'Multi Orth'}, 'Location', 'best');
grid on;
xlim([0.5 3.5]);

% MODERATE COLLINEARITY - T-stat Differences
subplot(2,2,4);
diff_tstat_orig_mod = stats_univ_orig_mod(:,2) - stats_multi_orig_mod(:,2);
diff_tstat_orth_mod = stats_univ_orth_mod(:,2) - stats_multi_orth_mod(:,2);
hold on;
for i = 1:3
    bar(i - 0.15, diff_tstat_orig_mod(i), 0.25, 'FaceColor', [0.8 0.4 0.2]);
    bar(i + 0.15, diff_tstat_orth_mod(i), 0.25, 'FaceColor', [0.2 0.8 0.4]);
end
hold off;
ylabel('T-Stat Difference (Univ - Multi)', 'FontSize', 11, 'FontWeight', 'bold');
title('MODERATE COLLINEARITY - T-Stat Differences', 'FontSize', 12, 'FontWeight', 'bold');
set(gca, 'XTick', x_pos, 'XTickLabel', {'X1', 'X2', 'X3'});
legend({'Original', 'Orthogonalized'}, 'Location', 'best');
grid on;
yline(0, 'k--', 'LineWidth', 1.5);
xlim([0.5 3.5]);

%% VISUALIZATION 5: F-TEST COMPARISON
fig5 = figure('Position', [250, 250, 800, 400]);
sgtitle('Omnibus F-Test: Model Significance', 'FontSize', 16, 'FontWeight', 'bold');

% HIGH COLLINEARITY - F-test comparison
subplot(1,2,1);
f_data_high = [fstat_multi_orig_high, fstat_multi_orth_high];
bar(f_data_high);
ylabel('F-Statistic', 'FontSize', 11, 'FontWeight', 'bold');
title(sprintf('HIGH COLLINEARITY\nOrig: F=%.1f, Orth: F=%.1f', ...
    fstat_multi_orig_high, fstat_multi_orth_high), 'FontSize', 11, 'FontWeight', 'bold');
set(gca, 'XTick', 1:2, 'XTickLabel', {'Original', 'Orthogonalized'});
grid on;
text(1, fstat_multi_orig_high*0.9, sprintf('R²=%.3f\np=%.2e', rsq_multi_orig_high, fpval_multi_orig_high), ...
    'HorizontalAlignment', 'center', 'FontSize', 9);
text(2, fstat_multi_orth_high*0.9, sprintf('R²=%.3f\np=%.2e', rsq_multi_orth_high, fpval_multi_orth_high), ...
    'HorizontalAlignment', 'center', 'FontSize', 9);

% MODERATE COLLINEARITY - F-test comparison
subplot(1,2,2);
f_data_mod = [fstat_multi_orig_mod, fstat_multi_orth_mod];
bar(f_data_mod);
ylabel('F-Statistic', 'FontSize', 11, 'FontWeight', 'bold');
title(sprintf('MODERATE COLLINEARITY\nOrig: F=%.1f, Orth: F=%.1f', ...
    fstat_multi_orig_mod, fstat_multi_orth_mod), 'FontSize', 11, 'FontWeight', 'bold');
set(gca, 'XTick', 1:2, 'XTickLabel', {'Original', 'Orthogonalized'});
grid on;
text(1, fstat_multi_orig_mod*0.9, sprintf('R²=%.3f\np=%.2e', rsq_multi_orig_mod, fpval_multi_orig_mod), ...
    'HorizontalAlignment', 'center', 'FontSize', 9);
text(2, fstat_multi_orth_mod*0.9, sprintf('R²=%.3f\np=%.2e', rsq_multi_orth_mod, fpval_multi_orth_mod), ...
    'HorizontalAlignment', 'center', 'FontSize', 9);

%% SUMMARY
fprintf('\n\n=== KEY FINDINGS SUMMARY ===\n');
fprintf('Expected: Univariate Original > Multivariate Original (due to collinearity)\n');
fprintf('Key Question: Are Univariate Orthogonalized ≈ Multivariate Orthogonalized?\n');
fprintf('              If YES → Orthogonalization successfully removes collinearity effects\n\n');

fprintf('HIGH COLLINEARITY:\n');
fprintf('  Beta Differences (Univ - Multi):\n');
fprintf('    Original:       X1=%+.3f, X2=%+.3f, X3=%+.3f\n', diff_orig_high(1), diff_orig_high(2), diff_orig_high(3));
fprintf('    Orthogonalized: X1=%+.3f, X2=%+.3f, X3=%+.3f\n', diff_orth_high(1), diff_orth_high(2), diff_orth_high(3));
fprintf('  T-Stat Differences (Univ - Multi):\n');
fprintf('    Original:       X1=%+.2f, X2=%+.2f, X3=%+.2f\n', diff_tstat_orig_high(1), diff_tstat_orig_high(2), diff_tstat_orig_high(3));
fprintf('    Orthogonalized: X1=%+.2f, X2=%+.2f, X3=%+.2f\n', diff_tstat_orth_high(1), diff_tstat_orth_high(2), diff_tstat_orth_high(3));

fprintf('\nMODERATE COLLINEARITY:\n');
fprintf('  Beta Differences (Univ - Multi):\n');
fprintf('    Original:       X1=%+.3f, X2=%+.3f, X3=%+.3f\n', diff_orig_mod(1), diff_orig_mod(2), diff_orig_mod(3));
fprintf('    Orthogonalized: X1=%+.3f, X2=%+.3f, X3=%+.3f\n', diff_orth_mod(1), diff_orth_mod(2), diff_orth_mod(3));
fprintf('  T-Stat Differences (Univ - Multi):\n');
fprintf('    Original:       X1=%+.2f, X2=%+.2f, X3=%+.2f\n', diff_tstat_orig_mod(1), diff_tstat_orig_mod(2), diff_tstat_orig_mod(3));
fprintf('    Orthogonalized: X1=%+.2f, X2=%+.2f, X3=%+.2f\n', diff_tstat_orth_mod(1), diff_tstat_orth_mod(2), diff_tstat_orth_mod(3));

fprintf('\n=== INTERPRETATION ===\n');
fprintf('1. T-tests: Higher |t| = more significant effect of that predictor\n');
fprintf('   - Collinearity inflates SE, reducing t-stats in multivariate models\n');
fprintf('   - Orthogonalization should equalize Univ vs Multi t-stats\n');
fprintf('2. F-test: Tests if ALL predictors jointly explain significant variance\n');
fprintf('   - Should be similar for Original vs Orthogonalized (same total R²)\n');
fprintf('3. Key insight: If Orth closes the Univ-Multi gap, collinearity was the culprit\n');

%% SAVE ALL FIGURES
fprintf('\n=== SAVING FIGURES ===\n');

saveas(fig1, 'fig1_original_data_structure.svg');
saveas(fig2, 'fig2_orthogonalized_data_structure.svg');
saveas(fig3, 'fig3_beta_comparison.svg');
saveas(fig4, 'fig4_tstatistics_comparison.svg');
saveas(fig5, 'fig5_ftest_comparison.svg');

fprintf('Figures saved to: %s\n', pwd);
fprintf('  - fig1_original_data_structure.svg\n');
fprintf('  - fig2_orthogonalized_data_structure.svg\n');
fprintf('  - fig3_beta_comparison.svg\n');
fprintf('  - fig4_tstatistics_comparison.svg\n');
fprintf('  - fig5_ftest_comparison.svg\n');

close(fig1);
close(fig2);
close(fig3);
close(fig4);
close(fig5);

fprintf('All figures saved and closed.\n');
fprintf('Done!\n');