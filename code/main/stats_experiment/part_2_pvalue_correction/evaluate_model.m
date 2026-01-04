function results = evaluate_model(predictions, data, training_info, cfg)
    % EVALUATE MODEL PERFORMANCE
    % Calculate metrics and create visualizations
    % Returns struct with all performance metrics
    
    fprintf('\n=== EVALUATING MODEL PERFORMANCE ===\n');
    
    y_test = data.test.UnbiasedP;
    alpha = cfg.eval.alpha;
    
    %% Calculate metrics
    results = struct();
    results.predictions = predictions;
    results.y_true = y_test;
    results.training_info = training_info;
    
    % Overall metrics
    results.mae_all = mean(abs(predictions - y_test));
    results.r2_all = 1 - sum((y_test - predictions).^2) / sum((y_test - mean(y_test)).^2);
    
    % Critical region metrics (both predicted and true < alpha)
    critical_idx = (y_test <= alpha) & (predictions <= alpha);
    results.n_critical = sum(critical_idx);
    
    if results.n_critical > 0
        results.mae_critical = mean(abs(predictions(critical_idx) - y_test(critical_idx)));
        results.r2_critical = 1 - sum((y_test(critical_idx) - predictions(critical_idx)).^2) / ...
                              sum((y_test(critical_idx) - mean(y_test(critical_idx))).^2);
    else
        results.mae_critical = NaN;
        results.r2_critical = NaN;
    end
    
    % False positive/negative analysis
    results.fpr = sum(predictions <= alpha) / length(predictions);
    results.true_fpr = sum(y_test <= alpha) / length(y_test);
    
    results.seg_A = sum(predictions < alpha & y_test >= alpha);  % False positives
    results.seg_B = sum(predictions >= alpha & y_test < alpha);  % False negatives
    results.seg_C = sum(predictions < alpha & y_test < alpha);   % True positives
    results.seg_D = sum(predictions >= alpha & y_test >= alpha); % True negatives
    
    % Per-distribution metrics
    results.per_dist = calculatePerDistributionMetrics(predictions, data.test, alpha);
    
    %% Print summary
    printSummary(results, training_info, cfg);
    
    %% Create visualizations
    if cfg.eval.save_plots
        fprintf('\nCreating visualizations...\n');
        visualizeResults(results, data, training_info, cfg);
    end
end

%% ============================================================================
%% PER-DISTRIBUTION METRICS
%% ============================================================================

function per_dist = calculatePerDistributionMetrics(predictions, test_data, alpha)
    unique_dists = unique(test_data.Distribution);
    n_dists = length(unique_dists);
    
    per_dist = struct();
    per_dist.distributions = unique_dists;
    per_dist.mae = zeros(n_dists, 1);
    per_dist.r2 = zeros(n_dists, 1);
    per_dist.fpr = zeros(n_dists, 1);
    per_dist.n_samples = zeros(n_dists, 1);
    per_dist.n_critical = zeros(n_dists, 1);
    
    for i = 1:n_dists
        dist_name = unique_dists{i};
        dist_idx = strcmp(test_data.Distribution, dist_name);
        
        y_true_dist = test_data.UnbiasedP(dist_idx);
        y_pred_dist = predictions(dist_idx);
        
        per_dist.n_samples(i) = sum(dist_idx);
        per_dist.fpr(i) = sum(y_pred_dist <= alpha) / length(y_pred_dist);
        
        % Critical region metrics
        critical_idx = (y_true_dist <= alpha) & (y_pred_dist <= alpha);
        per_dist.n_critical(i) = sum(critical_idx);
        
        if sum(critical_idx) > 0
            per_dist.mae(i) = mean(abs(y_pred_dist(critical_idx) - y_true_dist(critical_idx)));
            per_dist.r2(i) = 1 - sum((y_true_dist(critical_idx) - y_pred_dist(critical_idx)).^2) / ...
                             sum((y_true_dist(critical_idx) - mean(y_true_dist(critical_idx))).^2);
        else
            per_dist.mae(i) = NaN;
            per_dist.r2(i) = NaN;
        end
    end
end

%% ============================================================================
%% PRINT SUMMARY
%% ============================================================================

function printSummary(results, training_info, cfg)
    fprintf('\n========== MODEL PERFORMANCE SUMMARY ==========\n');
    fprintf('Model Type: %s\n', training_info.model_type);
    
    if isfield(training_info, 'training_ranges')
        fprintf('Range-specific training: %d ranges\n', length(training_info.training_ranges));
    end
    
    fprintf('\nOVERALL METRICS (Full Test Set, n=%d):\n', length(results.predictions));
    fprintf('  MAE:  %.6f\n', results.mae_all);
    fprintf('  R²:   %.4f\n', results.r2_all);
    fprintf('  FPR:  %.4f (True FPR: %.4f)\n', results.fpr, results.true_fpr);
    
    fprintf('\nCRITICAL REGION METRICS (α=%.2f, n=%d):\n', cfg.eval.alpha, results.n_critical);
    if results.n_critical > 0
        fprintf('  MAE:  %.6f\n', results.mae_critical);
        fprintf('  R²:   %.4f\n', results.r2_critical);
    else
        fprintf('  No samples in critical region!\n');
    end
    
    fprintf('\nCONFUSION ANALYSIS:\n');
    fprintf('  Segment A (False Pos): %6d (%.2f%%)\n', results.seg_A, 100*results.seg_A/length(results.predictions));
    fprintf('  Segment B (False Neg): %6d (%.2f%%)\n', results.seg_B, 100*results.seg_B/length(results.predictions));
    fprintf('  Segment C (True Pos):  %6d (%.2f%%)\n', results.seg_C, 100*results.seg_C/length(results.predictions));
    fprintf('  Segment D (True Neg):  %6d (%.2f%%)\n', results.seg_D, 100*results.seg_D/length(results.predictions));
    
    fprintf('\nPER-DISTRIBUTION METRICS:\n');
    fprintf('%-15s %8s %10s %10s %10s\n', 'Distribution', 'N', 'MAE', 'R²', 'FPR');
    fprintf('%s\n', repmat('-', 1, 63));
    for i = 1:length(results.per_dist.distributions)
        fprintf('%-15s %8d %10.6f %10.4f %10.4f\n', ...
                results.per_dist.distributions{i}, ...
                results.per_dist.n_critical(i), ...
                results.per_dist.mae(i), ...
                results.per_dist.r2(i), ...
                results.per_dist.fpr(i));
    end
    fprintf('===============================================\n');
end

%% ============================================================================
%% VISUALIZATION
%% ============================================================================

function visualizeResults(results, data, training_info, cfg)
    % Sample data for plotting (5% for speed)
    n_test = length(results.predictions);
    sample_size = round(cfg.eval.plot_sample_fraction * n_test);
    sample_idx = randperm(n_test, sample_size);
    
    test_data_sample = data.test(sample_idx, :);
    predictions_sample = results.predictions(sample_idx);
    
    % Get colors for distributions
    unique_dists = data.distributions;
    n_dists = length(unique_dists);
    colors = lines(n_dists);
    
    % Create figure with tighter layout
    figure('Position', [50, 50, 2200, 1200]);
    
    %% Plot 1: QQ Plot - Critical Region (0-0.075)
    subplot(2, 3, [1, 2]);
    
    critical_idx = (test_data_sample.UnbiasedP <= 0.075) & (predictions_sample <= 0.075);
    y_true_critical = test_data_sample.UnbiasedP(critical_idx);
    y_pred_critical = predictions_sample(critical_idx);
    dist_critical = test_data_sample.Distribution(critical_idx);
    
    % Plot all distributions first
    for i = 1:n_dists
        dist_name = unique_dists{i};
        dist_idx = strcmp(dist_critical, dist_name);
        
        if sum(dist_idx) > 0
            scatter(y_pred_critical(dist_idx), y_true_critical(dist_idx), 8, colors(i,:), ...
                   'filled', 'MarkerFaceAlpha', 0.5, 'HandleVisibility', 'off');
            hold on;
        end
    end
    
    % Add legend entries with invisible points
    for i = 1:n_dists
        scatter(NaN, NaN, 50, colors(i,:), 'filled', 'DisplayName', unique_dists{i});
    end
    
    plot([0, 0.075], [0, 0.075], 'r--', 'LineWidth', 3, 'DisplayName', 'Perfect');
    xline(cfg.eval.alpha, 'r-', 'LineWidth', 2, 'DisplayName', sprintf('α=%.2f', cfg.eval.alpha));
    yline(cfg.eval.alpha, 'r-', 'LineWidth', 2, 'HandleVisibility', 'off');
    
    % Add segment annotations
    text(0.025, 0.025, sprintf('C: %d', results.seg_C), 'FontSize', 14, 'FontWeight', 'bold', ...
         'Color', 'white', 'HorizontalAlignment', 'center', 'BackgroundColor', [0 0 0 0.7]);
    text(0.025, 0.065, sprintf('A: %d', results.seg_A), 'FontSize', 14, 'FontWeight', 'bold', ...
         'Color', 'white', 'HorizontalAlignment', 'center', 'BackgroundColor', [0 0 0 0.7]);
    text(0.065, 0.025, sprintf('B: %d', results.seg_B), 'FontSize', 14, 'FontWeight', 'bold', ...
         'Color', 'white', 'HorizontalAlignment', 'center', 'BackgroundColor', [0 0 0 0.7]);
    text(0.065, 0.065, sprintf('D: %d', results.seg_D), 'FontSize', 14, 'FontWeight', 'bold', ...
         'Color', 'white', 'HorizontalAlignment', 'center', 'BackgroundColor', [0 0 0 0.7]);
    
    xlabel('Predicted Unbiased P-Value', 'FontSize', 16, 'FontWeight', 'bold');
    ylabel('True Unbiased P-Value', 'FontSize', 16, 'FontWeight', 'bold');
    title('QQ Plot: Critical Region (0-0.075)', 'FontSize', 18, 'FontWeight', 'bold');
    xlim([0, 0.075]); ylim([0, 0.075]);
    grid on; axis square;
    set(gca, 'FontSize', 14);
    legend('Location', 'southeast', 'FontSize', 14);
    
    %% Plot 2: QQ Plot - Full Range
    subplot(2, 3, 3);
    
    % Plot all distributions first
    for i = 1:n_dists
        dist_name = unique_dists{i};
        dist_idx = strcmp(test_data_sample.Distribution, dist_name);
        
        if sum(dist_idx) > 0
            scatter(predictions_sample(dist_idx), test_data_sample.UnbiasedP(dist_idx), ...
                   8, colors(i,:), 'filled', 'MarkerFaceAlpha', 0.5, 'HandleVisibility', 'off');
            hold on;
        end
    end
    
    % Add legend entries with invisible points
    for i = 1:n_dists
        scatter(NaN, NaN, 50, colors(i,:), 'filled', 'DisplayName', unique_dists{i});
    end
    
    plot([0, 1], [0, 1], 'r--', 'LineWidth', 3, 'DisplayName', 'Perfect');
    xlabel('Predicted Unbiased P-Value', 'FontSize', 16, 'FontWeight', 'bold');
    ylabel('True Unbiased P-Value', 'FontSize', 16, 'FontWeight', 'bold');
    title('QQ Plot: Full Range', 'FontSize', 18, 'FontWeight', 'bold');
    xlim([0, 1]); ylim([0, 1]);
    grid on; axis square;
    set(gca, 'FontSize', 14);
    legend('Location', 'southeast', 'FontSize', 14);
    
    %% Plot 3: Residuals
    subplot(2, 3, 4);
    
    residuals = predictions_sample - test_data_sample.UnbiasedP;
    
    % Plot all distributions first
    for i = 1:n_dists
        dist_name = unique_dists{i};
        dist_idx = strcmp(test_data_sample.Distribution, dist_name);
        
        if sum(dist_idx) > 0
            scatter(residuals(dist_idx), test_data_sample.UnbiasedP(dist_idx), ...
                   50, colors(i,:), 'filled', 'MarkerFaceAlpha', 0.5, 'HandleVisibility', 'off');
            hold on;
        end
    end
    
    % Add legend entries with invisible points
    for i = 1:n_dists
        scatter(NaN, NaN, 50, colors(i,:), 'filled', 'DisplayName', unique_dists{i});
    end
    
    xline(0, 'r--', 'LineWidth', 2);
    xlabel('Residuals (Pred - True)', 'FontSize', 16, 'FontWeight', 'bold');
    ylabel('True Unbiased P-Value', 'FontSize', 16, 'FontWeight', 'bold');
    title('Residual Plot', 'FontSize', 18, 'FontWeight', 'bold');
    grid on;
    set(gca, 'FontSize', 14);
    legend('Location', 'best', 'FontSize', 14);
    
    %% Plot 4: Distribution Characteristics
    subplot(2, 3, 5);
    
    % Plot all distributions first
    for i = 1:n_dists
        dist_name = unique_dists{i};
        dist_idx = strcmp(test_data_sample.Distribution, dist_name);
        
        if sum(dist_idx) > 0
            dist_data = test_data_sample(dist_idx, :);
            valid_idx = ~isnan(dist_data.Skewness) & ~isnan(dist_data.Kurtosis) & ...
                       ~isinf(dist_data.Skewness) & ~isinf(dist_data.Kurtosis) & ...
                       abs(dist_data.Skewness) < 10 & abs(dist_data.Kurtosis) < 50;
            
            if sum(valid_idx) > 0
                scatter(dist_data.Skewness(valid_idx), dist_data.Kurtosis(valid_idx), ...
                       8, colors(i,:), 'filled', 'MarkerFaceAlpha', 0.5, 'HandleVisibility', 'off');
                hold on;
            end
        end
    end
    
    % Add legend entries with invisible points
    for i = 1:n_dists
        scatter(NaN, NaN, 50, colors(i,:), 'filled', 'DisplayName', unique_dists{i});
    end
    
    xlabel('Skewness', 'FontSize', 16, 'FontWeight', 'bold');
    ylabel('Kurtosis', 'FontSize', 16, 'FontWeight', 'bold');
    title('Distribution Characteristics', 'FontSize', 18, 'FontWeight', 'bold');
    grid on;
    set(gca, 'FontSize', 14);
    legend('Location', 'best', 'FontSize', 14);
    
    %% Plot 5: FPR by Distribution
    subplot(2, 3, 6);
    hold on;
    
    bar_data = results.per_dist.fpr;
    b = bar(bar_data);
    
    % Set color for FPR bars
    b.FaceColor = [0.9 0.3 0.3];  % Red for FPR
    
    % Fix x-axis labels
    set(gca, 'XTick', 1:n_dists, 'XTickLabel', unique_dists, 'FontSize', 14);
    xtickangle(45);
    ylabel('FPR', 'FontSize', 16, 'FontWeight', 'bold');
    title('FPR by Distribution', 'FontSize', 18, 'FontWeight', 'bold');
    grid on;
    
    %% Main title
    % Create friendly model name for display
    switch training_info.model_type
        case 'polynomial'
            display_name = 'Polynomial';
        case 'random_forest'
            display_name = 'Random Forest';
        case 'svr'
            display_name = 'SVR';
        case 'gbm'
            display_name = 'Gradient Boosting';
        otherwise
            display_name = training_info.model_type;
    end
    
    title_str = sprintf('%s Performance: MAE=%.6f, R²=%.4f, FPR=%.4f', ...
                       display_name, results.mae_critical, results.r2_critical, results.fpr);
    sgtitle(title_str, 'FontSize', 20, 'FontWeight', 'bold');
    
    %% Save figure
    save_dir = 'C:\Users\CDoga\Documents\Research\EEG-Neuroscience-PhD\code\main\stats_experiment\part_2_pvalue_correction\data_viz\plots';
    if ~exist(save_dir, 'dir')
        mkdir(save_dir);
    end
    
    timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
    filename = fullfile(save_dir, sprintf('model_performance_%s_%s.png', training_info.model_type, timestamp));
    saveas(gcf, filename);
    fprintf('Saved plot: %s\n', filename);
    close;
end