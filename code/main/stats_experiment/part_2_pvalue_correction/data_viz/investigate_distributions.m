%% QQ Plot Analysis Script with Distribution Comparison
% Analyzes p-value distributions using existing UnbiasedP values
% Creates QQ plots comparing RawP vs UnbiasedP distributions
% Includes histograms of raw p-values for each distribution
% NEW: Adds option to filter data to only critical region (p ≤ 0.05) before analysis

clear; close all; clc;

data_filename = 'C:\Users\CDoga\Documents\Research\EEG-Neuroscience-PhD\code\main\stats_experiment\part_2_pvalue_correction\sign_swap_40k_perms_n40_2026-01-03_14-54-10.parquet';  % Update with your filename
focus_critical_region = true;  % Focus on p ≤ 0.05 region in QQ plots
p_threshold = 0.05;
n_bins_comparison = 50;  % Number of bins comparison histograms
alpha_value = 0.6;  % Transparency level for overlapping histograms

% NEW OPTION: Filter entire dataset to only critical region before analysis
FILTER_DATA_TO_CRITICAL = false;  % Set to true to analyze only samples with BiasedP ≤ 0.05

% STANDARDIZATION: Font sizes for publication
TITLE_FONT_SIZE = 14;
LABEL_FONT_SIZE = 14;
LEGEND_FONT_SIZE = 12;
SGTITLE_FONT_SIZE = 16;

%% Create plots directory if it doesn't exist
if ~exist('plots', 'dir')
    mkdir('plots');
end

%% Main Script
% Load Data
fprintf('Loading data from: %s\n', data_filename);
try
    data = loadParquetData(data_filename);
    fprintf('Data loaded successfully! Found %d samples\n', height(data));
    
    % Verify UnbiasedP column exists
    if ~ismember('UnbiasedP', data.Properties.VariableNames)
        error('UnbiasedP column not found in data!');
    end
    fprintf('UnbiasedP values found in data\n');
catch ME
    fprintf('Error loading data: %s\n', ME.message);
    fprintf('Please update data_filename variable with correct path\n');
    return;
end

% NEW: Filter data to critical region if requested
if FILTER_DATA_TO_CRITICAL
    fprintf('\n*** FILTERING DATA TO CRITICAL REGION (BiasedP ≤ %.3f) ***\n', p_threshold);
    original_size = height(data);
    data = data(data.BiasedP <= p_threshold, :);
    filtered_size = height(data);
    fprintf('Data filtered: %d samples → %d samples (%.1f%% retained)\n', ...
        original_size, filtered_size, 100*filtered_size/original_size);
    
    if filtered_size == 0
        error('No data remaining after filtering to critical region!');
    end
end

% Analyze data structure
analyzeDataStructure(data);

createSkewnessVsPValueDensity(data, FILTER_DATA_TO_CRITICAL);

createBiasedUnbiasedComparison(data, FILTER_DATA_TO_CRITICAL);

% Create original visualizations
createFullRangeHistograms(data, FILTER_DATA_TO_CRITICAL);
createCriticalRegionHistograms(data, p_threshold, FILTER_DATA_TO_CRITICAL);
createQQPlots(data, focus_critical_region, p_threshold, FILTER_DATA_TO_CRITICAL);

% NEW: Create distribution comparison histograms
createDistributionComparison(data, n_bins_comparison, alpha_value, FILTER_DATA_TO_CRITICAL);

% NEW: Create individual variance plots for symmetric distributions
createSymmetricVariancePlots(data, n_bins_comparison, FILTER_DATA_TO_CRITICAL);

% NEW: Create skewness vs biased p-value scatter plot
filter_critical = true;  % Set to true to show only p < 0.05, false for all data
createSkewnessVsPValueScatter(data, filter_critical, FILTER_DATA_TO_CRITICAL);

clear;
clc;
close;
% Print summary statistics
%printSummaryStatistics(data, focus_critical_region, p_threshold, FILTER_DATA_TO_CRITICAL);

%% Methods

function analyzeDataStructure(data)
    % Analyze and display the structure of the loaded data
    unique_distributions = unique(data.Distribution);
    n_distributions = length(unique_distributions);
    
    fprintf('\nFound %d distributions:\n', n_distributions);
    for i = 1:n_distributions
        dist_name = unique_distributions{i};
        n_samples = sum(strcmp(data.Distribution, dist_name));
        fprintf('  %s: %d samples\n', dist_name, n_samples);
    end
end

function createFullRangeHistograms(data, is_filtered_data)
    % Create histograms showing full range (0-1) of p-values
    fprintf('\nGenerating full range histograms...\n');
    
    unique_distributions = unique(data.Distribution);
    n_distributions = length(unique_distributions);
    grid_size = ceil(sqrt(n_distributions));
    
    % Create histogram figure with extra space for titles
    figure('Position', [50, 50, 350*grid_size, 350*grid_size]);
    
    % Define histogram bins with 0.5% (0.005) width
    % If data is pre-filtered, adjust bins to critical region
    if is_filtered_data
        bin_width = 0.001;  % Finer bins for filtered data
        bin_edges = 0:bin_width:0.05;
    else
        bin_width = 0.005;
        bin_edges = 0:bin_width:1;
    end
    bin_centers = bin_edges(1:end-1) + bin_width/2;
    
    for i = 1:n_distributions
        subplot(grid_size, grid_size, i);
        
        dist_name = unique_distributions{i};
        
        % Extract BiasedP values for this distribution
        dist_idx = strcmp(data.Distribution, dist_name);
        biased_p_values = data.BiasedP(dist_idx);
        
        % Create histogram
        hist_counts = histcounts(biased_p_values, bin_edges);
        
        % Plot histogram
        bar(bin_centers, hist_counts, 'BarWidth', 0.8, 'FaceColor', [0.3 0.6 0.9], ...
            'EdgeColor', [0.2 0.4 0.7], 'LineWidth', 0.5, 'FaceAlpha', 0.7);
        
        % Formatting
        if is_filtered_data
            xlim([0 0.05]);
            xticks(0:0.01:0.05);
        else
            xlim([0 1]);
            xticks(0:0.1:1);
        end
        xlabel('P-value', 'FontSize', 14);
        
        % Calculate FPR and add to title
        if ~is_filtered_data
            critical_count = sum(biased_p_values <= 0.05);
            fpr = critical_count / length(biased_p_values);
            title(sprintf('%s (FPR = %.3f)', dist_name, fpr), ...
                  'FontSize', 12, 'FontWeight', 'bold');
        else
            title(dist_name, 'FontSize', 12, 'FontWeight', 'bold');
        end
        grid on;
        
        % Add expected uniform line
        expected_count = length(biased_p_values) * bin_width;
        hold on;
        if is_filtered_data
            plot([0 0.05], [expected_count expected_count], 'r--', 'LineWidth', 2);
        else
            plot([0 1], [expected_count expected_count], 'r--', 'LineWidth', 2);
        end
        hold off;
        
        % Handle y-axis scaling
        max_count = max(hist_counts);
        min_nonzero_count = min(hist_counts(hist_counts > 0));
        
        if max_count / min_nonzero_count > 1000
            set(gca, 'YScale', 'log');
            ylim([min_nonzero_count/2, max_count*2]);
            ylabel('Count (log scale)', 'FontSize', 14);
        else
            ylim([0 max(ylim())]);
            ylabel('Count', 'FontSize', 14);
        end
    end
    
    if is_filtered_data
        sgtitle('Raw P-Value Distributions (FILTERED DATA: p ≤ 0.05)', ...
            'FontSize', 16, 'FontWeight', 'bold');
        saveas(gcf, 'plots/full_range_histograms_filtered.png');
    else
        sgtitle('Raw P-Value Distributions (0.5% bins)', 'FontSize', 16, 'FontWeight', 'bold');
        saveas(gcf, 'plots/full_range_histograms.png');
    end
    close(gcf);
end

function createCriticalRegionHistograms(data, p_threshold, is_filtered_data)
    % Create histograms focusing on critical region (0-0.05)
    % Skip if data is already filtered to critical region
    if is_filtered_data
        fprintf('Skipping separate critical region histograms (data already filtered)\n');
        return;
    end
    
    fprintf('\nGenerating critical region histograms...\n');
    
    unique_distributions = unique(data.Distribution);
    n_distributions = length(unique_distributions);
    grid_size = ceil(sqrt(n_distributions));
    
    % Create figure with extra vertical space for titles
    figure('Position', [150, 150, 350*grid_size, 380*grid_size]);
    
    % Define bins for critical region
    n_bins_critical = 50;
    bin_edges_critical = linspace(0, p_threshold, n_bins_critical + 1);
    bin_centers_critical = bin_edges_critical(1:end-1) + diff(bin_edges_critical(1:2))/2;
    bin_width_critical = bin_edges_critical(2) - bin_edges_critical(1);
    
    for i = 1:n_distributions
        subplot(grid_size, grid_size, i);
        
        dist_name = unique_distributions{i};
        
        % Extract BiasedP values for this distribution
        dist_idx = strcmp(data.Distribution, dist_name);
        biased_p_values = data.BiasedP(dist_idx);
        
        % Filter to critical region
        critical_p_values = biased_p_values(biased_p_values <= p_threshold);
        
        % Create histogram
        hist_counts_critical = histcounts(critical_p_values, bin_edges_critical);
        
        % Calculate FPR
        fpr_critical = length(critical_p_values) / length(biased_p_values);
        
        % Plot histogram
        bar(bin_centers_critical, hist_counts_critical, 'BarWidth', 0.8, ...
            'FaceColor', [0.9 0.3 0.3], 'EdgeColor', [0.7 0.2 0.2], ...
            'LineWidth', 0.5, 'FaceAlpha', 0.7);
        
        % Formatting
        xlim([0 p_threshold]);
        xlabel('P-value', 'FontSize', 14);
        ylabel('Count', 'FontSize', 14);
        title(sprintf('%s (FPR=%.3f)', dist_name, fpr_critical), ...
              'FontSize', 12, 'FontWeight', 'bold');
        grid on;
        xticks(0:0.01:p_threshold);
        
        % Add expected uniform line
        expected_count_critical = length(biased_p_values) * bin_width_critical;
        hold on;
        plot([0 p_threshold], [expected_count_critical expected_count_critical], 'r--', 'LineWidth', 2);
        hold off;
        
        % Set y-axis
        ylim([0 max(ylim())]);
    end
    
    sgtitle('Critical Region P-Value Distributions (0-0.05, 50 bins)', ...
            'FontSize', 16, 'FontWeight', 'bold');
    saveas(gcf, 'plots/critical_region_histograms.png');
    close(gcf);
end

function createDistributionComparison(data, n_bins, alpha_value, is_filtered_data)
    % Creates overlapping histograms comparing symmetric and non-symmetric distributions
    fprintf('\nGenerating distribution comparison histograms...\n');
    
    % Define which distributions are symmetric vs non-symmetric
    symmetric_dists = {'NORMAL', 'UNIFORM', 'LAPLACE'};
    nonsymmetric_dists = {'EXPONENTIAL', 'GAMMA', 'LOGNORMAL', 'CHI2', 'WEIBULL'};
    
    % Create logical indices for symmetric vs non-symmetric
    is_symmetric = false(height(data), 1);
    is_nonsymmetric = false(height(data), 1);
    
    for i = 1:length(symmetric_dists)
        is_symmetric = is_symmetric | contains(data.Distribution, symmetric_dists{i});
    end
    
    for i = 1:length(nonsymmetric_dists)
        is_nonsymmetric = is_nonsymmetric | contains(data.Distribution, nonsymmetric_dists{i});
    end
    
    % Extract data for each group
    symmetric_data = data(is_symmetric, :);
    nonsymmetric_data = data(is_nonsymmetric, :);
    
    fprintf('  Symmetric distributions: %d samples\n', height(symmetric_data));
    fprintf('  Non-symmetric distributions: %d samples\n', height(nonsymmetric_data));
    
    % Create comparison figure
    figure('Position', [200, 100, 1400, 800]);
    
    % Define variables to plot (check which are available)
    all_variables = {'BiasedP', 'Mean', 'Variance', 'Skewness', 'Kurtosis'};
    var_titles = {'Raw P-values', 'Mean', 'Variance', 'Skewness', 'Kurtosis'};
    
    % Filter to available variables
    available_vars = {};
    available_titles = {};
    for v = 1:length(all_variables)
        if ismember(all_variables{v}, data.Properties.VariableNames)
            available_vars{end+1} = all_variables{v};
            available_titles{end+1} = var_titles{v};
        end
    end
    
    if isempty(available_vars)
        fprintf('Warning: No feature variables found for comparison\n');
        return;
    end
    
    % Determine subplot layout
    n_plots = length(available_vars);
    n_cols = min(3, n_plots);
    n_rows = ceil(n_plots / n_cols);
    
    for v = 1:length(available_vars)
        subplot(n_rows, n_cols, v);
        
        var_name = available_vars{v};
        
        % Get data for current variable
        sym_values = symmetric_data.(var_name);
        nonsym_values = nonsymmetric_data.(var_name);
        
        % Remove NaN values if any
        sym_values = sym_values(~isnan(sym_values));
        nonsym_values = nonsym_values(~isnan(nonsym_values));
        
        if isempty(sym_values) || isempty(nonsym_values)
            fprintf('Warning: Empty data for %s\n', var_name);
            continue;
        end
        
        % Determine histogram bins
        all_values = [sym_values; nonsym_values];
        min_val = min(all_values);
        max_val = max(all_values);
        
        % Special handling for p-values
        if strcmp(var_name, 'BiasedP')
            if is_filtered_data
                bin_edges = linspace(0, 0.05, n_bins + 1);
            else
                bin_edges = linspace(0, 1, n_bins + 1);
            end
        else
            % Add small padding for other variables
            range_val = max_val - min_val;
            if range_val > 0
                padding = range_val * 0.05;
                bin_edges = linspace(min_val - padding, max_val + padding, n_bins + 1);
            else
                % Handle case where all values are the same
                bin_edges = linspace(min_val - 0.1, max_val + 0.1, n_bins + 1);
            end
        end
        
        % Plot overlapping histograms
        hold on;
        
        % Plot symmetric distribution (blue)
        h1 = histogram(sym_values, bin_edges, 'FaceColor', [0.2 0.4 0.8], ...
            'EdgeColor', 'none', 'FaceAlpha', alpha_value, 'Normalization', 'probability');
        
        % Plot non-symmetric distribution (red)
        h2 = histogram(nonsym_values, bin_edges, 'FaceColor', [0.8 0.2 0.2], ...
            'EdgeColor', 'none', 'FaceAlpha', alpha_value, 'Normalization', 'probability');
        
        % Formatting
        xlabel(available_titles{v}, 'FontSize', 14, 'FontWeight', 'bold');
        ylabel('Probability', 'FontSize', 14);
        grid on;
        set(gca, 'GridAlpha', 0.3);
        
        % Create patch objects for legend (workaround for transparency issue)
        h1_patch = patch(NaN, NaN, [0.2 0.4 0.8], 'FaceAlpha', alpha_value, 'EdgeColor', 'none');
        h2_patch = patch(NaN, NaN, [0.8 0.2 0.2], 'FaceAlpha', alpha_value, 'EdgeColor', 'none');
        
        % Add legend with patch handles
        legend([h1_patch, h2_patch], {'Symmetric', 'Non-symmetric'}, 'Location', 'best', 'FontSize', 12);
        
        % Add statistics to title
        sym_mean = mean(sym_values);
        nonsym_mean = mean(nonsym_values);
        title(sprintf('%s (Sym μ = %.3f, Non-sym μ = %.3f)', ...
            available_titles{v}, sym_mean, nonsym_mean), ...
            'FontSize', 12, 'FontWeight', 'bold');
        
        hold off;
    end
    
    % Overall title
    if is_filtered_data
        sgtitle('Distribution Comparison: Symmetric vs Non-Symmetric (FILTERED DATA: p ≤ 0.05)', ...
            'FontSize', 16, 'FontWeight', 'bold');
        saveas(gcf, 'plots/distribution_comparison_filtered.png');
    else
        sgtitle('Distribution Comparison: Symmetric vs Non-Symmetric', ...
            'FontSize', 16, 'FontWeight', 'bold');
        saveas(gcf, 'plots/distribution_comparison.png');
    end
    close(gcf);
    
    % Print comparison statistics
    fprintf('\n--- Distribution Comparison Statistics ---\n');
    if is_filtered_data
        fprintf('(Using FILTERED data: BiasedP ≤ 0.05)\n');
    end
    
    for v = 1:length(available_vars)
        var_name = available_vars{v};
        sym_values = symmetric_data.(var_name);
        nonsym_values = nonsymmetric_data.(var_name);
        
        % Remove NaN values
        sym_values = sym_values(~isnan(sym_values));
        nonsym_values = nonsym_values(~isnan(nonsym_values));
        
        fprintf('\n%s:\n', available_titles{v});
        fprintf('  Symmetric:     mean=%.4f, std=%.4f, median=%.4f\n', ...
            mean(sym_values), std(sym_values), median(sym_values));
        fprintf('  Non-symmetric: mean=%.4f, std=%.4f, median=%.4f\n', ...
            mean(nonsym_values), std(nonsym_values), median(nonsym_values));
        
        % Perform two-sample KS test if BiasedP
        if strcmp(var_name, 'BiasedP')
            [~, p_ks] = kstest2(sym_values, nonsym_values);
            fprintf('  KS test p-value: %.4e\n', p_ks);
        end
    end
end

function fpr_values = createQQPlots(data, focus_critical_region, p_threshold, is_filtered_data)
    % Create QQ plots comparing observed p-values against uniform distribution
    fprintf('\nGenerating QQ plots for p-value calibration...\n');
    unique_distributions = unique(data.Distribution);
    n_distributions = length(unique_distributions);
    
    % Set up figure
    figure('Position', [100, 100, 800, 700]);
    colors = lines(n_distributions);
    hold on;
    
    % Store data for legend and analysis
    legend_entries = {};
    plot_handles = [];
    fpr_values = zeros(n_distributions, 1);
    
    for i = 1:n_distributions
        dist_name = unique_distributions{i};
        
        % Extract BiasedP values for this distribution
        dist_idx = strcmp(data.Distribution, dist_name);
        biased_p_values = data.BiasedP(dist_idx);
        
        % If data is pre-filtered, use all values
        if is_filtered_data
            observed_p = sort(biased_p_values);
            n = length(observed_p);
            % Generate expected uniform quantiles for filtered range
            expected_p = linspace(0, p_threshold, n)';
        else
            % Focus on critical region if requested
            if focus_critical_region
                % Filter to critical region
                critical_idx = biased_p_values <= p_threshold;
                p_values_to_plot = biased_p_values(critical_idx);
                
                if isempty(p_values_to_plot)
                    fprintf('Warning: No data in critical region for %s\n', dist_name);
                    continue;
                end
                
                % Sort the observed p-values
                observed_p = sort(p_values_to_plot);
                n = length(observed_p);
                
                % Generate expected uniform quantiles for critical region
                expected_p = linspace(0, p_threshold, n)';
            else
                % Use all p-values
                observed_p = sort(biased_p_values);
                n = length(observed_p);
                
                % Generate expected uniform quantiles
                expected_p = ((1:n)' - 0.5) / n;
            end
            
            % Calculate False Positive Rate
            fpr = sum(biased_p_values <= p_threshold) / length(biased_p_values);
            fpr_values(i) = fpr;
        end
        
        % Plot QQ line with SWAPPED AXES (x=observed/raw, y=expected/unbiased)
        h = plot(observed_p, expected_p, 'Color', colors(i,:), 'LineWidth', 2.5);
        plot_handles(end+1) = h;
        
        % Create legend entry
        if is_filtered_data
            legend_entries{end+1} = sprintf('%s (n=%d)', dist_name, length(biased_p_values));
        else
            legend_entries{end+1} = sprintf('%s (FPR=%.3f)', dist_name, fpr);
        end
    end
    
    % Add perfect calibration line
    if is_filtered_data || focus_critical_region
        h_perfect = plot([0, p_threshold], [0, p_threshold], 'k--', 'LineWidth', 2.5);
        xlim([0, p_threshold]);
        ylim([0, p_threshold]);
        if is_filtered_data
            title('QQ Plot: Unbiased vs Raw P-values (FILTERED DATA: p ≤ 0.05)', ...
                'FontSize', 14, 'FontWeight', 'bold');
        else
            title('QQ Plot: Unbiased vs Raw P-values - Critical Region (p ≤ 0.05)', ...
                'FontSize', 14, 'FontWeight', 'bold');
        end
    else
        h_perfect = plot([0, 1], [0, 1], 'k--', 'LineWidth', 2.5);
        xlim([0, 1]);
        ylim([0, 1]);
        title('QQ Plot: Unbiased vs Raw P-values', ...
            'FontSize', 14, 'FontWeight', 'bold');
    end
    
    % Add perfect calibration handle and legend entry at the end
    plot_handles(end+1) = h_perfect;
    legend_entries{end+1} = 'Perfect Calibration';
    
    % Create legend with proper handle-label pairing
    legend(plot_handles, legend_entries, 'Location', 'southeast', 'FontSize', 12);
    
    % Update axis labels with swapped names
    xlabel('Raw (Observed) P-values', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('Unbiased (Desired) P-values', 'FontSize', 14, 'FontWeight', 'bold');
    
    grid on;
    set(gca, 'GridAlpha', 0.3);
    axis square;
    hold off;
    
    % Save figure with proper renderer AFTER all labels are set
    set(gcf, 'Renderer', 'painters');
    if is_filtered_data
        saveas(gcf, 'plots/qq_plots_filtered.png');
    elseif focus_critical_region
        saveas(gcf, 'plots/qq_plots_critical.png');
    else
        saveas(gcf, 'plots/qq_plots_full.png');
    end
    close(gcf);
end

function printSummaryStatistics(data, focus_critical_region, p_threshold, is_filtered_data)
    % Print summary statistics and analysis results
    unique_distributions = unique(data.Distribution);
    n_distributions = length(unique_distributions);
    
    fprintf('\n========== QQ PLOT ANALYSIS SUMMARY ==========\n');
    if is_filtered_data
        fprintf('*** ANALYZING FILTERED DATA (BiasedP ≤ %.3f) ***\n', p_threshold);
    end
    fprintf('Using existing UnbiasedP values from data\n');
    if focus_critical_region && ~is_filtered_data
        fprintf('Analysis focused on critical region (p ≤ %.3f)\n', p_threshold);
    elseif ~is_filtered_data
        fprintf('Analysis on full p-value range (0-1)\n');
    end
    
    if ~is_filtered_data
        fprintf('\nFalse Positive Rates by Distribution:\n');
        for i = 1:n_distributions
            dist_name = unique_distributions{i};
            dist_idx = strcmp(data.Distribution, dist_name);
            biased_p_values = data.BiasedP(dist_idx);
            fpr = sum(biased_p_values <= p_threshold) / length(biased_p_values);
            fprintf('  %-15s: %.3f\n', dist_name, fpr);
        end
        
        % Calculate overall FPR
        all_biased_p = data.BiasedP;
        fpr_combined = sum(all_biased_p <= p_threshold) / length(all_biased_p);
        fprintf('\nCombined FPR: %.3f\n', fpr_combined);
    else
        fprintf('\nSample counts by Distribution (filtered data):\n');
        for i = 1:n_distributions
            dist_name = unique_distributions{i};
            dist_idx = strcmp(data.Distribution, dist_name);
            n_samples = sum(dist_idx);
            fprintf('  %-15s: %d samples\n', dist_name, n_samples);
        end
    end
    
    fprintf('Total samples analyzed: %d\n', height(data));
    
    % Verify UnbiasedP are uniform
    fprintf('\nVerifying UnbiasedP uniformity:\n');
    all_unbiased = data.UnbiasedP;
    fprintf('  Std:  %.4f\n', std(all_unbiased));
    fprintf('  Min:  %.4f\n', min(all_unbiased));
    fprintf('  Max:  %.4f\n', max(all_unbiased));
    
    fprintf('\nInterpretation:\n');
    fprintf('- Lines above diagonal: Conservative (fewer significant results)\n');
    fprintf('- Lines below diagonal: Liberal (more significant results)\n');
    if ~is_filtered_data
        fprintf('- Perfect calibration: FPR = %.3f\n', p_threshold);
    end
    fprintf('============================================\n');
end

function createSkewnessVsPValueScatter(data, filter_critical, is_filtered_data)
    % Side-by-side scatter plots: Skewness vs Raw P-values
    
    if nargin < 2
        filter_critical = false;
    end
    
    % Don't apply additional filtering if data is already filtered
    if is_filtered_data
        filter_critical = false;
        title_suffix = ' (FILTERED DATA: p ≤ 0.05)';
        x_limit = [0 0.05];
    elseif filter_critical
        data = data(data.BiasedP < 0.05, :);
        title_suffix = ' (P < 0.05)';
        x_limit = [0 0.05];
    else
        title_suffix = '';
        x_limit = [0 1];
    end
    
    % Define distributions
    symmetric_dists = {'NORMAL', 'UNIFORM', 'LAPLACE', 'STUDENT_T'};
    nonsymmetric_dists = {'EXPONENTIAL', 'GAMMA', 'LOGNORMAL'};
        
    is_symmetric = false(height(data), 1);
    is_nonsymmetric = false(height(data), 1);
    
    for i = 1:length(symmetric_dists)
        is_symmetric = is_symmetric | strcmp(data.Distribution, symmetric_dists{i});
    end
    
    for i = 1:length(nonsymmetric_dists)
        is_nonsymmetric = is_nonsymmetric | strcmp(data.Distribution, nonsymmetric_dists{i});
    end
    
    % Create side-by-side subplots
    figure('Position', [300, 200, 1400, 600]);
    
    % Define bins for mean skewness calculation
    bin_width = 0.005;
    bin_edges = 0:bin_width:0.05;
    bin_centers = bin_edges(1:end-1) + bin_width/2;
    
    % LEFT: Symmetric
    subplot(1, 2, 1);
    scatter(data.BiasedP(is_symmetric), data.Skewness(is_symmetric), ...
            5, [0.2 0.4 0.8], 'filled', 'MarkerFaceAlpha', 0.6);
    
    % Add mean skewness bars for each 0.005 bin
    hold on;
    sym_p = data.BiasedP(is_symmetric);
    sym_skew = data.Skewness(is_symmetric);
    
    % Calculate mean skewness values for all bins
    mean_skew_values_sym = [];
    for b = 1:length(bin_centers)
        bin_mask = sym_p >= bin_edges(b) & sym_p < bin_edges(b+1);
        if sum(bin_mask) > 0
            mean_skew_values_sym(end+1) = mean(sym_skew(bin_mask));
        end
    end
    
    % Calculate overall average of mean skewness
    overall_avg_sym = mean(mean_skew_values_sym);
    purple_offset = 0.05; % Offset below the average
    purple_line_y_sym = overall_avg_sym - purple_offset;
    
    % Draw purple horizontal line at calculated position
    plot([0 0.05], [purple_line_y_sym purple_line_y_sym], ...
         'Color', [0.5 0 0.5], 'LineWidth', 2);
    
    % Draw green mean skewness bars
    for b = 1:length(bin_centers)
        bin_mask = sym_p >= bin_edges(b) & sym_p < bin_edges(b+1);
        if sum(bin_mask) > 0
            mean_skew = mean(sym_skew(bin_mask));
            plot([bin_edges(b), bin_edges(b+1)], [mean_skew, mean_skew], ...
                 'Color', [0 1 0], 'LineWidth', 3);
        end
    end
    
    % Add zero horizontal line
    plot([0 0.05], [0 0], 'k-', 'LineWidth', 1);
    
    hold off;
    
    xlabel('Raw P-value', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('Skewness', 'FontSize', 14, 'FontWeight', 'bold');
    title(['Symmetric Distributions' title_suffix], 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    xlim(x_limit);
    ylim([-1 2]);
    
    fprintf('Symmetric - Green line average: %.4f, Purple line at: %.4f\n', ...
        overall_avg_sym, purple_line_y_sym);
    
    % RIGHT: Non-symmetric  
    subplot(1, 2, 2);
    scatter(data.BiasedP(is_nonsymmetric), data.Skewness(is_nonsymmetric), ...
            5, [0.8 0.2 0.2], 'filled', 'MarkerFaceAlpha', 0.6);
    
    % Add mean skewness bars for each 0.005 bin
    hold on;
    nonsym_p = data.BiasedP(is_nonsymmetric);
    nonsym_skew = data.Skewness(is_nonsymmetric);
    
    % Calculate mean skewness values for all bins
    mean_skew_values_nonsym = [];
    for b = 1:length(bin_centers)
        bin_mask = nonsym_p >= bin_edges(b) & nonsym_p < bin_edges(b+1);
        if sum(bin_mask) > 0
            mean_skew_values_nonsym(end+1) = mean(nonsym_skew(bin_mask));
        end
    end
    
    % Calculate overall average of mean skewness
    overall_avg_nonsym = mean(mean_skew_values_nonsym);
    purple_line_y_nonsym = overall_avg_nonsym - purple_offset;
    
    % Draw purple horizontal line at calculated position
    plot([0 0.05], [purple_line_y_nonsym purple_line_y_nonsym], ...
         'Color', [0.5 0 0.5], 'LineWidth', 2);
    
    % Draw green mean skewness bars
    for b = 1:length(bin_centers)
        bin_mask = nonsym_p >= bin_edges(b) & nonsym_p < bin_edges(b+1);
        if sum(bin_mask) > 0
            mean_skew = mean(nonsym_skew(bin_mask));
            plot([bin_edges(b), bin_edges(b+1)], [mean_skew, mean_skew], ...
                 'Color', [0 1 0], 'LineWidth', 3);
        end
    end
    
    % Add zero horizontal line
    plot([0 0.05], [0 0], 'k-', 'LineWidth', 1);
    
    hold off;
    
    xlabel('Raw P-value', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('Skewness', 'FontSize', 14, 'FontWeight', 'bold');
    title(['Non-Symmetric Distributions' title_suffix], 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    xlim(x_limit);
    ylim([-1 2]);
    
    fprintf('Non-symmetric - Green line average: %.4f, Purple line at: %.4f\n', ...
        overall_avg_nonsym, purple_line_y_nonsym);
    
    % Save and close
    if is_filtered_data
        saveas(gcf, 'plots/skewness_vs_pvalue_separated_filtered.png');
    else
        saveas(gcf, 'plots/skewness_vs_pvalue_separated_full.png');
    end
    close(gcf);
    
    % Print statistics
    fprintf('\n--- Skewness vs P-value Statistics%s ---\n', title_suffix);
    fprintf('Symmetric: n=%d\n', sum(is_symmetric));
    fprintf('Non-symmetric: n=%d\n', sum(is_nonsymmetric));
end

function data = loadParquetData(filename)
    % Load data from Parquet file
    [~, ~, ext] = fileparts(filename);
    
    switch lower(ext)
        case '.parquet'
            fprintf('Loading Parquet file: %s\n', filename);
            data = parquetread(filename);
            
            % Remove metadata columns from data if they exist
            metadata_cols = {'nExperimentsPerDist', 'nParticipants', 'ftPermutations', 'timestamp'};
            for i = 1:length(metadata_cols)
                if ismember(metadata_cols{i}, data.Properties.VariableNames)
                    data = removevars(data, metadata_cols{i});
                end
            end
            
        case '.mat'
            fprintf('Loading MAT file: %s\n', filename);
            loaded = load(filename);
            data = loaded.data;
            
        otherwise
            error('Unsupported file format: %s. Use .parquet or .mat', ext);
    end
end

function createSymmetricVariancePlots(data, n_bins, is_filtered_data)
    % Creates individual variance histogram plots for each symmetric distribution
    fprintf('\nGenerating individual variance plots for symmetric distributions...\n');
    
    % Define symmetric distributions
    symmetric_dists = {'NORMAL', 'UNIFORM', 'LAPLACE'};
    
    % Check if Variance column exists
    if ~ismember('Variance', data.Properties.VariableNames)
        fprintf('Warning: Variance column not found in data\n');
        return;
    end
    
    % Filter to only symmetric distributions
    is_symmetric = false(height(data), 1);
    for i = 1:length(symmetric_dists)
        is_symmetric = is_symmetric | contains(data.Distribution, symmetric_dists{i});
    end
    
    symmetric_data = data(is_symmetric, :);
    
    if height(symmetric_data) == 0
        fprintf('Warning: No symmetric distribution data found\n');
        return;
    end
    
    % Create figure with subplots
    figure('Position', [250, 150, 1200, 400]);
    
    % Set default number of bins if not provided
    if nargin < 2 || isempty(n_bins)
        n_bins = 30;
    end
    
    % Define colors for each distribution
    colors = [0.2 0.4 0.8;   % Blue for NORMAL
              0.8 0.6 0.2;   % Orange for UNIFORM  
              0.2 0.8 0.4];  % Green for LAPLACE
    
    for i = 1:length(symmetric_dists)
        subplot(1, length(symmetric_dists), i);
        
        dist_name = symmetric_dists{i};
        
        % Extract variance values for this distribution
        dist_idx = strcmp(symmetric_data.Distribution, dist_name);
        variance_values = symmetric_data.Variance(dist_idx);
        
        % Remove NaN values if any
        variance_values = variance_values(~isnan(variance_values));
        
        if isempty(variance_values)
            fprintf('Warning: No variance data for %s\n', dist_name);
            title(sprintf('%s\n(No Data)', dist_name), 'FontSize', 14, 'FontWeight', 'bold');
            continue;
        end
        
        % Create histogram
        histogram(variance_values, n_bins, 'FaceColor', colors(i,:), ...
            'EdgeColor', colors(i,:) * 0.7, 'LineWidth', 0.5, 'FaceAlpha', 0.7);
        
        % Calculate statistics
        mean_var = mean(variance_values);
        std_var = std(variance_values);
        median_var = median(variance_values);
        
        % Formatting
        xlabel('Variance', 'FontSize', 14, 'FontWeight', 'bold');
        ylabel('Count', 'FontSize', 14);
        title(sprintf('%s\n(n=%d, μ=%.3f, σ=%.3f)', dist_name, length(variance_values), mean_var, std_var), ...
              'FontSize', 14, 'FontWeight', 'bold');
        grid on;
        set(gca, 'GridAlpha', 0.3);
        
        % Add vertical line for mean
        hold on;
        y_lim = ylim();
        plot([mean_var mean_var], y_lim, '--', 'Color', colors(i,:) * 0.5, 'LineWidth', 2);
        hold off;
        
        fprintf('  %s: n=%d, mean=%.4f, std=%.4f, median=%.4f\n', ...
            dist_name, length(variance_values), mean_var, std_var, median_var);
    end
    
    % Overall title
    if is_filtered_data
        sgtitle('Variance Distributions for Symmetric Distributions (FILTERED DATA: p ≤ 0.05)', ...
            'FontSize', 16, 'FontWeight', 'bold');
        saveas(gcf, 'plots/symmetric_variance_plots_filtered.png');
    else
        sgtitle('Variance Distributions for Symmetric Distributions', ...
            'FontSize', 16, 'FontWeight', 'bold');
        saveas(gcf, 'plots/symmetric_variance_plots.png');
    end
    close(gcf);
    
    fprintf('Symmetric variance plots completed.\n');
end

function createBiasedUnbiasedComparison(data, is_filtered_data)
    % Creates k subplots comparing RawP and UnbiasedP distributions
    % One subplot per distribution with 1% bins
    fprintf('\nGenerating Raw vs Unbiased p-value comparison plots...\n');
    
    % Check if UnbiasedP exists
    if ~ismember('UnbiasedP', data.Properties.VariableNames)
        fprintf('Warning: UnbiasedP column not found in data\n');
        return;
    end
    
    unique_distributions = unique(data.Distribution);
    n_distributions = length(unique_distributions);
    grid_size = ceil(sqrt(n_distributions));
    
    % Create figure with extra space for y-axis labels
    figure('Position', [50, 50, 380*grid_size, 320*grid_size]);
    
    % Define histogram bins with 1% (0.01) width
    if is_filtered_data
        bin_width = 0.01;
        bin_edges = 0:bin_width:0.05;
    else
        bin_width = 0.01;
        bin_edges = 0:bin_width:1;
    end
    bin_centers = bin_edges(1:end-1) + bin_width/2;
    
    % Transparency for overlapping histograms
    alpha_value = 0.6;
    
    % Print statistics header
    fprintf('\n--- Raw vs Unbiased P-value Counts (0-0.05 vs Rest) ---\n');
    if is_filtered_data
        fprintf('(Using FILTERED data: BiasedP ≤ 0.05)\n');
    end
    fprintf('%-15s | %10s | %10s | %10s | %10s\n', 'Distribution', 'Raw≤0.05', 'Raw>0.05', 'Unbiased≤0.05', 'Unbiased>0.05');
    fprintf('%s\n', repmat('-', 1, 75));
    
    for i = 1:n_distributions
        subplot(grid_size, grid_size, i);
        
        dist_name = unique_distributions{i};
        
        % Extract p-values for this distribution
        dist_idx = strcmp(data.Distribution, dist_name);
        biased_p_values = data.BiasedP(dist_idx);
        unbiased_p_values = data.UnbiasedP(dist_idx);
        
        % Calculate counts for statistics
        biased_critical = sum(biased_p_values <= 0.05);
        biased_rest = sum(biased_p_values > 0.05);
        unbiased_critical = sum(unbiased_p_values <= 0.05);
        unbiased_rest = sum(unbiased_p_values > 0.05);
        
        % Print statistics for this distribution
        fprintf('%-15s | %10d | %10d | %10d | %10d\n', ...
            dist_name, biased_critical, biased_rest, unbiased_critical, unbiased_rest);
        
        % Create histograms
        hist_biased = histcounts(biased_p_values, bin_edges);
        hist_unbiased = histcounts(unbiased_p_values, bin_edges);
        
        % Plot overlapping histograms
        hold on;
        
        % Plot raw p-values (blue)
        bar(bin_centers, hist_biased, 'BarWidth', 0.8, ...
            'FaceColor', [0.3 0.5 0.9], 'EdgeColor', 'none', ...
            'FaceAlpha', alpha_value);
        
        % Plot unbiased p-values (red)
        bar(bin_centers, hist_unbiased, 'BarWidth', 0.8, ...
            'FaceColor', [0.9 0.3 0.3], 'EdgeColor', 'none', ...
            'FaceAlpha', alpha_value);
        
        hold off;
        
        % Formatting
        if is_filtered_data
            xlim([0 0.05]);
            xticks(0:0.01:0.05);
        else
            xlim([0 1]);
            xticks(0:0.1:1);
        end
        xlabel('P-value', 'FontSize', 14);
        
        % Title with distribution name and FPR only
        if ~is_filtered_data
            fpr_biased = biased_critical / length(biased_p_values);
            title(sprintf('%s (FPR = %.3f)', dist_name, fpr_biased), ...
                'FontSize', 12, 'FontWeight', 'bold');
        else
            title(dist_name, 'FontSize', 12, 'FontWeight', 'bold');
        end
        
        grid on;
        
        % Handle y-axis scaling for large ranges
        max_count = max([hist_biased, hist_unbiased]);
        min_nonzero = min([hist_biased(hist_biased > 0), hist_unbiased(hist_unbiased > 0)]);
        
        if ~isempty(min_nonzero) && max_count / min_nonzero > 1000
            set(gca, 'YScale', 'log');
            ylim([min_nonzero/2, max_count*2]);
            ylabel('Count (log scale)', 'FontSize', 14);
        else
            ylim([0 max(ylim())]);
            ylabel('Count', 'FontSize', 14);
        end
    end
    
    % Overall title
    if is_filtered_data
        sgtitle('Raw vs Unbiased P-Value Distributions (1% bins, FILTERED DATA: p ≤ 0.05)', ...
            'FontSize', 16, 'FontWeight', 'bold');
        saveas(gcf, 'plots/biased_unbiased_comparison_filtered.png');
    else
        sgtitle('Raw vs Unbiased P-Value Distributions (1% bins)', ...
            'FontSize', 16, 'FontWeight', 'bold');
        saveas(gcf, 'plots/biased_unbiased_comparison.png');
    end
    close(gcf);
    
    % Print summary statistics
    fprintf('%s\n', repmat('-', 1, 75));
    
    % Overall totals
    all_biased_critical = sum(data.BiasedP <= 0.05);
    all_biased_rest = sum(data.BiasedP > 0.05);
    all_unbiased_critical = sum(data.UnbiasedP <= 0.05);
    all_unbiased_rest = sum(data.UnbiasedP > 0.05);
    
    fprintf('%-15s | %10d | %10d | %10d | %10d\n', ...
        'TOTAL', all_biased_critical, all_biased_rest, all_unbiased_critical, all_unbiased_rest);
    
    if ~is_filtered_data
        fpr_biased_overall = all_biased_critical / height(data);
        fpr_unbiased_overall = all_unbiased_critical / height(data);
        fprintf('\nOverall FPR - Raw: %.4f, Unbiased: %.4f\n', ...
            fpr_biased_overall, fpr_unbiased_overall);
    end
    
    fprintf('\n');
end

function createSkewnessVsPValueDensity(data, is_filtered_data)
    % 2D density heatmap: Skewness vs Raw P-values
    % Top row: Symmetric distributions (grouped)
    % Bottom row: Individual non-symmetric distributions (EXPONENTIAL, GAMMA, LOGNORMAL)
    
    if nargin < 2
        is_filtered_data = false;
    end
    
    % Set title and limits - FULL RANGE 0-1
    if is_filtered_data
        title_suffix = ' (FILTERED DATA: p ≤ 0.05)';
        x_limit = [0 0.05];
        x_edges = 0:0.0005:0.05;  % Keep fine bins for filtered data
    else
        title_suffix = '';
        x_limit = [0 1];
        x_edges = 0:0.001:1;  % 1000 bins across full range
    end
    
    % Define distributions
    symmetric_dists = {'NORMAL', 'UNIFORM', 'LAPLACE', 'STUDENT_T'};
    nonsymmetric_dists = {'EXPONENTIAL', 'GAMMA', 'LOGNORMAL'};
        
    is_symmetric = false(height(data), 1);
    is_nonsymmetric = false(height(data), 1);
    
    for i = 1:length(symmetric_dists)
        is_symmetric = is_symmetric | strcmp(data.Distribution, symmetric_dists{i});
    end
    
    for i = 1:length(nonsymmetric_dists)
        is_nonsymmetric = is_nonsymmetric | strcmp(data.Distribution, nonsymmetric_dists{i});
    end
    
    % Create 2x3 subplot layout
    figure('Position', [200, 100, 1800, 1000]);
    
    % Define 2D bins
    y_edges = -1:0.025:2;      % Skewness bins
    x_centers = x_edges(1:end-1) + diff(x_edges(1:2))/2;
    y_centers = y_edges(1:end-1) + diff(y_edges(1:2))/2;
    
    % Define bins for mean skewness calculation (green lines)
    bin_width_mean = 0.005;
    if is_filtered_data
        bin_edges_mean = 0:bin_width_mean:0.05;
    else
        bin_edges_mean = 0:bin_width_mean:1;
    end
    
    % TOP ROW: Symmetric Distributions (grouped)
    subplot(2, 3, [1 2 3]);  % Span all three columns
    
    sym_p = data.BiasedP(is_symmetric);
    sym_skew = data.Skewness(is_symmetric);
    
    % Create 2D histogram
    [N_sym, ~, ~] = histcounts2(sym_p, sym_skew, x_edges, y_edges);
    
    % Plot as heatmap
    imagesc(x_centers, y_centers, N_sym');
    set(gca, 'YDir', 'normal');  % Correct y-axis direction
    
    % Color settings
    colormap(jet);
    c = colorbar;
    c.Label.String = 'Count';
    c.Label.FontSize = 12;
    
    % Use log scale for colorbar if range is large
    if max(N_sym(:)) / (min(N_sym(N_sym>0)) + 1) > 100
        set(gca, 'ColorScale', 'log');
        caxis([1, max(N_sym(:))]);  % Start from 1 for log scale
    end
    
    % Add zero line and bin-wise means
    hold on;
    plot(x_limit, [0 0], 'w-', 'LineWidth', 2);  % Zero line in white
    
    % Calculate and plot bin-wise mean skewness (green line)
    for b = 1:length(bin_edges_mean)-1
        bin_mask = sym_p >= bin_edges_mean(b) & sym_p < bin_edges_mean(b+1);
        if sum(bin_mask) > 0
            mean_skew = mean(sym_skew(bin_mask));
            plot([bin_edges_mean(b), bin_edges_mean(b+1)], [mean_skew, mean_skew], ...
                 'Color', [0 1 0], 'LineWidth', 3);
        end
    end
    hold off;
    
    xlabel('Raw P-value', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('Skewness', 'FontSize', 14, 'FontWeight', 'bold');
    title(['Symmetric Distributions (Grouped)' title_suffix], 'FontSize', 14, 'FontWeight', 'bold');
    xlim(x_limit);
    ylim([-1 2]);
    
    fprintf('Symmetric: n=%d samples\n', sum(is_symmetric));
    
    % BOTTOM ROW: Individual Non-symmetric Distributions
    for i = 1:length(nonsymmetric_dists)
        subplot(2, 3, 3 + i);  % Positions 4, 5, 6
        
        dist_name = nonsymmetric_dists{i};
        
        % Filter to this specific distribution
        dist_mask = strcmp(data.Distribution, dist_name);
        dist_p = data.BiasedP(dist_mask);
        dist_skew = data.Skewness(dist_mask);
        
        % Create 2D histogram
        [N_dist, ~, ~] = histcounts2(dist_p, dist_skew, x_edges, y_edges);
        
        % Plot as heatmap
        imagesc(x_centers, y_centers, N_dist');
        set(gca, 'YDir', 'normal');
        
        % Color settings
        colormap(jet);
        c = colorbar;
        c.Label.String = 'Count';
        c.Label.FontSize = 12;
        
        % Use log scale for colorbar if range is large
        if max(N_dist(:)) / (min(N_dist(N_dist>0)) + 1) > 100
            set(gca, 'ColorScale', 'log');
            caxis([1, max(N_dist(:))]);
        end
        
        % Add zero line and bin-wise means
        hold on;
        plot(x_limit, [0 0], 'w-', 'LineWidth', 2);  % Zero line in white
        
        % Calculate and plot bin-wise mean skewness (green line)
        for b = 1:length(bin_edges_mean)-1
            bin_mask = dist_p >= bin_edges_mean(b) & dist_p < bin_edges_mean(b+1);
            if sum(bin_mask) > 0
                mean_skew = mean(dist_skew(bin_mask));
                plot([bin_edges_mean(b), bin_edges_mean(b+1)], [mean_skew, mean_skew], ...
                     'Color', [0 1 0], 'LineWidth', 3);
            end
        end
        hold off;
        
        xlabel('Raw P-value', 'FontSize', 14, 'FontWeight', 'bold');
        ylabel('Skewness', 'FontSize', 14, 'FontWeight', 'bold');
        title([dist_name title_suffix], 'FontSize', 14, 'FontWeight', 'bold');
        xlim(x_limit);
        ylim([-1 2]);
        
        fprintf('%s: n=%d samples\n', dist_name, sum(dist_mask));
    end
    
    % Overall title
    sgtitle('Skewness vs P-value Density Heatmaps', 'FontSize', 16, 'FontWeight', 'bold');
    
    % Save figure
    if is_filtered_data
        saveas(gcf, 'plots/skewness_vs_pvalue_density_filtered.png');
    else
        saveas(gcf, 'plots/skewness_vs_pvalue_density.png');
    end
    close(gcf);
    
    fprintf('Density heatmap saved successfully.\n');
end