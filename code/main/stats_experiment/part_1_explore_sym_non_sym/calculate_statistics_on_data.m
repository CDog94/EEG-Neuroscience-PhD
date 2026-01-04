%% Modified PGI Analysis Script with Skewness-based T-value Analysis
% This script loads mean intercept data for all participants, calculates PGI,
% computes skewness across participants, and calculates average t-values
% for high skewness (|skewness| >= 1) and low skewness (|skewness| < 1) regions
% for post-stimulus time points.

%% Main function to execute the entire analysis
function main_pgi_skewness_analysis()
    % Set paths
    main_path = 'C:\Users\CDoga\Documents\Research\PhD\participant_';
    n_participants = 40;
    
    % Path to the stat.mat file
    stat_file_path = 'C:\Users\CDoga\Documents\Research\PhD\results\time_domain\56_256ms\mean_intercept\no-factor\stat.mat';
    
    % Run the analysis
    analyze_skewness_tvalues(main_path, n_participants, stat_file_path);
end

%% Function to load data, calculate PGI, compute skewness, and analyze t-values
function analyze_skewness_tvalues(main_path, n_participants, stat_file_path)
    % Step 1: Load PGI data for all participants
    [pgi_matrix, participant_ids] = load_pgi_matrix(main_path, n_participants);
    
    % Check if data was loaded successfully
    if isempty(pgi_matrix)
        error('No data could be loaded. Check the paths and data files.');
    end
    
    % Get dimensions
    [n_participants, n_timepoints, n_channels] = size(pgi_matrix);
    fprintf('Loaded data: %d participants, %d timepoints, %d channels\n', ...
            n_participants, n_timepoints, n_channels);
    
    % Step 2: Calculate skewness across participants for each time×space point
    skewness_map = zeros(n_channels, n_timepoints);
    for c = 1:n_channels
        for t = 1:n_timepoints
            % Extract data across participants for this channel and time point
            data_point = squeeze(pgi_matrix(:, t, c));
            
            % Calculate skewness
            skewness_map(c, t) = skewness(data_point);
        end
    end
    
    % Step 3: Load a sample dataset to get time info and electrode labels
    sample_path = [main_path, num2str(participant_ids(1))];
    cd([sample_path, '/time_domain']);
    load('time_domain_mean_intercept_onsets_2_3_4_5_6_7_8_grand-average.mat');
    
    % Get time vector
    time_vector = data.time{1};
    
    % Find index where time >= 0 (post-stimulus)
    post_stimulus_idx = find(time_vector >= 0, 1);
    
    % Create mask for post-stimulus time points
    post_stimulus_mask = false(size(skewness_map));
    post_stimulus_mask(:, post_stimulus_idx:end) = true;
    
    % Create masks for high and low skewness
    high_skewness_mask = abs(skewness_map) >= 1;
    low_skewness_mask = abs(skewness_map) < 1;
    
    % Combine with post-stimulus mask
    high_skewness_post = high_skewness_mask & post_stimulus_mask;
    low_skewness_post = low_skewness_mask & post_stimulus_mask;
    
    % Step 4: Load the stat.mat file containing t-values
    fprintf('Loading t-values from %s\n', stat_file_path);
    load(stat_file_path);
    
    % Extract t-values
    if isfield(stat, 'stat')
        t_values = stat.stat;
        fprintf('Loaded t-values with dimensions: %s\n', mat2str(size(t_values)));
    else
        error('The stat field was not found in the loaded stat.mat file.');
    end
    
    % Step 5: Align time points between skewness data and t-values
    % Load time information from stat
    if isfield(stat, 'time')
        stat_time = stat.time;
    else
        warning('Time information not found in stat. Assuming same time points as PGI data.');
        stat_time = time_vector;
    end
    
    % Find matching time points
    [~, pgi_time_idx, stat_time_idx] = align_time_vectors(time_vector, stat_time);
    
    % Adjust masks and t-values to aligned time points
    if ~isempty(pgi_time_idx) && ~isempty(stat_time_idx)
        % Adjust masks
        high_skewness_post_aligned = high_skewness_post(:, pgi_time_idx);
        low_skewness_post_aligned = low_skewness_post(:, pgi_time_idx);
        
        % Adjust t-values
        t_values_aligned = t_values(:, stat_time_idx);
        
        % Check if sizes match after alignment
        if size(high_skewness_post_aligned, 2) ~= size(t_values_aligned, 2)
            warning('Size mismatch after alignment. Attempting to use common time points.');
            % Use the minimum common length
            min_length = min(size(high_skewness_post_aligned, 2), size(t_values_aligned, 2));
            high_skewness_post_aligned = high_skewness_post_aligned(:, 1:min_length);
            low_skewness_post_aligned = low_skewness_post_aligned(:, 1:min_length);
            t_values_aligned = t_values_aligned(:, 1:min_length);
        end
    else
        warning('Could not align time vectors. Using original data.');
        high_skewness_post_aligned = high_skewness_post;
        low_skewness_post_aligned = low_skewness_post;
        t_values_aligned = t_values;
    end
    
    % Step 6: Calculate average t-values for high and low skewness regions
    % High skewness regions (|skewness| >= 1)
    t_values_high_skewness = t_values_aligned(high_skewness_post_aligned);
    avg_t_high_skewness = mean(t_values_high_skewness(:), 'omitnan');
    
    % Low skewness regions (|skewness| < 1)
    t_values_low_skewness = t_values_aligned(low_skewness_post_aligned);
    avg_t_low_skewness = mean(t_values_low_skewness(:), 'omitnan');
    
    % Step 7: Display the results
    fprintf('\n----- RESULTS -----\n');
    fprintf('Post-stimulus results (t >= 0):\n');
    fprintf('High skewness regions (|skewness| >= 1):\n');
    fprintf('  - Number of data points: %d\n', numel(t_values_high_skewness));
    fprintf('  - Average t-value: %.4f\n', avg_t_high_skewness);
    fprintf('Low skewness regions (|skewness| < 1):\n');
    fprintf('  - Number of data points: %d\n', numel(t_values_low_skewness));
    fprintf('  - Average t-value: %.4f\n', avg_t_low_skewness);
    
    % Calculate additional statistics
    high_skewness_count = sum(high_skewness_post_aligned(:));
    low_skewness_count = sum(low_skewness_post_aligned(:));
    total_post_points = high_skewness_count + low_skewness_count;
    
    fprintf('\nDistribution of post-stimulus data points:\n');
    fprintf('  - High skewness: %d (%.1f%%)\n', high_skewness_count, 100*high_skewness_count/total_post_points);
    fprintf('  - Low skewness: %d (%.1f%%)\n', low_skewness_count, 100*low_skewness_count/total_post_points);
    
    % Step 8: Create visualizations
    create_skewness_t_value_plots(skewness_map, t_values_aligned, time_vector, high_skewness_post_aligned, low_skewness_post_aligned, pgi_matrix, post_stimulus_idx);
end

%% Function to align two time vectors and find matching indices
function [common_time, pgi_indices, stat_indices] = align_time_vectors(pgi_time, stat_time)
    % Initialize output
    common_time = [];
    pgi_indices = [];
    stat_indices = [];
    
    % Find tolerance based on the smallest time step in either vector
    tol = min(min(diff(pgi_time)), min(diff(stat_time))) / 2;
    
    % Find common time points within tolerance
    for i = 1:length(pgi_time)
        % Find the closest stat_time point to this pgi_time point
        [min_diff, idx] = min(abs(stat_time - pgi_time(i)));
        
        % If within tolerance, consider it a match
        if min_diff <= tol
            common_time(end+1) = pgi_time(i);
            pgi_indices(end+1) = i;
            stat_indices(end+1) = idx;
        end
    end
    
    % Report on the alignment
    fprintf('Time alignment: Found %d matching time points out of %d PGI time points and %d stat time points\n', ...
            length(common_time), length(pgi_time), length(stat_time));
end

%% Function to create visualizations of the skewness and t-value analysis
function create_skewness_t_value_plots(skewness_map, t_values, time_vector, high_skewness_mask, low_skewness_mask, pgi_matrix, post_stimulus_idx)
    % Extract all skewness values and t-values from post-stimulus regions
    skewness_values = skewness_map(high_skewness_mask | low_skewness_mask);
    t_vals = t_values(high_skewness_mask | low_skewness_mask);
    
    % Remove any NaN values
    valid_idx = ~isnan(skewness_values) & ~isnan(t_vals);
    skewness_values = skewness_values(valid_idx);
    t_vals = t_vals(valid_idx);
    
    % Calculate absolute skewness values
    abs_skewness = abs(skewness_values);
    
    % Figure 1: Bar chart of average t-values by skewness range
    figure('Position', [100, 100, 1000, 600]);
    
    % Define skewness bins with 0.5 increments - ensure they're strictly increasing
    skew_edges = [0, 1, 1.5, 2.0, 2.5, 3.0];  % Bin edges with 0.5 increments
    n_bins = length(skew_edges) - 1;
    
    % Initialize arrays for results
    avg_t_values = zeros(n_bins, 1);
    sample_sizes = zeros(n_bins, 1);
    bin_centers = zeros(n_bins, 1);
    
    % Calculate bin centers (midpoint of each bin)
    for i = 1:n_bins
        bin_centers(i) = (skew_edges(i) + skew_edges(i+1)) / 2;
    end
    
    % Verify bin centers are strictly increasing
    if any(diff(bin_centers) <= 0)
        error('Bin centers must be strictly increasing');
    end
    
    % Calculate average t-value for each skewness bin
    for i = 1:n_bins
        % Find t-values in this skewness range
        if i == 1  % First bin is |skewness| <= 1
            bin_idx = abs_skewness <= 1;
        else
            bin_idx = abs_skewness > skew_edges(i) & abs_skewness <= skew_edges(i+1);
        end
        
        % Calculate average t-value and sample size for this bin
        t_in_bin = t_vals(bin_idx);
        if ~isempty(t_in_bin)
            avg_t_values(i) = mean(t_in_bin, 'omitnan');
            sample_sizes(i) = sum(~isnan(t_in_bin));
        else
            avg_t_values(i) = NaN;
            sample_sizes(i) = 0;
        end
    end
    
    % Create the main plot
    bar_handle = bar(bin_centers, avg_t_values);
    
    % Set color based on skewness range
    low_skew_color = [0.3, 0.6, 0.9]; % Blue for low skewness
    high_skew_color = [0.9, 0.3, 0.3]; % Red for high skewness
    
    % Color the first bar differently (low skewness)
    bar_handle.FaceColor = 'flat';
    bar_handle.CData(1,:) = low_skew_color;
    bar_handle.CData(2:end,:) = repmat(high_skew_color, n_bins-1, 1);
    
    % Add labels and title
    xlabel('Absolute Skewness', 'FontSize', 14);
    ylabel('Average t-value', 'FontSize', 14);
    title('Average t-values by Absolute Skewness Range (Post-Stimulus)', 'FontSize', 16);
    
    % Add vertical line to separate low and high skewness regions
    hold on;
    xline(1.05, 'k--', 'LineWidth', 1.5);
    
    % Customize x-axis with custom labels
    ax = gca;
    ax.XTick = bin_centers;
    
    % Create custom x-tick labels
    tick_labels = cell(n_bins, 1);
    tick_labels{1} = '\leq 1';
    for i = 2:n_bins
        % Show the range for each bin
        tick_labels{i} = sprintf('%.1f-%.1f', skew_edges(i), skew_edges(i+1));
    end
    
    % Set x-tick labels
    ax.XTickLabel = tick_labels;
    xtickangle(45);
    
    % Add zero line for reference
    yline(0, 'k:', 'LineWidth', 1);
    
    % Add grid
    grid on;
    
    % Create sample size text above each bar
    for i = 1:n_bins
        if ~isnan(avg_t_values(i)) && sample_sizes(i) > 0
            text(bin_centers(i), avg_t_values(i), ['n=', num2str(sample_sizes(i))], ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
                'FontSize', 8, 'Color', 'k');
        end
    end
    
    % Create text box with summary
    low_skew_avg = avg_t_values(1);
    high_skew_avg = mean(avg_t_values(2:end), 'omitnan');
    low_skew_n = sample_sizes(1);
    high_skew_n = sum(sample_sizes(2:end));
    
    annotation('textbox', [0.15, 0.8, 0.3, 0.15], ...
               'String', {sprintf('Low Skewness (|skew| ≤ 1): %.4f (n=%d)', low_skew_avg, low_skew_n), ...
                          sprintf('High Skewness (|skew| > 1): %.4f (n=%d)', high_skew_avg, high_skew_n)}, ...
               'EdgeColor', 'k', ...
               'BackgroundColor', [0.95, 0.95, 0.95], ...
               'FitBoxToText', 'on', ...
               'FontSize', 10);
    
    % Save the figure
    saveas(gcf, 'skewness_tvalue_bins.png');
    fprintf('Bar chart saved as skewness_tvalue_bins.png\n');
    
    % Figure 2: Skewness heatmap with extreme values highlighted
    figure('Position', [100, 100, 1200, 800]);
    
    % Find index where time >= 0 (post-stimulus)
    post_stimulus_idx = find(time_vector >= 0, 1);
    
    % Create mask for extreme values (>1 or <-1) occurring after time=0
    extreme_mask = (skewness_map > 1 | skewness_map < -1);
    post_stimulus_mask_full = false(size(skewness_map));
    post_stimulus_mask_full(:, post_stimulus_idx:end) = true;
    
    % Combine masks to get extreme values after time=0
    highlight_mask = extreme_mask & post_stimulus_mask_full;
    
    % Calculate proportion of highlighted points to total points
    total_points = numel(skewness_map);
    highlighted_points = sum(highlight_mask(:));
    highlight_proportion = highlighted_points / total_points;
    
    % Plot the basic skewness heatmap
    imagesc(time_vector, 1:size(skewness_map, 1), skewness_map);
    
    % Add colorbar and set its label
    c = colorbar;
    ylabel(c, 'Skewness');
    
    % Set appropriate colormap
    colormap('parula');
    
    % Now highlight extreme post-stimulus values by overlaying a second plot
    hold on;
    
    % Create a matrix for plotting only highlighted values
    highlight_map = nan(size(skewness_map)); % NaN for non-highlighted values
    highlight_map(highlight_mask) = skewness_map(highlight_mask);
    
    % Plot highlighted values in red
    h = imagesc(time_vector, 1:size(skewness_map, 1), highlight_map);
    set(h, 'AlphaData', ~isnan(highlight_map)); % Make NaN values transparent
    
    % Apply a red colormap to the highlighted values
    % Create a red-based RGB image for the highlights
    rgb_highlight = nan(size(highlight_map,1), size(highlight_map,2), 3);
    
    % Fill with red where we have values
    valid_idx = ~isnan(highlight_map);
    rgb_highlight(:,:,1) = ones(size(highlight_map)); % Red channel at max
    rgb_highlight(:,:,2) = zeros(size(highlight_map)); % Green channel at 0
    rgb_highlight(:,:,3) = zeros(size(highlight_map)); % Blue channel at 0
    
    % Make everything else transparent
    alpha_channel = zeros(size(highlight_map));
    alpha_channel(valid_idx) = 0.7; % Semi-transparent red
    
    % Update the highlight image
    set(h, 'CData', rgb_highlight);
    set(h, 'AlphaData', alpha_channel);
    
    % Add labels and title with proportion of highlighted points
    xlabel('Time (s)', 'FontSize', 14);
    ylabel('Channel', 'FontSize', 14);
    title_str = sprintf('Skewness of PGI Values (%.1f%% of points highlighted)', ...
                       highlight_proportion * 100);
    title(title_str, 'FontSize', 16);
    
    % Add line at time = 0
    xline(0, '--w', 'LineWidth', 1.5);
    
    % Add text annotation explaining the red highlights
    annotation('textbox', [0.15, 0.02, 0.7, 0.05], ...
               'String', 'Red highlights: Post-stimulus (t>=0) points with extreme skewness (|skewness| > 1)', ...
               'EdgeColor', 'none', ...
               'Color', 'red', ...
               'FontWeight', 'bold', ...
               'HorizontalAlignment', 'center');
    
    % Save the figure
    saveas(gcf, 'skewness_heatmap.png');
    fprintf('Heatmap saved as skewness_heatmap.png\n');
    
    % Figure 3: Histogram of channel means (MODIFIED TO SHOW MEAN PGI VALUES)
    figure('Position', [100, 100, 800, 600]);
    
    % Calculate mean PGI value for each channel across all participants and post-stimulus time
    % Extract post-stimulus data from pgi_matrix
    post_stimulus_data = pgi_matrix(:, post_stimulus_idx:end, :);
    
    % Calculate the mean for each channel across participants and time
    % Reshape to combine participants and time dimensions
    [n_participants, n_post_times, n_channels] = size(post_stimulus_data);
    reshaped_data = reshape(post_stimulus_data, n_participants * n_post_times, n_channels);
    
    % Calculate mean for each channel
    channel_means = mean(reshaped_data, 1, 'omitnan')';  % Transpose to make it a column vector
    
    % Create histogram of channel means
    histogram(channel_means, 40, 'FaceColor', [0.4, 0.4, 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.7);
    
    % Add labels and title
    xlabel('Mean PGI Value', 'FontSize', 14);
    ylabel('Frequency', 'FontSize', 14);
    title('Distribution of Mean PGI Values Across Channels (Post-Stimulus)', 'FontSize', 16);
    
    % Add grid
    grid on;
    
    % Add vertical line at PGI = 0
    xline(0, 'k--', 'LineWidth', 1.5);
    
    % Add statistics annotation
    mean_val = mean(channel_means, 'omitnan');
    median_val = median(channel_means, 'omitnan');
    std_val = std(channel_means, 'omitnan');
    skew_val = skewness(channel_means);
    
    stats_str = {
        sprintf('Mean: %.3f', mean_val),
        sprintf('Median: %.3f', median_val),
        sprintf('Std Dev: %.3f', std_val),
        sprintf('Skewness: %.3f', skew_val)
    };
    
    annotation('textbox', [0.7, 0.7, 0.2, 0.2], ...
               'String', stats_str, ...
               'EdgeColor', 'k', ...
               'BackgroundColor', [0.95, 0.95, 0.95], ...
               'FitBoxToText', 'on', ...
               'FontSize', 10);
    
    % Save the figure
    saveas(gcf, 'channel_mean_pgi_histogram.png');
    fprintf('Histogram saved as channel_mean_pgi_histogram.png\n');
    
    % Figure 4: Scatter plot with Spearman's correlation
    figure('Position', [100, 100, 1000, 800]);
    
    % Create scatter plot
    scatter(skewness_values, t_vals, 20, 'filled', 'MarkerFaceAlpha', 0.5);
    
    % Calculate Spearman's correlation
    [rho, pval] = corr(skewness_values(:), t_vals(:), 'Type', 'Spearman', 'rows', 'complete');
    
    % Add labels and title
    xlabel('Skewness', 'FontSize', 14);
    ylabel('t-value', 'FontSize', 14);
    title('T-values vs. Skewness (Post-Stimulus)', 'FontSize', 16);
    
    % Add grid
    grid on;
    
    % Add reference lines
    xline(0, 'k:', 'LineWidth', 1);
    yline(0, 'k:', 'LineWidth', 1);
    xline(-1, 'r--', 'LineWidth', 1.5);
    xline(1, 'r--', 'LineWidth', 1.5);
    
    % Add Spearman's correlation results as text
    corr_str = {
        sprintf('Spearman''s \\rho = %.4f', rho),
        sprintf('p-value = %.4e', pval),
        sprintf('n = %d', length(skewness_values))
    };
    
    % Determine significance level text
    if pval < 0.001
        sig_text = '***';
    elseif pval < 0.01
        sig_text = '**';
    elseif pval < 0.05
        sig_text = '*';
    else
        sig_text = 'n.s.';
    end
    
    % Add significance to the text box
    corr_str{1} = sprintf('Spearman''s \\rho = %.4f %s', rho, sig_text);
    
    annotation('textbox', [0.15, 0.8, 0.25, 0.15], ...
               'String', corr_str, ...
               'EdgeColor', 'k', ...
               'BackgroundColor', [0.95, 0.95, 0.95], ...
               'FitBoxToText', 'on', ...
               'FontSize', 12, ...
               'FontWeight', 'bold');
    
    % Add text explaining the red lines
    annotation('textbox', [0.65, 0.02, 0.3, 0.05], ...
               'String', 'Red lines: |skewness| = 1', ...
               'EdgeColor', 'none', ...
               'Color', 'red', ...
               'HorizontalAlignment', 'center');
    
    % Optionally add a trend line (using robust regression)
    hold on;
    
    % Sort data for plotting the trend
    [sorted_skew, sort_idx] = sort(skewness_values);
    
    % Calculate a smooth trend using local polynomial regression (loess)
    try
        % Use smooth function if available
        smoothed_t = smooth(sorted_skew, t_vals(sort_idx), 0.3, 'loess');
        plot(sorted_skew, smoothed_t, 'b-', 'LineWidth', 2);
        legend('Data points', 'LOESS trend', 'Location', 'best');
    catch
        % If smooth is not available, skip the trend line
        fprintf('Note: Could not add LOESS trend line. The smooth function may not be available.\n');
    end
    
    % Save the figure
    saveas(gcf, 'skewness_tvalue_scatter.png');
    fprintf('Scatter plot saved as skewness_tvalue_scatter.png\n');
    
    % Print the correlation results to console as well
    fprintf('\n----- SPEARMAN''S RANK CORRELATION -----\n');
    fprintf('Correlation between skewness and t-values (post-stimulus):\n');
    fprintf('Spearman''s rho = %.4f\n', rho);
    fprintf('p-value = %.4e\n', pval);
    fprintf('Significance: %s\n', sig_text);
    if abs(rho) < 0.3
        fprintf('Interpretation: Weak correlation\n');
    elseif abs(rho) < 0.7
        fprintf('Interpretation: Moderate correlation\n');
    else
        fprintf('Interpretation: Strong correlation\n');
    end
end


%% Function to load PGI data as a matrix of size participants × time × space
function [pgi_matrix, participant_ids] = load_pgi_matrix(main_path, n_participants)
    % Initialize variables
    participant_ids = [];
    pgi_cells = {};
    count = 0;
    
    % Data file to load
    data_file = 'time_domain_mean_intercept_onsets_2_3_4_5_6_7_8_grand-average.mat';
    
    % Loop through participants
    for i = 1:n_participants
        participant_path = [main_path, num2str(i)];
        
        % Check if participant directory exists
        if exist(participant_path, 'dir')
            cd(participant_path);
            
            % Check if time_domain directory exists
            if exist('time_domain', 'dir')
                cd('time_domain');
                
                % Check if data file exists
                if isfile(data_file)
                    % Load the data
                    load(data_file);
                    
                    % Calculate PGI
                    pgi = data.med - (data.thin + data.thick)/2;
                    
                    % Store the participant data
                    count = count + 1;
                    pgi_cells{count} = pgi;
                    participant_ids(count) = i;
                end
            end
        end
    end
    
    % If no participants were found, return empty matrix
    if count == 0
        pgi_matrix = [];
        return;
    end
    
    % Get dimensions from the first participant
    [n_channels, n_timepoints] = size(pgi_cells{1});
    
    % Create the matrix (participants × time × channels)
    pgi_matrix = zeros(count, n_timepoints, n_channels);
    
    % Fill the matrix
    for p = 1:count
        for c = 1:n_channels
            pgi_matrix(p, :, c) = pgi_cells{p}(c, :);
        end
    end
    
    fprintf('Successfully loaded PGI data for %d out of %d participants\n', count, n_participants);
end

%% Call the main function when script is executed
main_pgi_skewness_analysis();