%% Main simulation script for Gaussian or Gamma (shifted to median or mean = 0)
clear; close all;

%% ===== User Settings =====
% Choose distribution type: 'gaussian' or 'gamma'
distType = 'gamma';  % Change to 'gaussian' for Gaussian simulation
% Choose centering method for Gamma: 'median' or 'mean'
centerType = 'mean';  % Set to 'mean' if you want to shift by the mean
nExperiments = 5000;  % Number of experiments per parameter value  
n = 50;               % Sample size per group
nPermutations = 5000; % Number of permutations in the permutation test

% For Gaussian, we loop over sigma (standard deviation) values.
% For Gamma, we loop over scale (θ) values with a fixed shape.
if strcmpi(distType, 'gaussian')
    paramValues = 0.2:0.5:1.5;  % These are σ values.
else
    % For Gamma: fix a shape parameter and loop over scale values.
    fixedShape = 2;            % Fixed shape parameter (a)
    paramValues = 0.2:0.5:4.0;  % These will be used as the scale (θ) values.
end

% skewness of 2 top right plot
% compute averages in the 2d plot
% plot gamma densities and p value
% increase the number of permutations with the skewness
% plot two seperate histograms for group 2 data at the bottom

% Settings for p-value histogram: bins of width 0.01 from 0 to 1
edges = 0:0.01:1;
binCenters = edges(1:end-1) + diff(edges)/2;

% Colors for plotting (one per parameter value):
colors = lines(length(paramValues));

%% Create a figure with six subplots:
figure;
    
%% --- Subplot 1: Histogram of p-values ---
subplot(4,2,1); % Create a 3x2 subplot grid, use the first subplot
hold on;
legendEntries = strings(length(paramValues), 1);
allPValues = {}; % Store p-values for each parameter value
allGroup2Data = {}; % Store Group 2 data for each parameter value

% Create progress bar
totalIterations = length(paramValues) * nExperiments;
progressBar = waitbar(0, 'Processing p-value calculations...', 'Name', 'Progress');
iterationCount = 0;

for idx = 1:length(paramValues)
    paramVal = paramValues(idx);
    pvalues = zeros(nExperiments, 1);
    group2Data = cell(nExperiments, 1); % Store Group 2 data for each experiment
    
    for iExp = 1:nExperiments
        % Update progress bar
        iterationCount = iterationCount + 1;
        waitbar(iterationCount/totalIterations, progressBar, ...
            sprintf('Processing parameter %d of %d, experiment %d of %d', ...
            idx, length(paramValues), iExp, nExperiments));
        
        % Group 1: all zeros.
        group1 = zeros(n, 1);
        
        % Group 2: generate data according to the selected distribution.
        if strcmpi(distType, 'gaussian')
            % For Gaussian: generate with mean = 0 and standard deviation = paramVal.
            group2 = normrnd(0, paramVal, [n, 1]);
        else
            % For Gamma: generate gamma data with fixed shape and scale = paramVal.
            % Shift by subtracting the chosen center so that the data have center = 0.
            if strcmpi(centerType, 'median')
                centerVal = gaminv(0.5, fixedShape, paramVal);
            else % use mean
                centerVal = fixedShape * paramVal;
            end
            group2 = gamrnd(fixedShape, paramVal, [n, 1]) - centerVal;
        end
        
        % Store Group 2 data
        group2Data{iExp} = group2;
        
        % Compute the two-tailed p-value using a permutation test (difference of means).
        p = permutationMeanTest(group1, group2, nPermutations);
        pvalues(iExp) = p;
    end
    
    % Store p-values and Group 2 data for later use
    allPValues{idx} = pvalues;
    allGroup2Data{idx} = group2Data;
    
    % Compute histogram counts for the p-values using the specified bins.
    counts = histcounts(pvalues, edges);
    
    % Plot the histogram as a bar plot.
    bar(binCenters, counts, 'FaceColor', colors(idx, :), 'FaceAlpha', 0.5, 'EdgeAlpha', 0.6);
    
    % Create an appropriate legend entry.
    if strcmpi(distType, 'gaussian')
        legendEntries(idx) = sprintf('\\sigma = %.1f', paramVal);
    else
        if strcmpi(centerType, 'median')
            legendEntries(idx) = sprintf('Scale = %.1f, Shape = %.1f (Median)', paramVal, fixedShape);
        else
            legendEntries(idx) = sprintf('Scale = %.1f, Shape = %.1f (Mean)', paramVal, fixedShape);
        end
    end
end

% Close the progress bar
close(progressBar);

xlabel('p-value');
ylabel('Frequency');
if strcmpi(distType, 'gaussian')
    title('Histogram of p-values (Gaussian)');
else
    if strcmpi(centerType, 'median')
        title('Histogram of p-values (Gamma, Median)');
    else
        title('Histogram of p-values (Gamma, Mean)');
    end
end
legend(legendEntries, 'Location', 'northwest');
grid on;
hold off;

%% --- Subplot 2: Theoretical Density Functions ---
subplot(4,2,2); % Use the second subplot
hold on;
legendEntries2 = strings(length(paramValues), 1);
for idx = 1:length(paramValues)
    paramVal = paramValues(idx);
    if strcmpi(distType, 'gaussian')
        % For Gaussian: mean = 0, standard deviation = paramVal.
        sigma = paramVal;
        x = linspace(-4*sigma, 4*sigma, 1000);
        y = normpdf(x, 0, sigma);
        legendEntries2(idx) = sprintf('\\sigma = %.1f', sigma);
    else
        % For Gamma: fixed shape and scale = paramVal.
        theta = paramVal;
        if strcmpi(centerType, 'median')
            centerVal = gaminv(0.5, fixedShape, theta);
        else
            centerVal = fixedShape * theta;
        end
        sigma_gamma = sqrt(fixedShape) * theta;  % standard deviation
        
        % Define an x-range: starting at -centerVal (shifted support) and going up
        % To capture most of the mass, use the 99th percentile of the unshifted gamma.
        xMin = -centerVal;
        xMax = gaminv(0.99, fixedShape, theta) - centerVal;
        x = linspace(xMin, xMax, 1000);
        % Evaluate the density of the shifted gamma:
        y = gampdf(x + centerVal, fixedShape, theta);
        if strcmpi(centerType, 'median')
            legendEntries2(idx) = sprintf('Scale = %.1f, Shape = %.1f (Median)', theta, fixedShape);
        else
            legendEntries2(idx) = sprintf('Scale = %.1f, Shape = %.1f (Mean)', theta, fixedShape);
        end
    end
    plot(x, y, 'Color', colors(idx, :), 'LineWidth', 2);
end
xlabel('x');
ylabel('Probability Density');
if strcmpi(distType, 'gaussian')
    title('Gaussian Densities');
else
    if strcmpi(centerType, 'median')
        title('Gamma Densities (Median)');
    else
        title('Gamma Densities (Mean)');
    end
end
legend(legendEntries2, 'Location', 'northeast');
grid on;
hold off;
    %% --- Subplot 3: Surface Plot: Type I Error Inflation for Gamma (shifted accordingly) ---
    subplot(4,2,3); % Use the third subplot
    
    % Define shape and scale values to ensure skewness ranges from 0.1 to 20
    shapeVec = logspace(log10(0.01), log10(400), 50); % Log-spaced for large range
    scaleVec = linspace(0.01, 4, 5); % Linearly spaced for diversity
    nExpSurface = 500;  % Number of experiments per grid cell
    
    % Preallocate matrices for inflation, standard deviation, and skewness.
    inflationMat = zeros(length(shapeVec), length(scaleVec));
    stdMat = zeros(length(shapeVec), length(scaleVec));
    skewMat = zeros(length(shapeVec), length(scaleVec));
        
    % Create new matrix to store parameters for each iteration
    paramMat = zeros(length(shapeVec), length(scaleVec), 2); % 2 for shapeVal, scaleVal

    % Initialize progress bar
    totalIterations = length(shapeVec) * length(scaleVec);
    progressBar = waitbar(0, 'Processing...');
    
    iterationCount = 0;
    for i = 1:length(shapeVec)
        for j = 1:length(scaleVec)
            iterationCount = iterationCount + 1;
            waitbar(iterationCount / totalIterations, progressBar, ...
                    sprintf('Processing %d of %d', iterationCount, totalIterations));
    
            shapeVal = shapeVec(i);
            scaleVal = scaleVec(j);
            
            if strcmpi(centerType, 'median')
                centerVal = gaminv(0.5, shapeVal, scaleVal);
            else
                centerVal = shapeVal * scaleVal;
            end
    
                % Store parameters in the new matrix
            paramMat(i, j, 1) = shapeVal;
            paramMat(i, j, 2) = scaleVal;

            % Compute theoretical standard deviation and skewness.
            sigmaVal = sqrt(shapeVal) * scaleVal;
            skewVal = 2 / sqrt(shapeVal);
    
            stdMat(i, j) = sigmaVal;
            skewMat(i, j) = skewVal;
    
            % Simulate experiments for this parameter combination.
            pvals = zeros(nExpSurface, 1);
            parfor k = 1:nExpSurface
                group1 = zeros(n, 1);
                group2 = gamrnd(shapeVal, scaleVal, [n, 1]) - centerVal;
                pvals(k) = permutationMeanTest(group1, group2, nPermutations);
            end
    
            % Inflation: fraction of experiments with p < 0.05
            inflationMat(i, j) = mean(pvals < 0.05);
        end
    end
    
    % Close progress bar
    close(progressBar);
    
    % Plot the surface
    surf(stdMat, skewMat, inflationMat);
    xlabel('Standard Deviation');
    ylabel('Skewness');
    zlabel('Type I Error Inflation');
    
    if strcmpi(centerType, 'median')
        title('Type I Error Inflation (Median)');
    else
        title('Type I Error Inflation (Mean)');
    end
    
    colorbar;
    shading interp;

%% subplot test
%% --- Subplot 4: Skewness vs Type I Error Rate ---
% subplot(4,2,4); % Use the fourth subplot
% 
% % Flatten the matrices into column vectors
% x = skewMat(:);
% y = inflationMat(:);
% 
% % Scatter plot
% scatter(x, y, 50, 'filled');
% xlabel('Skewness');
% ylabel('Type I Error Inflation');
% grid on;
% hold on;
% 
% % Fit a second-degree polynomial (adjust degree as needed)
% p = polyfit(x, y, 2); 
% 
% % Generate fitted values
% x_fit = linspace(min(x), max(x), 100); % Smooth line
% y_fit = polyval(p, x_fit);
% 
% % Overlay polynomial fit
% plot(x_fit, y_fit, 'r-', 'LineWidth', 2);
% 
% % Compute R²
% y_pred = polyval(p, x);
% SS_res = sum((y - y_pred).^2);
% SS_tot = sum((y - mean(y)).^2);
% R2 = 1 - (SS_res / SS_tot);
% 
% % Update title with R²
% title(sprintf('Skewness vs. Type I Error Rate (R² = %.3f)', R2));
% 
% hold off;
subplot(4,2,4); % Use the fourth subplot

% Flatten the matrices into column vectors
x = skewMat(:);
y = inflationMat(:);

% Filter data to include only skewness <= 2
valid_indices = x <= 2;
x_filtered = x(valid_indices);
y_filtered = y(valid_indices);

% Scatter plot with filtered data
scatter(x_filtered, y_filtered, 50, 'filled');
xlabel('Skewness');
ylabel('Type I Error Inflation');
grid on;
hold on;

% Create a smooth x range for plotting fits (limit to max skewness of 2)
x_fit = linspace(min(x_filtered), 2, 100);

% Fit a second-degree polynomial (quadratic) on filtered data
p_quad = polyfit(x_filtered, y_filtered, 2);
y_fit_quad = polyval(p_quad, x_fit);
plot(x_fit, y_fit_quad, 'r-', 'LineWidth', 2); % Quadratic fit in red

% Fit a first-degree polynomial (linear) on filtered data
p_lin = polyfit(x_filtered, y_filtered, 1);
y_fit_lin = polyval(p_lin, x_fit);
plot(x_fit, y_fit_lin, 'b--', 'LineWidth', 2); % Linear fit in blue (dashed)

% Compute R² for quadratic fit
y_pred_quad = polyval(p_quad, x_filtered);
SS_res_quad = sum((y_filtered - y_pred_quad).^2);
SS_tot = sum((y_filtered - mean(y_filtered)).^2);
R2_quad = 1 - (SS_res_quad / SS_tot);

% Compute R² for linear fit
y_pred_lin = polyval(p_lin, x_filtered);
SS_res_lin = sum((y_filtered - y_pred_lin).^2);
R2_lin = 1 - (SS_res_lin / SS_tot);

% Compute R² for Skewness ≤ 1 (low skewness region)
low_skew_indices = x_filtered <= 1;
x_low = x_filtered(low_skew_indices);
y_low = y_filtered(low_skew_indices);

% Fit quadratic model in low skewness region
p_low = polyfit(x_low, y_low, 2);
y_pred_low = polyval(p_low, x_low);
SS_res_low = sum((y_low - y_pred_low).^2);
SS_tot_low = sum((y_low - mean(y_low)).^2);
R2_low = 1 - (SS_res_low / SS_tot_low);

% Display polynomial equation on the plot
eqn_text_quad = sprintf('Quad: y = %.3fx^2 + %.3fx + %.3f', p_quad(1), p_quad(2), p_quad(3));
text(min(x_fit), max(y_fit_quad), eqn_text_quad, 'FontSize', 10, 'Color', 'red', 'FontWeight', 'bold');
eqn_text_lin = sprintf('Lin: y = %.3fx + %.3f', p_lin(1), p_lin(2));
text(min(x_fit), max(y_fit_quad) - 0.05, eqn_text_lin, 'FontSize', 10, 'Color', 'blue', 'FontWeight', 'bold');

% Update title with R² values
title(sprintf('Skewness vs. Type I Error Rate (Skewness ≤ 2)\nQuad R² = %.3f, Lin R² = %.3f, Low Skew R² = %.3f', R2_quad, R2_lin, R2_low));
legend({'Data', 'Quadratic Fit', 'Linear Fit'}, 'Location', 'best');
hold off;


%% --- Subplot 5 & 6: Histograms for Specific p-value Ranges ---

% Clear and prepare containers for two p-value bins:
% Bin 1: p in [0, 0.01)  i.e., 0–1%
% Bin 2: p in [0.99, 1]  i.e., 99–100%
group2Data_bin1 = [];  % For p-values between 0 and 0.01
group2Data_bin2 = [];  % For p-values between 0.99 and 1

% Loop through all stored p-values and corresponding Group 2 data
for idx = 1:length(paramValues)
    pvalues = allPValues{idx};
    group2Data = allGroup2Data{idx};
    
    for iExp = 1:nExperiments
        p = pvalues(iExp);
        group2 = group2Data{iExp};
        
        if p >= 0 && p < 0.01
            group2Data_bin1 = [group2Data_bin1; group2];
        elseif p >= 0.99 && p <= 1
            group2Data_bin2 = [group2Data_bin2; group2];
        end
    end
end

% Determine bin edges for each histogram separately.
numBins = 20; % Number of bins

% For p-values between 0–1%
if ~isempty(group2Data_bin1)
    minValue1 = min(group2Data_bin1);
    maxValue1 = max(group2Data_bin1);
    binEdges1 = linspace(minValue1, maxValue1, numBins + 1);
    probMass_bin1 = mean(group2Data_bin1 < 0);
else
    binEdges1 = [];
end

% For p-values between 99–100%
if ~isempty(group2Data_bin2)
    minValue2 = min(group2Data_bin2);
    maxValue2 = max(group2Data_bin2);
    binEdges2 = linspace(minValue2, maxValue2, numBins + 1);
    probMass_bin2 = mean(group2Data_bin2 < 0);
else
    binEdges2 = [];
end

%% Plot for p-value 0–1%
subplot(4,2,5);  % Left subplot (first column in row three)
hold on;
if ~isempty(group2Data_bin1)
    histogram(group2Data_bin1, binEdges1, 'Normalization', 'pdf', ...
              'FaceColor', 'r', 'FaceAlpha', 0.5, 'EdgeAlpha', 0.6);
end
xlabel('Group 2 Data Values');
ylabel('Density');
title(sprintf('Group 2 Data for p-value 0–1%% (P(X<0) = %.3f)', probMass_bin1));
xlim([-12, 12]);
grid on;
hold off;

%% Plot for p-value 99–100%
subplot(4,2,6);  % Right subplot (second column in row three)
hold on;
if ~isempty(group2Data_bin2)
    histogram(group2Data_bin2, binEdges2, 'Normalization', 'pdf', ...
              'FaceColor', 'b', 'FaceAlpha', 0.5, 'EdgeAlpha', 0.6);
end
xlabel('Group 2 Data Values');
ylabel('Density');
title(sprintf('Group 2 Data for p-value 99–100%% (P(X<0) = %.3f)', probMass_bin2));
xlim([-12, 12]);
grid on;
hold off;


%% additional bit of analysis for howard
% Check if there is data in the red distribution
%% additional bit of analysis for howard
% Check if there is data in the red distribution
if ~isempty(group2Data_bin1)
    % Parameters
    nExperiments = 5000;              % Number of experiments to run
    sampleSize = 50;                 % Sample size for each experiment
    nPerm = 1000;                    % Number of permutations per experiment
    
    % Preallocate arrays
    allObsMeans = zeros(nExperiments, 1);
    allPValues = zeros(nExperiments, 1);
    allPermStats = zeros(nExperiments, nPerm);
    
    % Run multiple experiments
    for exp = 1:nExperiments
        % Make sure we have enough data points
        if length(group2Data_bin1) < sampleSize
            error('Not enough data points in red distribution for the specified sample size');
        end
        
        % Randomly select sampleSize data points from group2Data_bin1
        sampleIndices = randperm(length(group2Data_bin1), sampleSize);
        redSample = group2Data_bin1(sampleIndices);
        
        % Compute the observed test statistic: mean of the sample
        obsMean = mean(redSample);
        allObsMeans(exp) = obsMean;
        
        % Generate the permutation (null) distribution by sign flipping
        permStats = zeros(nPerm, 1);
        for i = 1:nPerm
            % Randomly flip signs: generate a vector of +1 and -1
            signs = randi([0, 1], sampleSize, 1) * 2 - 1;  % Maps 0->-1 and 1->+1
            % Apply sign flip to the sample
            permSample = redSample .* signs;
            % Compute the test statistic (mean) for the permuted sample
            permStats(i) = mean(permSample);
        end
        
        % Store all permutation statistics for this experiment
        allPermStats(exp, :) = permStats;
        
        % Compute the two-tailed p-value
        allPValues(exp) = mean(abs(permStats) >= abs(obsMean));
    end
    
    % Aggregate all permutation statistics across experiments
    aggregatedPermStats = reshape(allPermStats, [], 1);
    
    % Calculate average observed mean
    avgObsMean = mean(allObsMeans);
    
    % Plot the aggregated null distribution
    subplot(4,2,[7])
    histogram(aggregatedPermStats, 'Normalization', 'pdf', 'FaceColor','r', 'FaceAlpha', 0.5);
    hold on;
    
    % Mark the average observed mean on the plot
    xline(avgObsMean, 'r', 'LineWidth', 2);
    
    % Calculate percentage of p-values < 0.05
    percentSignificant = mean(allPValues < 0.05) * 100;
    
    xlabel('Mean Value');
    ylabel('Probability Density');
    title(sprintf('Data Sampled from Red Distribution (%d experiments) - Avg Mean = %.3f, %.1f%% significant', ...
          nExperiments, avgObsMean, percentSignificant));
    grid on;
    hold off;
    
    % Additional informative plot: histogram of observed means
    % figure;
    % histogram(allObsMeans, 'Normalization', 'pdf', 'FaceColor', 'r', 'FaceAlpha', 0.5);
    % hold on;
    % xline(avgObsMean, 'r', 'LineWidth', 2);
    % xlabel('Observed Mean Values');
    % ylabel('Probability Density');
    % title(sprintf('Distribution of Observed Means Across %d Experiments', nExperiments));
    % grid on;
    % hold off;
else
    disp('No data available in the red distribution for the permutation test.');
end

%% second plot of group 2 
if ~isempty(group2Data_bin2)
    % Parameters
    nExperiments = 5000;              % Number of experiments to run
    sampleSize = 50;                 % Sample size for each experiment
    nPerm = 1000;                    % Number of permutations per experiment
    
    % Preallocate arrays
    allObsMeans = zeros(nExperiments, 1);
    allPValues = zeros(nExperiments, 1);
    allPermStats = zeros(nExperiments, nPerm);
    
    % Run multiple experiments
    for exp = 1:nExperiments
        % Make sure we have enough data points
        if length(group2Data_bin2) < sampleSize
            error('Not enough data points in red distribution for the specified sample size');
        end
        
        % Randomly select sampleSize data points from group2Data_bin1
        sampleIndices = randperm(length(group2Data_bin2), sampleSize);
        redSample = group2Data_bin2(sampleIndices);
        
        % Compute the observed test statistic: mean of the sample
        obsMean = mean(redSample);
        allObsMeans(exp) = obsMean;
        
        % Generate the permutation (null) distribution by sign flipping
        permStats = zeros(nPerm, 1);
        for i = 1:nPerm
            % Randomly flip signs: generate a vector of +1 and -1
            signs = randi([0, 1], sampleSize, 1) * 2 - 1;  % Maps 0->-1 and 1->+1
            % Apply sign flip to the sample
            permSample = redSample .* signs;
            % Compute the test statistic (mean) for the permuted sample
            permStats(i) = mean(permSample);
        end
        
        % Store all permutation statistics for this experiment
        allPermStats(exp, :) = permStats;
        
        % Compute the two-tailed p-value
        allPValues(exp) = mean(abs(permStats) >= abs(obsMean));
    end
    
    % Aggregate all permutation statistics across experiments
    aggregatedPermStats = reshape(allPermStats, [], 1);
    
    % Calculate average observed mean
    avgObsMean = mean(allObsMeans);
    
    % Plot the aggregated null distribution
    subplot(4,2,8)
    histogram(aggregatedPermStats, 'Normalization', 'pdf', 'FaceColor', 'b', 'FaceAlpha', 0.5);
    hold on;
    
    % Mark the average observed mean on the plot
    xline(avgObsMean, 'r', 'LineWidth', 2);
    
    % Calculate percentage of p-values < 0.05
    percentSignificant = mean(allPValues < 0.05) * 100;
    
    xlabel('Mean Value');
    ylabel('Probability Density');
    title(sprintf('Data Sampled from Blue Distribution (%d experiments) - Avg Mean = %.3f, %.1f%% significant', ...
          nExperiments, avgObsMean, percentSignificant));
    grid on;
    hold off;
    
    % Additional informative plot: histogram of observed means
    % figure;
    % histogram(allObsMeans, 'Normalization', 'pdf', 'FaceColor', 'b', 'FaceAlpha', 0.5);
    % hold on;
    % xline(avgObsMean, 'r', 'LineWidth', 2);
    % xlabel('Observed Mean Values');
    % ylabel('Probability Density');
    % title(sprintf('Distribution of Observed Means Across %d Experiments', nExperiments));
    % grid on;
    % hold off;
else
    disp('No data available in the red distribution for the permutation test.');
end
savePath = 'C:\Users\CDoga\Documents\Research\EEG-Neuroscience-PhD\code\first_plot.png';
exportgraphics(gcf, savePath, 'Resolution', 300, 'BackgroundColor', 'white');
close(gcf);
%% Plot skewness vs. inflation with matching gamma distributions

% Create a 2x1 figure
figure('Position', [100, 100, 900, 700]);

% Flatten the matrices into column vectors
x = skewMat(:);
y = inflationMat(:);

% Get shape and scale values from paramMat
shape_vals = paramMat(:,:,1);
scale_vals = paramMat(:,:,2);
shape_flat = shape_vals(:);
scale_flat = scale_vals(:);

% Filter data to include only skewness <= 2
valid_indices = x <= 2;
x_filtered = x(valid_indices);
y_filtered = y(valid_indices);
shape_filtered = shape_flat(valid_indices);
scale_filtered = scale_flat(valid_indices);

% Top subplot - Scatter plot
subplot(2, 1, 1);

% Basic scatter plot with filtered data (all in gray initially)
scatter(x_filtered, y_filtered, 50, [0.7 0.7 0.7], 'filled');
xlabel('Skewness');
ylabel('Type I Error Inflation');
grid on;
hold on;

% Create a smooth x range for plotting fits (limit to max skewness of 2)
x_fit = linspace(min(x_filtered), 2, 100);

% Fit a second-degree polynomial (quadratic) on filtered data
p_quad = polyfit(x_filtered, y_filtered, 2);
y_fit_quad = polyval(p_quad, x_fit);
plot(x_fit, y_fit_quad, 'r-', 'LineWidth', 2); % Quadratic fit in red

% Fit a first-degree polynomial (linear) on filtered data
p_lin = polyfit(x_filtered, y_filtered, 1);
y_fit_lin = polyval(p_lin, x_fit);
plot(x_fit, y_fit_lin, 'b--', 'LineWidth', 2); % Linear fit in blue (dashed)

% Compute R² for quadratic fit
y_pred_quad = polyval(p_quad, x_filtered);
SS_res_quad = sum((y_filtered - y_pred_quad).^2);
SS_tot = sum((y_filtered - mean(y_filtered)).^2);
R2_quad = 1 - (SS_res_quad / SS_tot);

% Compute R² for linear fit
y_pred_lin = polyval(p_lin, x_filtered);
SS_res_lin = sum((y_filtered - y_pred_lin).^2);
R2_lin = 1 - (SS_res_lin / SS_tot);

% Compute R² for Skewness ≤ 1 (low skewness region)
low_skew_indices = x_filtered <= 1;
x_low = x_filtered(low_skew_indices);
y_low = y_filtered(low_skew_indices);

% Fit quadratic model in low skewness region
p_low = polyfit(x_low, y_low, 2);
y_pred_low = polyval(p_low, x_low);
SS_res_low = sum((y_low - y_pred_low).^2);
SS_tot_low = sum((y_low - mean(y_low)).^2);
R2_low = 1 - (SS_res_low / SS_tot_low);

% Define colors for the selected points
colors = {'b', 'r', 'g', 'm', 'c'};

% Bottom subplot - 5 gamma distributions
subplot(2, 1, 2);
hold on;

% Randomly select 5 different indices from the filtered data
num_points = 5;
total_points = length(x_filtered);
rng(42); % Set seed for reproducibility
selected_indices = randperm(total_points, num_points);

% Arrays to store the selected values
selected_shapes = zeros(1, num_points);
selected_scales = zeros(1, num_points);
selected_skewness = zeros(1, num_points);
selected_x = zeros(1, num_points);
selected_y = zeros(1, num_points);
legendEntries = cell(1, num_points);

% Plot the 5 gamma distributions and highlight corresponding points in scatter plot
for i = 1:num_points
    idx = selected_indices(i);
    selected_shapes(i) = shape_filtered(idx);
    selected_scales(i) = scale_filtered(idx);
    selected_skewness(i) = x_filtered(idx); % Skewness value
    selected_x(i) = x_filtered(idx);
    selected_y(i) = y_filtered(idx);
    
    % Plot the gamma distribution
    subplot(2, 1, 2);
    plotSingleGamma(selected_shapes(i), selected_scales(i), colors{i});
    legendEntries{i} = sprintf('Shape=%.2f, Scale=%.2f, Skew=%.2f', selected_shapes(i), selected_scales(i), selected_skewness(i));
    
    % Highlight the corresponding point in the scatter plot
    subplot(2, 1, 1);
    scatter(selected_x(i), selected_y(i), 100, colors{i}, 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
end

% Finalize the top subplot
subplot(2, 1, 1);
% Display polynomial equation on the plot
eqn_text_quad = sprintf('Quad: y = %.3fx^2 + %.3fx + %.3f', p_quad(1), p_quad(2), p_quad(3));
%text(min(x_fit), max(y_fit_quad), eqn_text_quad, 'FontSize', 10, 'Color', 'red', 'FontWeight', 'bold');
eqn_text_lin = sprintf('Lin: y = %.3fx + %.3f', p_lin(1), p_lin(2));
%text(min(x_fit), max(y_fit_quad) - 0.05, eqn_text_lin, 'FontSize', 10, 'Color', 'blue', 'FontWeight', 'bold');

% Update title with R² values
title(sprintf('Skewness vs. Type I Error Rate (Skewness ≤ 2)\nQuad R² = %.3f, Lin R² = %.3f, Low Skew R² = %.3f', R2_quad, R2_lin, R2_low));
legend({'Data Points', 'Quadratic Fit', 'Linear Fit', 'Selected Point 1', 'Selected Point 2', 'Selected Point 3', 'Selected Point 4', 'Selected Point 5'}, 'Location', 'best');

% Finalize the bottom subplot
subplot(2, 1, 2);
% Add legend for gamma distributions
legend(legendEntries, 'Location', 'northeast');
title('Mean-Centered Gamma Distributions for Selected Points');
xlabel('Value');
ylabel('Probability Density');
xlim([-150, 150])
ylim([0, 0.1])

% Adjust layout
sgtitle('Relationship Between Skewness and Type I Error Rate', 'FontSize', 14, 'FontWeight', 'bold');
set(gcf, 'Color', 'w');
savePath = 'C:\Users\CDoga\Documents\Research\EEG-Neuroscience-PhD\code\second_plot.png';
exportgraphics(gcf, savePath, 'Resolution', 300, 'BackgroundColor', 'white');
    close(gcf);

%% --- Local Function: Permutation Test Using Difference of Means ---
function p = permutationMeanTest(group1, group2, nPermutations)
    % Calculate the observed difference of means.
    obsDiff = mean(group1) - mean(group2);
    
    % Combine both groups.
    combined = [group1; group2];
    n = length(group1);  % assumes equal group sizes
    
    % Preallocate an array for permutation differences.
    permDiff = zeros(nPermutations, 1);
    
    for j = 1:nPermutations
        % Randomly permute the combined data.
        permutedIdx = randperm(length(combined));
        
        % Divide the permuted data into two groups.
        permGroup1 = combined(permutedIdx(1:n));
        permGroup2 = combined(permutedIdx(n+1:end));
        
        % Compute the difference of means for this permutation.
        permDiff(j) = mean(permGroup1) - mean(permGroup2);
    end
    
    % The two-tailed p-value is the fraction of permuted differences (in absolute value)
    % that are at least as large as the observed absolute difference.
    p = mean(abs(permDiff) >= abs(obsDiff));
end

function plotSingleGamma(shape, scale, plotColor)
% PLOTSINGLEGAMMA Creates a plot of a mean-centered gamma distribution
%
% Inputs:
%   shape     - Shape parameter of the gamma distribution
%   scale     - Scale parameter of the gamma distribution
%   plotColor - Color for the plot (default: blue)
%
% Example:
%   plotSingleGamma(2, 1.5);
%   plotSingleGamma(0.5, 1, 'r');

% Set default color if not provided
if nargin < 3 || isempty(plotColor)
    plotColor = 'b';
end

% Calculate mean (centering value)
centerVal = shape * scale;

% Calculate standard deviation and skewness
sigma = sqrt(shape) * scale;
skewness = 2 / sqrt(shape);

% Define x-range: starting at -centerVal (shifted support) and extending
% to capture most of the density (99th percentile)
xMin = -centerVal;
xMax = gaminv(0.99, shape, scale) - centerVal;
x = linspace(xMin, xMax, 1000);

% Evaluate the density of the shifted gamma
y = gampdf(x + centerVal, shape, scale);

% Create figure if one doesn't exist
if isempty(get(0, 'CurrentFigure'))
    figure;
else
    % If we're in a subplot context, just use the current axes
    if ~isempty(get(gcf, 'CurrentAxes'))
        hold on;
    end
end

% Plot the distribution
plot(x, y, 'Color', plotColor, 'LineWidth', 2);

% Add labels and title
xlabel('x');
ylabel('Probability Density');
title(sprintf('Gamma Distribution (Mean-Centered): Shape=%.2f, Scale=%.2f', shape, scale));

% Add a text box with distribution properties
textStr = {
    sprintf('Shape = %.2f', shape), 
    sprintf('Scale = %.2f', scale),
    sprintf('StdDev = %.2f', sigma),
    sprintf('Skewness = %.2f', skewness)
};

%text(0.7, 0.9, textStr, 'Units', 'normalized', 'BackgroundColor', [0.9 0.9 0.9]);
grid on;

% If we started with hold on, leave it that way
if ~ishold
    hold off;
end
end