% Main script to generate training data for p-value correction model
clear; close all;
rng(42); % Set random seed for reproducibility

%% Setup parameters
nPValues = 5000;             % Reduced number of p-values per parameter value
sampleSize = 100;            % Sample size per group (N in your description)
nPermutations = 1000;       % Number of permutations for permutation test

% Generate 1,000 skewness values in the range 0.1 to 2.0
numSkewnessValues = 23;
skewnessValues = linspace(0.1, 3.0, numSkewnessValues);  % Range of skewness values from 0.1 to 2.0
fixedScale = 3.0;                        % Fixed scale parameter for all distributions

% Display summary of skewness values
fprintf('Generated %d skewness values from %.2f to %.2f\n', length(skewnessValues), min(skewnessValues), max(skewnessValues));
fprintf('First 5 skewness values: %.2f, %.2f, %.2f, %.2f, %.2f\n', skewnessValues(1:5));
fprintf('Last 5 skewness values: %.2f, %.2f, %.2f, %.2f, %.2f\n', skewnessValues(end-4:end));

% Calculate corresponding shape parameters for the first few and last few values
shapeFirst5 = 4 ./ (skewnessValues(1:5).^2);
shapeLast5 = 4 ./ (skewnessValues(end-4:end).^2);

fprintf('First 5 shape parameters: %.2f, %.2f, %.2f, %.2f, %.2f\n', shapeFirst5);
fprintf('Last 5 shape parameters: %.2f, %.2f, %.2f, %.2f, %.2f\n', shapeLast5);

% Start parallel pool if not already started
if isempty(gcp('nocreate'))
    c = parcluster('local');
    c.NumWorkers = 24;
    saveProfile(c);
    parpool('local', 23);
end

% Get worker information
poolObj = gcp('nocreate');
numWorkers = poolObj.NumWorkers;
fprintf('Using %d parallel workers\n', numWorkers);

%% Create a progress monitoring system
progressFile = 'progress_data.txt';
if exist(progressFile, 'file')
    delete(progressFile);
end

% Write initial progress data
fid = fopen(progressFile, 'w');
fprintf(fid, '0\n');
fclose(fid);

% Create a waitbar
h = waitbar(0, 'Starting training data generation...', 'Name', 'Training Data Generation Progress');

totalSimulations = length(skewnessValues) * nPValues;
startTime = tic;

% Initialize a progress counter
pctDone = 0;

%% Generate training data
fprintf('Generating training data...\n');
tic;

% Create temporary files to track completion by parallel workers
for idx = 1:length(skewnessValues)
    tmpFile = sprintf('param_done_%d.txt', idx);
    if exist(tmpFile, 'file')
        delete(tmpFile);
    end
end

trainingData = generateTrainingData(nPValues, sampleSize, skewnessValues, fixedScale, nPermutations);

% Count completed parameter values and update progress bar
numCompleted = 0;
for idx = 1:length(skewnessValues)
    tmpFile = sprintf('param_done_%d.txt', idx);
    if exist(tmpFile, 'file')
        numCompleted = numCompleted + 1;
        delete(tmpFile);  % Clean up
    end
end

% Update waitbar one last time
waitbar(1, h, sprintf('Completed %d/%d skewness values', length(skewnessValues), length(skewnessValues)));

elapsedTime = toc;
fprintf('Training data generation complete in %.2f seconds.\n', elapsedTime);

% Close the waitbar
close(h);

%% Save the training data
save('pvalue_correction_training_data.mat', 'trainingData');
fprintf('Training data saved to pvalue_correction_training_data.mat\n');

%% Display summary statistics
fprintf('\nSummary statistics:\n');
fprintf('Total number of paired p-values: %d\n', height(trainingData));
fprintf('Number of skewness values: %d\n', length(skewnessValues));

% Count significant p-values (p < 0.05) for both distributions
sigBiased = sum(trainingData.BiasedPValue < 0.05);
sigUnbiased = sum(trainingData.UnbiasedPValue < 0.05);

fprintf('Number of significant biased p-values (p < 0.05): %d (%.2f%%)\n', ...
    sigBiased, 100*sigBiased/height(trainingData));
fprintf('Number of significant unbiased p-values (p < 0.05): %d (%.2f%%)\n', ...
    sigUnbiased, 100*sigUnbiased/height(trainingData));

%% Create plots showing the relationship between biased and unbiased p-values
% We'll plot only a subset of skewness values for clarity
plotIndices = round(linspace(1, length(skewnessValues), 8));  % Choose 8 values evenly spaced
plotSkewness = skewnessValues(plotIndices);
plotShapes = 4 ./ (plotSkewness.^2);

figure('Position', [100, 100, 1200, 600]);

% Determine subplot layout
numRows = 2;
numCols = ceil(length(plotSkewness) / numRows);

for s = 1:length(plotSkewness)
    subplot(numRows, numCols, s);
    
    % Find the closest skewness values in our data to the ones we want to plot
    targetSkew = plotSkewness(s);
    [~, closestIdxs] = sort(abs(trainingData.SkewnessParam - targetSkew));
    closestIdxs = closestIdxs(1:min(1000, length(closestIdxs))); % Take the 1000 closest points
    plotData = trainingData(closestIdxs, :);
    
    % Plot p-values
    scatter(plotData.BiasedPValue, plotData.UnbiasedPValue, 10, 'filled', 'MarkerFaceAlpha', 0.3);
    hold on;
    
    % Add diagonal line for reference
    plot([0, 1], [0, 1], 'r--');
    
    % Add title and labels
    title(sprintf('Skewness ≈ %.2f (Shape ≈ %.2f)', targetSkew, 4/(targetSkew^2)));
    xlabel('Biased p-value (Gamma)');
    ylabel('Unbiased p-value (Uniform)');
    grid on;
    axis([0, 0.05, 0, 0.05]);
    
    % Add text showing percentage of significant p-values
    sigGamma = 100 * sum(plotData.BiasedPValue < 0.05) / height(plotData);
    sigUniform = 100 * sum(plotData.UnbiasedPValue < 0.05) / height(plotData);
    text(0.6, 0.1, sprintf('Gamma p < 0.05: %.1f%%', sigGamma), 'FontSize', 8);
    text(0.6, 0.05, sprintf('Uniform p < 0.05: %.1f%%', sigUniform), 'FontSize', 8);
end

sgtitle('Relationship between Biased and Unbiased p-values by Skewness Parameter');
saveas(gcf, 'pvalue_relationship_by_skewness.png');

%% Focus on significant p-values (p < 0.05)
figure('Position', [100, 100, 1200, 600]);

for s = 1:length(plotSkewness)
    subplot(numRows, numCols, s);
    
    % Find the closest skewness values in our data to the ones we want to plot
    targetSkew = plotSkewness(s);
    [~, closestIdxs] = sort(abs(trainingData.SkewnessParam - targetSkew));
    closestIdxs = closestIdxs(1:min(1000, length(closestIdxs))); % Take the 1000 closest points
    plotData = trainingData(closestIdxs, :);
    
    % Further filter to focus on small p-values
    sigData = plotData(plotData.BiasedPValue < 0.05, :);
    
    % Plot p-values
    scatter(sigData.BiasedPValue, sigData.UnbiasedPValue, 10, 'filled', 'MarkerFaceAlpha', 0.3);
    hold on;
    
    % Add diagonal line for reference
    plot([0, 0.05], [0, 0.05], 'r--');
    
    % Add title and labels
    title(sprintf('Skewness ≈ %.2f (p < 0.05)', targetSkew));
    xlabel('Biased p-value (Gamma)');
    ylabel('Unbiased p-value (Uniform)');
    grid on;
    axis([0, 0.05, 0, 0.05]);
end

sgtitle('Relationship between Significant Biased and Unbiased p-values');
saveas(gcf, 'significant_pvalue_relationship.png');


%% Create histograms of p-values across skewness ranges
figure('Position', [100, 100, 1200, 600]);

% Settings for p-value histogram: bins of width 0.01 from 0 to 1
edges = 0:0.01:1;
binCenters = edges(1:end-1) + diff(edges)/2;

% Define skewness ranges for grouping
skewnessRanges = [0.1 0.5; 0.5 1.0; 1.0 1.5; 1.5 2.0];
numRanges = size(skewnessRanges, 1);

% Colors for plotting
colors = lines(numRanges);

% --- Plot 1: Gamma Distribution p-values ---
subplot(1, 2, 1);
hold on;
legendEntries = strings(numRanges, 1);

for idx = 1:numRanges
    
    % Get range boundaries
    lowerBound = skewnessRanges(idx, 1);
    upperBound = skewnessRanges(idx, 2);
    
    % Filter p-values in this skewness range
    rangeFilter = trainingData.SkewnessParam >= lowerBound & trainingData.SkewnessParam < upperBound;
    pvalues = trainingData.BiasedPValue(rangeFilter);
    
    % Compute histogram counts
    counts = histcounts(pvalues, edges);
    
    % Normalize counts to percentage for easier comparison
    countsPercent = 100 * counts / sum(counts);
    
    % Plot the histogram as a bar plot
    bar(binCenters, countsPercent, 'FaceColor', colors(idx, :), 'FaceAlpha', 0.5, 'EdgeAlpha', 0.6);
    ylim([0, 2])
    
    % Create legend entry
    legendEntries(idx) = sprintf('Skewness %.1f-%.1f', lowerBound, upperBound);
    
    % Calculate percentage of significant p-values
    sigPercent = 100 * sum(pvalues < 0.05) / length(pvalues);
    fprintf('Skewness range %.1f-%.1f: %.2f%% significant p-values\n', lowerBound, upperBound, sigPercent);
end

xlabel('p-value');
ylabel('Percentage of Total (%)');
title('Histogram of p-values (Gamma Distribution)');
legend(legendEntries, 'Location', 'northwest');
grid on;
hold off;

% --- Plot 2: Uniform Distribution p-values ---
subplot(1, 2, 2);
hold on;

% Create a single histogram for uniform distribution
allUniformPValues = trainingData.UnbiasedPValue;
counts = histcounts(allUniformPValues, edges);

% Normalize counts to percentage
countsPercent = 100 * counts / sum(counts);
bar(binCenters, countsPercent, 'FaceColor', [0.3 0.7 0.9], 'FaceAlpha', 0.7);
ylim([0, 2])
xlabel('p-value');
ylabel('Percentage of Total (%)');
title('Histogram of p-values (Uniform Distribution)');
grid on;
hold off;

% Add a common title to the figure
sgtitle('P-value Distributions: Gamma vs Uniform');
saveas(gcf, 'pvalue_histograms.png');


% Cleanup temporary files
if exist('progress_data.txt', 'file'), delete('progress_data.txt'); end

fprintf('Analysis complete! Elapsed time: %.2f seconds\n', elapsedTime);