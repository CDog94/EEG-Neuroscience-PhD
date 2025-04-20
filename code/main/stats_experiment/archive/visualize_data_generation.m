function visualize_data_generation(progressData, trainingData, skewnessBins)
% Creates a comprehensive visualization of the data generation process
% and final dataset characteristics
%
% Parameters:
%   progressData - Structure containing progress information
%   trainingData - Table containing the generated training data
%   skewnessBins - Bins used for skewness stratification

% Create a figure with multiple subplots
figure('Position', [50, 50, 1400, 900], 'Color', 'white');

% 1. Bin filling progress over iterations
subplot(2, 3, 1);
plot(progressData.iterations, progressData.samplesCollected, 'b-', 'LineWidth', 2);
hold on;
xlabel('Iterations');
ylabel('Total Samples Collected');
title('Data Collection Progress');
grid on;

% 2. Time efficiency
subplot(2, 3, 2);
plot(progressData.iterations, progressData.samplesCollected ./ progressData.elapsedTime, 'r-', 'LineWidth', 2);
xlabel('Iterations');
ylabel('Samples per Second');
title('Collection Efficiency');
grid on;

% 3. Bin filling progress
subplot(2, 3, 3);
binCenters = (skewnessBins(1:end-1) + skewnessBins(2:end)) / 2;
plot(progressData.iterations, progressData.binCounts, 'LineWidth', 1.5);
xlabel('Iterations');
ylabel('Samples per Bin');
title('Bin Filling Progress');
grid on;

% Create a custom legend for the bin plot
legend(arrayfun(@(x,y) sprintf('%.2f-%.2f', x, y), skewnessBins(1:end-1), skewnessBins(2:end), 'UniformOutput', false), ...
    'Location', 'eastoutside', 'FontSize', 8);

% 4. Final distribution of skewness values
subplot(2, 3, 4);
histogram(trainingData.MeasuredSkewness, skewnessBins, 'FaceColor', [0.4, 0.6, 0.8], 'FaceAlpha', 0.7);
xlabel('Bowley''s Skewness');
ylabel('Frequency');
title('Final Distribution of Skewness Values');
grid on;

% 5. Relationship between biased and unbiased p-values, colored by skewness
subplot(2, 3, 5);
scatter(trainingData.BiasedPValue, trainingData.UnbiasedPValue, 10, trainingData.MeasuredSkewness, 'filled', 'MarkerFaceAlpha', 0.5);
hold on;
plot([0, 1], [0, 1], 'r--', 'LineWidth', 1.5);
colormap(jet);
c = colorbar;
c.Label.String = 'Skewness';
xlabel('Biased P-value');
ylabel('Unbiased P-value');
title('P-value Relationship by Skewness');
grid on;
axis([0, 1, 0, 1]);

% 6. Distribution of points by distribution type
subplot(2, 3, 6);
distTypes = unique(trainingData.DistributionType);
numDistTypes = length(distTypes);
colors = jet(numDistTypes);
hold on;

for i = 1:numDistTypes
    idx = strcmp(trainingData.DistributionType, distTypes{i});
    scatter(trainingData.MeasuredSkewness(idx), trainingData.BiasedPValue(idx), 20, colors(i,:), 'filled', 'MarkerFaceAlpha', 0.5);
end

xlabel('Measured Skewness');
ylabel('Biased P-value');
title('Distribution of P-values by Distribution Type');
legend(distTypes, 'Location', 'best');
grid on;

% Add overall title
sgtitle('Data Generation Characteristics', 'FontSize', 16, 'FontWeight', 'bold');

% Save the figure
saveas(gcf, 'data_generation_characteristics.png');
end