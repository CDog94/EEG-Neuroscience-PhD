function visualize_test_results(chiSquaredResults, weibullResults, lognormalResults, ...
    chiSquaredDfValues, weibullShapeValues, lognormalSigmaValues, savePath)
% Visualizes the results of testing with multiple distributions
%
% Parameters:
%   chiSquaredResults - Results for Chi-Squared distributions
%   weibullResults - Results for Weibull distributions
%   lognormalResults - Results for Log-normal distributions
%   chiSquaredDfValues - Array of chi-squared df values tested
%   weibullShapeValues - Array of Weibull shape parameters tested
%   lognormalSigmaValues - Array of log-normal sigma parameters tested

% Calculate combined RMSE
chiCount = length(chiSquaredResults.UnbiasedPValues);
weibCount = length(weibullResults.UnbiasedPValues);
logNCount = length(lognormalResults.UnbiasedPValues);
totalCount = chiCount + weibCount + logNCount;

combinedRMSE = sqrt((chiSquaredResults.RMSE_Overall^2 * chiCount + ...
                    weibullResults.RMSE_Overall^2 * weibCount + ...
                    lognormalResults.RMSE_Overall^2 * logNCount) / totalCount);

% Create a multi-panel figure for comparing all distributions
figure('Position', [50, 50, 1200, 1800], 'Renderer', 'painters'); % Changed to painters renderer for vector graphics
set(gcf, 'Color', 'white'); % Set white background
fontName = 'Arial';
fontSize = 12;
titleSize = 14;
mainTitleSize = 16;

% Row 1: Boxplots of Skewness for all distributions
ax1 = subplot(4, 1, 1); % Now 4 rows instead of 3

% Create data for boxplot
allSkew = [chiSquaredResults.MeasuredSkewness; 
           weibullResults.MeasuredSkewness; 
           lognormalResults.MeasuredSkewness];

% Create grouping vector
groupVec = zeros(size(allSkew));
totalParams = length(chiSquaredDfValues) + length(weibullShapeValues) + length(lognormalSigmaValues);
labelCell = cell(1, totalParams);

% Assign group numbers for Chi-Squared
paramIdx = 1;
for d = 1:length(chiSquaredDfValues)
    idx = find(chiSquaredResults.Parameter == chiSquaredDfValues(d));
    groupVec(idx) = paramIdx;
    labelCell{paramIdx} = sprintf('χ²(df=%d)', chiSquaredDfValues(d));
    paramIdx = paramIdx + 1;
end

% Assign group numbers for Weibull
weibullStart = length(chiSquaredResults.MeasuredSkewness) + 1;
for s = 1:length(weibullShapeValues)
    idx = weibullStart - 1 + find(weibullResults.Parameter == weibullShapeValues(s));
    groupVec(idx) = paramIdx;
    labelCell{paramIdx} = sprintf('Weib(s=%.1f)', weibullShapeValues(s));
    paramIdx = paramIdx + 1;
end

% Assign group numbers for Log-normal
lognormalStart = weibullStart + length(weibullResults.MeasuredSkewness);
for s = 1:length(lognormalSigmaValues)
    idx = lognormalStart - 1 + find(lognormalResults.Parameter == lognormalSigmaValues(s));
    groupVec(idx) = paramIdx;
    labelCell{paramIdx} = sprintf('LogN(σ=%.1f)', lognormalSigmaValues(s));
    paramIdx = paramIdx + 1;
end

% Create boxplot with proper format
boxplotHandle = boxplot(allSkew, groupVec, 'Labels', labelCell, 'Symbol', 'r+', 'OutlierSize', 6);
set(boxplotHandle, 'LineWidth', 1.2);
set(gca, 'XTickLabelRotation', 45); % Rotate x-axis labels for better readability

% Improve text formatting
title('Measured Bowley''s Skewness Across Distributions', 'FontSize', titleSize, 'FontName', fontName, 'FontWeight', 'bold');
ylabel('Skewness', 'FontSize', fontSize, 'FontName', fontName, 'FontWeight', 'bold');
set(gca, 'FontSize', fontSize-1, 'FontName', fontName, 'XGrid', 'on', 'YGrid', 'on', 'Box', 'on');

% Add distribution type color coding
hold on;
numChiParams = length(chiSquaredDfValues);
numWeibParams = length(weibullShapeValues);
numLogNParams = length(lognormalSigmaValues);

% Set box colors by distribution type
h = findobj(gca, 'Tag', 'Box');
for j = 1:length(h)
    if j <= numLogNParams
        patch(get(h(j), 'XData'), get(h(j), 'YData'), [0.8, 0.4, 0.2], 'FaceAlpha', 0.5);
    elseif j <= numLogNParams + numWeibParams
        patch(get(h(j), 'XData'), get(h(j), 'YData'), [0.2, 0.6, 0.8], 'FaceAlpha', 0.5);
    else
        patch(get(h(j), 'XData'), get(h(j), 'YData'), [0.4, 0.8, 0.4], 'FaceAlpha', 0.5);
    end
end

% Make sure outlier markers are visible above box colors
h_outliers = findobj(gca, 'Tag', 'Outliers');
for j = 1:length(h_outliers)
    uistack(h_outliers(j), 'top');
end

% Remove legend (as requested)
% leg1 = legend({'Log-normal', 'Weibull', 'Chi-Squared'}, 'Location', 'NorthEast');
% set(leg1, 'FontSize', fontSize-1, 'Box', 'off', 'Position', [0.85, 0.92, 0.1, 0.05]);

% Fix ylim to better show outliers
ylim([-0.6, 0.8]);

% Row 2: Scatter plot of predicted vs. true p-values for all distributions
ax2 = subplot(4, 1, 2);

% Create a combined scatter plot for all distributions
hold on;
s1 = scatter(chiSquaredResults.UnbiasedPValues, chiSquaredResults.PredictedPValues, 25, 'filled', 'MarkerFaceColor', [0.4, 0.8, 0.4], 'MarkerFaceAlpha', 0.5);
s2 = scatter(weibullResults.UnbiasedPValues, weibullResults.PredictedPValues, 25, 'filled', 'MarkerFaceColor', [0.2, 0.6, 0.8], 'MarkerFaceAlpha', 0.5);
s3 = scatter(lognormalResults.UnbiasedPValues, lognormalResults.PredictedPValues, 25, 'filled', 'MarkerFaceColor', [0.8, 0.4, 0.2], 'MarkerFaceAlpha', 0.5);
p1 = plot([0, 1], [0, 1], 'k--', 'LineWidth', 2);

% Improve text formatting
% Removed title as requested
xlabel('True P-Value', 'FontSize', fontSize, 'FontName', fontName, 'FontWeight', 'bold');
ylabel('Predicted P-Value', 'FontSize', fontSize, 'FontName', fontName, 'FontWeight', 'bold');
set(gca, 'FontSize', fontSize-1, 'FontName', fontName, 'XGrid', 'on', 'YGrid', 'on', 'Box', 'on');
axis square;
axis([0 1 0 1]);

% Improve legend appearance
leg2 = legend([s1, s2, s3, p1], {'Chi-Squared', 'Weibull', 'Log-normal', 'Ideal'}, 'Location', 'SouthEast');
set(leg2, 'FontSize', fontSize-1, 'Box', 'off');

% Row 3: Bar chart of RMSE by distribution and parameter
ax3 = subplot(4, 1, 3);

% Create data for grouped bar chart
numGroups = 3; % Chi-squared, Weibull, Log-normal
maxBars = max([length(chiSquaredDfValues), length(weibullShapeValues), length(lognormalSigmaValues)]);
barData = nan(maxBars, numGroups);

% Fill bar data matrix
barData(1:length(chiSquaredDfValues), 1) = chiSquaredResults.RMSE_ByParam;
barData(1:length(weibullShapeValues), 2) = weibullResults.RMSE_ByParam;
barData(1:length(lognormalSigmaValues), 3) = lognormalResults.RMSE_ByParam;

% Create grouped bar chart
barHandles = bar(barData, 'grouped');
set(barHandles(1), 'FaceColor', [0.4, 0.8, 0.4]); % Chi-squared
set(barHandles(2), 'FaceColor', [0.2, 0.6, 0.8]); % Weibull
set(barHandles(3), 'FaceColor', [0.8, 0.4, 0.2]); % Log-normal

% Find worst performing distribution from each class
[~, worstChiIdx] = max(chiSquaredResults.RMSE_ByParam);
[~, worstWeibIdx] = max(weibullResults.RMSE_ByParam);
[~, worstLogNIdx] = max(lognormalResults.RMSE_ByParam);

% Extract parameters for the worst performers
worstChiDF = chiSquaredDfValues(worstChiIdx);
worstWeibShape = weibullShapeValues(worstWeibIdx);
worstLogNSigma = lognormalSigmaValues(worstLogNIdx);

% Extract RMSE values for the worst performers
worstChiRMSE = chiSquaredResults.RMSE_ByParam(worstChiIdx);
worstWeibRMSE = weibullResults.RMSE_ByParam(worstWeibIdx);
worstLogNRMSE = lognormalResults.RMSE_ByParam(worstLogNIdx);

% Highlight the worst performing bars with different colors
% We'll use a simpler approach using hatched patterns
hold on;

% Create x-coordinates for the bars (need to calculate correct x positions)
numBars = size(barData, 2);
xvals = 1:size(barData, 1);
width = 0.8;  % Default width
groupwidth = width/numBars;

% Calculate x-coordinates for each bar group
offsets = (1:numBars) - (numBars+1)/2;
offsets = offsets * groupwidth;

% Add patterned patches for worst performers
% Chi-Squared worst (green)
x = xvals(worstChiIdx) + offsets(1);
y = chiSquaredResults.RMSE_ByParam(worstChiIdx);
patch([x-groupwidth/2 x+groupwidth/2 x+groupwidth/2 x-groupwidth/2], ...
      [0 0 y y], [0.0, 0.5, 0.0], 'FaceAlpha', 0.7, 'EdgeColor', 'k', 'LineWidth', 1.5);

% Weibull worst (blue)
x = xvals(worstWeibIdx) + offsets(2);
y = weibullResults.RMSE_ByParam(worstWeibIdx);
patch([x-groupwidth/2 x+groupwidth/2 x+groupwidth/2 x-groupwidth/2], ...
      [0 0 y y], [0.0, 0.3, 0.6], 'FaceAlpha', 0.7, 'EdgeColor', 'k', 'LineWidth', 1.5);

% Log-normal worst (orange)
x = xvals(worstLogNIdx) + offsets(3);
y = lognormalResults.RMSE_ByParam(worstLogNIdx);
patch([x-groupwidth/2 x+groupwidth/2 x+groupwidth/2 x-groupwidth/2], ...
      [0 0 y y], [0.6, 0.2, 0.0], 'FaceAlpha', 0.7, 'EdgeColor', 'k', 'LineWidth', 1.5);

% Improve legend appearance
leg3 = legend({sprintf('Chi-Squared (worst: df=%d)', worstChiDF), ...
        sprintf('Weibull (worst: shape=%.1f)', worstWeibShape), ...
        sprintf('Log-normal (worst: sigma=%.2f)', worstLogNSigma)}, ...
        'Location', 'NorthEast');
set(leg3, 'FontSize', fontSize-1, 'Box', 'off');

% Add parameter labels above worst bars (only for the worst ones)
text(worstChiIdx, worstChiRMSE + 0.01, ...
    sprintf('df=%d', worstChiDF), ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', fontSize-2);

text(worstWeibIdx, worstWeibRMSE + 0.01, ...
    sprintf('shape=%.1f', worstWeibShape), ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', fontSize-2);

text(worstLogNIdx, worstLogNRMSE + 0.01, ...
    sprintf('sigma=%.2f', worstLogNSigma), ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', fontSize-2);

% Improve text formatting
% Removed title as requested
ylabel('RMSE', 'FontSize', fontSize, 'FontName', fontName, 'FontWeight', 'bold');
set(gca, 'FontSize', fontSize-1, 'FontName', fontName, 'XGrid', 'on', 'YGrid', 'on', 'Box', 'on');

% Set y-axis limit for better visualization
ylim([0, max([worstChiRMSE, worstWeibRMSE, worstLogNRMSE])*1.2]);

% Create clear x-axis labels
xLabels = cell(1, maxBars);
for i = 1:maxBars
    xLabels{i} = sprintf('Set %d', i);
end
set(gca, 'XTickLabel', xLabels);

% Remove parameter info text (as requested)
% paramText = sprintf('Chi-Squared (df): [%s]\nWeibull (shape): [%s]\nLog-normal (sigma): [%s]', ...
%     regexprep(mat2str(chiSquaredDfValues), '\s+', ' '), ...
%     regexprep(mat2str(weibullShapeValues), '\s+', ' '), ...
%     regexprep(mat2str(lognormalSigmaValues), '\s+', ' '));
% 
% text(0.5, -0.25, paramText, 'Units', 'normalized', ...
%     'HorizontalAlignment', 'center', 'FontName', fontName, 'FontSize', fontSize-2);

% NEW ROW 4: Add worst-performing distribution shapes
% Create 3-panel subplot for the worst performers
ax4 = subplot(4, 3, 10);  % Chi-Squared
ax5 = subplot(4, 3, 11);  % Weibull  
ax6 = subplot(4, 3, 12);  % Log-normal

% Define colors
chiColor = [0.4, 0.8, 0.4];  % Green for Chi-squared
weibColor = [0.2, 0.6, 0.8];  % Blue for Weibull
logNColor = [0.8, 0.4, 0.2];  % Orange for Log-normal

% Plot Chi-Squared worst performer
axes(ax4);
hold on;
x = linspace(-5, 15, 1000);
mean_chi = worstChiDF;
y = chi2pdf(x + mean_chi, worstChiDF);
plot(x, y, 'Color', chiColor, 'LineWidth', 2.5);
plot([0, 0], [0, max(y)*1.1], 'k--', 'LineWidth', 1.5);
title(sprintf('Chi-Squared (RMSE: %.4f, df=%d)', worstChiRMSE, worstChiDF), 'FontSize', fontSize, 'FontName', fontName, 'FontWeight', 'bold');
xlabel('Centered Value', 'FontSize', fontSize-1, 'FontName', fontName);
ylabel('Probability Density', 'FontSize', fontSize-1, 'FontName', fontName);
grid on;
xlim([-5, 15]);
% Set appropriate ylim
ylim([0, max(y)*1.2]);
set(gca, 'FontSize', fontSize-2, 'FontName', fontName);

% Plot Weibull worst performer
axes(ax5);
hold on;
x = linspace(-2, 5, 1000);
scale = 1;
mean_weibull = scale * gamma(1 + 1/worstWeibShape);
y_raw = wblpdf(x + mean_weibull, scale, worstWeibShape);
plot(x, y_raw, 'Color', weibColor, 'LineWidth', 2.5);
plot([0, 0], [0, max(y_raw)*1.1], 'k--', 'LineWidth', 1.5);
title(sprintf('Weibull (RMSE: %.4f, shape=%.1f)', worstWeibRMSE, worstWeibShape), 'FontSize', fontSize, 'FontName', fontName, 'FontWeight', 'bold');
xlabel('Centered Value', 'FontSize', fontSize-1, 'FontName', fontName);
ylabel('Probability Density', 'FontSize', fontSize-1, 'FontName', fontName);
grid on;
xlim([-2, 5]);
% Set appropriate ylim
ylim([0, max(y_raw)*1.2]);
set(gca, 'FontSize', fontSize-2, 'FontName', fontName);

% Plot Log-normal worst performer
axes(ax6);
hold on;
x = linspace(-2, 8, 1000);
mu = -worstLogNSigma^2/2;
y_raw = lognpdf(x + 1, mu, worstLogNSigma);
plot(x, y_raw, 'Color', logNColor, 'LineWidth', 2.5);
plot([0, 0], [0, max(y_raw)*1.1], 'k--', 'LineWidth', 1.5);
title(sprintf('Log-normal (RMSE: %.4f, sigma=%.2f)', worstLogNRMSE, worstLogNSigma), 'FontSize', fontSize, 'FontName', fontName, 'FontWeight', 'bold');
xlabel('Centered Value', 'FontSize', fontSize-1, 'FontName', fontName);
ylabel('Probability Density', 'FontSize', fontSize-1, 'FontName', fontName);
grid on;
xlim([-2, 8]);
% Set appropriate ylim based on the peak
ylim([0, max(y_raw)*1.2]);
set(gca, 'FontSize', fontSize-2, 'FontName', fontName);

% Remove the annotation title for the bottom row completely
% No title needed since each subplot has its own title

% Adjust the spacing between subplots
drawnow;
set(gcf, 'PaperPositionMode', 'auto');
set(gcf, 'Renderer', 'painters');  % For better vector graphics output

% Position axes to better use the space
pos1 = get(ax1, 'Position');
pos1(2) = 0.75; % Move up
pos1(4) = 0.20; % Make taller
set(ax1, 'Position', pos1);

pos2 = get(ax2, 'Position');
pos2(2) = 0.52; % Move up
pos2(4) = 0.20; % Make taller
set(ax2, 'Position', pos2);

pos3 = get(ax3, 'Position');
pos3(2) = 0.29; % Move up
pos3(4) = 0.20; % Make taller
set(ax3, 'Position', pos3);

% Adjust position of the bottom row panels
set(ax4, 'Position', [0.1300, 0.05, 0.2134, 0.15]);
set(ax5, 'Position', [0.4108, 0.05, 0.2134, 0.15]);
set(ax6, 'Position', [0.6916, 0.05, 0.2134, 0.15]);

% Save the multi-panel plot with high resolution
print('multi_distribution_comparison.png', '-dpng', '-r300');
saveas(gcf, fullfile(savePath, 'multi_distribution_comparison.png'));
end