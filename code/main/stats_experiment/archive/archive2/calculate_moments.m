function moments = calculate_moments(data)
% Calculates the four central moments of a data distribution
% 
% Parameters:
%   data - Array of data points
%
% Returns:
%   moments - Structure containing the four moments:
%     .mean - First moment (mean)
%     .variance - Second moment (variance)
%     .skewness - Third moment (skewness)
%     .kurtosis - Fourth moment (kurtosis)
%     .bowleySkewness - Bowley's Quartile Skewness (for backward compatibility)

    % Calculate first moment - mean
    moments.mean = mean(data);
    
    % Calculate second moment - variance
    moments.variance = var(data);
    
    % Calculate third moment - skewness using MATLAB's built-in function
    moments.skewness = skewness(data);
    
    % Calculate fourth moment - kurtosis using MATLAB's built-in function
    % Note: MATLAB's kurtosis is excess kurtosis (normal = 0)
    moments.kurtosis = kurtosis(data);
    
    % Calculate Bowley's Quartile Skewness for backward compatibility
    q1 = quantile(data, 0.25);
    q2 = quantile(data, 0.50); % median
    q3 = quantile(data, 0.75);
    
    if (q3 - q1) == 0
        moments.bowleySkewness = 0;
    else
        moments.bowleySkewness = (q3 - 2*q2 + q1) / (q3 - q1);
    end
    
    % Ensure Bowley's skewness is in valid range
    moments.bowleySkewness = min(max(moments.bowleySkewness, -1), 1);
end