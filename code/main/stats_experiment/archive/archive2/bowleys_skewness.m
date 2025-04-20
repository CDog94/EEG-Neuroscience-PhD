function sk = bowleys_skewness(data)
% Calculates Bowley's Quartile Skewness
% 
% Parameters:
%   data - Array of data points
%
% Returns:
%   sk - Bowley's Quartile Skewness value

    % Calculate quartiles
    q1 = quantile(data, 0.25);
    q2 = quantile(data, 0.50); % median
    q3 = quantile(data, 0.75);
    
    % Calculate Bowley's Quartile Skewness
    if (q3 - q1) == 0
        sk = 0;
    else
        sk = (q3 - 2*q2 + q1) / (q3 - q1);
    end
    
    % Ensure skewness is in valid range
    sk = min(max(sk, -1), 1);
end