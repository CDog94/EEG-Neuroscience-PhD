%% P-VALUE CORRECTION PIPELINE
% Main orchestration script for training and evaluating p-value correction models
% Author: Cihan
% Date: 2025

clear; close all; clc;

% Add current directory to path (so MATLAB can find all functions)
addpath(pwd);

fprintf('========================================\n');
fprintf('P-VALUE CORRECTION PIPELINE\n');
fprintf('========================================\n\n');

%% 1. LOAD CONFIGURATION
fprintf('Step 1: Loading configuration...\n');
cfg = config_pipeline();


%% 2.5 GENERATE DIAGNOSTIC PLOTS 
if cfg.generate_diagnostics
    fprintf('\nStep 2.5: Generating diagnostic plots...\n');
    generateDiagnosticPlots(cfg);
end

%% 2. PREPARE DATA
fprintf('\nStep 2: Loading and preparing data...\n');
data = load_and_prepare_data(cfg);

%% 3. BUILD MODEL AND PREDICT
fprintf('\nStep 3: Building model and making predictions...\n');
[models, predictions, training_info] = model_builder(data, cfg);

%% 4. EVALUATE MODEL
fprintf('\nStep 4: Evaluating model performance...\n');
results = evaluate_model(predictions, data, training_info, cfg);

%% 5. SUMMARY
fprintf('\n========================================\n');
fprintf('PIPELINE COMPLETED SUCCESSFULLY!\n');
fprintf('========================================\n');
fprintf('Model: %s\n', training_info.model_type);
fprintf('Test R² (critical): %.4f\n', results.r2_critical);
fprintf('Test FPR: %.4f\n', results.fpr);
fprintf('========================================\n\n');
