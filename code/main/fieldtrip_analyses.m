%% PATHS AND SETTING UP FIELDTRIP AND PATHS
clear classes;
clear all;
master_dir = 'C:\Users\CDoga\Documents\Research\PhD\fieldtrip';
main_path = 'C:\Users\CDoga\Documents\Research\PhD\participant_';
results_dir = 'C:\Users\CDoga\Documents\Research\PhD\results';
addpath('C:\Users\CDoga\Documents\Research\fieldtrip-20240214');

ft_defaults;
cd(master_dir);

%% WHAT TYPE OF EXPERIMENT(s) ARE WE RUNNING?
experiment_types = {'onsets-2-8-explicit'};
desired_design_mtxs = {"no-factor"};
type_of_interaction = 'habituation';
start_latency = 0.056;
end_latency = 0.256;

%% SHALL WE APPLY A ROI, IF SO HOW?
region_of_interest =0 ;
roi_applied = 'two-tailed';
weight_roi = 0;
roi_to_apply = 0;

%% GENERATE ERPS AND COMPUTE CONFIDENCE INTERVALS
create_topographic_maps =1;
generate_erps = 1;
weight_erps = 0; % weights based on quartiles
weighting_factor = 0.75; % weights based on quartiles

%% CHOOSE THE TYPE OF ANALYSIS EITHER 'frequency_domain' or 'time_domain'
type_of_analysis = 'time_domain';

if strcmp(type_of_analysis, 'frequency_domain') || strcmp(type_of_analysis, 'frequency_domain_p1')
    disp('RUNNING A FREQUENCY-DOMAIN ANALYSIS');
    run_mua = 1; % run a MUA in the frequnecy domain?
    analyse_spectrogram = 1 ; % analysis on the aggregate power data?
    frequency_level = 'trial-level'; % freq analyses on 'participant-level' or 'trial-level'
    extract_timeseries_values = 0;
    toi = [0.5, 3.05];
    foi_of_interest = [[10, 15]; [20, 30]; [30,45]; [45, 60]; [60, 80]];
    %foi_of_interest = [[45, 60]; [60, 80]];
    analysis = 'load'; % 'load' or 'preprocess'
elseif strcmp(type_of_analysis, 'time_domain') || strcmp(type_of_analysis, 'time_domain_p1')
    disp('RUNNING A TIME-DOMAIN ANALYSIS');
    foi_of_interest = [[-999, -999]];
    toi = [0.056, 0.256];
end


%% OFF TO THE RACES WE GO
 for f = 1:numel(foi_of_interest)
    foi = foi_of_interest(f,:);
    for exp_type = 1:numel(experiment_types)
        for j = 1:numel(desired_design_mtxs)
            experiment_type = experiment_types{exp_type};
            desired_design_mtx = desired_design_mtxs{j};
            %% create the results save path depending on the experiment

            % time of interest
            start_time = num2str(toi(1)*1000);
            end_time = num2str(toi(2)*1000);
            time_name = start_time + "_" + end_time + "ms";

            if strcmp(experiment_type, 'partitions-2-8')
                save_path = strcat(results_dir, '/', type_of_analysis, '/', time_name,'/', type_of_interaction, '/ ', 'partitions', '/', desired_design_mtx);
            elseif contains(experiment_type, 'erps-23-45-67')
                save_path = strcat(results_dir, '/', type_of_analysis,'/', time_name,'/', type_of_interaction, '/ ', 'onsets', '/', desired_design_mtx);
            elseif contains(experiment_type,'onsets-2-8-explicit')
                save_path = strcat(results_dir, '/', type_of_analysis,'/', time_name, '/', 'mean_intercept', '/', desired_design_mtx);
            elseif strcmp(experiment_type, 'partitions_vs_onsets')
                save_path = strcat(results_dir, '/', type_of_analysis,'/', time_name,'/', type_of_interaction, '/ ', 'partitions_vs_onsets', '/', desired_design_mtx);
            elseif strcmp(experiment_types, 'trial-level-2-8')
                save_path = strcat(results_dir, '/', type_of_analysis,'/', time_name,'/',type_of_interaction, '/ ', 'trial_level_2_8', '/', desired_design_mtx);
            elseif strcmp(experiment_types, 'pure-factor-effect')
                save_path = strcat(results_dir, '/', type_of_analysis, '/' , time_name,'/','pure-factor-effect', '/', desired_design_mtx);
            elseif strcmp(experiment_types, 'factor_effect')
                save_path = strcat(results_dir, '/', type_of_analysis, '/ ', time_name,'/','factor-effect', '/', desired_design_mtx);
            elseif strcmp(experiment_types, 'three-way-interaction')
                save_path = strcat(results_dir, '/', type_of_analysis, '/', time_name,'/','three-way-interaction', '/', desired_design_mtx);
            end

            % check if its a frequency experiment
            if strcmp(type_of_analysis, 'frequency_domain')
                start_freq = int2str(foi(1));
                end_freq = int2str(foi(2));
                save_path = save_path + "_" + start_freq + "_" + end_freq;

                if region_of_interest == 1
                    roi = "C:\Users\CDoga\Documents\Research\PhD\fieldtrip\roi\" + "roi_" + "freq_" +  int2str(foi(1)) + "_" + int2str(foi(2));
                    load(roi)
                end
            end

            % check if the folder exists, else make it
            if ~exist(save_path, 'dir')
                mkdir(save_path);
            end

            %% Are we looking at onsets 2-8 or partitions
            % set up the experiment as needed
            if contains(type_of_analysis, 'time_domain')
                if strcmp(experiment_type, 'onsets-2-8-explicit')

                    if contains(desired_design_mtx, 'eye')
                        data_file = 'time_domain_eye_confound_onsets_2_3_4_5_6_7_8_grand-average.mat';
                    else
                        data_file = 'time_domain_mean_intercept_onsets_2_3_4_5_6_7_8_grand-average.mat';
                    end


                    regressor = 'ft_statfun_depsamplesT';
                    type_of_effect = 'null';
                    regression_type = desired_design_mtx;
                    n_participants = 40;
                    start_latency = 0.056;
                    end_latency = 0.256;

                    partition.is_partition = 0;
                    partition.partition_number = 0;

                    [data, participant_order_1] = load_postprocessed_data(main_path, n_participants, ...
                        data_file, partition, type_of_analysis);
                    n_part = numel(data);
                    %[design_matrix, data] =  create_design_matrix_partitions(participant_order_1, data, ...
                    %            regression_type, 0, type_of_effect);
                    design_matrix =  [1:n_part 1:n_part; ones(1,n_part) 2*ones(1,n_part)];

                    if contains(desired_design_mtx, 'eye')
                        data = apply_dummy_coordinates_to_eye_electrodes(data);
                    end

                    all_data{1} = data;
                    all_designs{1} = design_matrix;

                elseif strcmp(experiment_type, 'pure-factor-effect')
                    data_file = 'time_domain_mean_intercept_onsets_2_3_4_5_6_7_8_grand-average.mat';
                    regressor = 'ft_statfun_indepsamplesregrT';
                    type_of_effect = 'null';
                    regression_type = desired_design_mtx;
                    n_participants = 40;
                    start_latency = 0.056;
                    end_latency = 0.256;

                    partition.is_partition = 0;
                    partition.partition_number = 999;


                    [data, participant_order_1] = load_postprocessed_data(main_path, n_participants, ...
                        data_file, partition, type_of_analysis);
                    n_part = numel(data);

                    [design_matrix, data] =  create_design_matrix_partitions(participant_order_1, data, ...
                        regression_type, 0, type_of_effect);

                    if region_of_interest == 1
                        if strcmp(experiment_type, 'partitions-2-8') || strcmp(experiment_type, 'pure-factor-effect')
                            if strcmp(roi_applied, 'one-tailed')
                                load('E:/PhD/fieldtrip/roi/one_tailed_roi_28.mat');
                            elseif strcmp(roi_applied, 'two-tailed')
                               load('C:\Users\CDoga\Documents\Research\PhD\fieldtrip\roi\two_tailed_roi_28.mat');
                            end
                        end
                        data = create_hacked_roi(data, roi, weight_roi);
                    end

                    plot(design_matrix(1:numel(design_matrix)), 'color', 'r', 'LineWidth', 3.5);
                    hold on;
                    xlabel('Participants', 'FontSize',11);
                    ylabel('Value in Regressor', 'FontSize',11);

                    set(gcf,'Position',[100 100 500 500])
                    save_dir = strcat(save_path, '/', 'design_matrix.png');
                    exportgraphics(gcf,save_dir,'Resolution',500);
                    close;

                    all_data{1} = data;
                    all_designs{1} = design_matrix;

                elseif strcmp(experiment_type, 'trial-level-2-8')
                    partition1.is_partition = 1; % partition 1
                    partition1.partition_number = 999;
                    n_participants = 40;
                    regression_type = desired_design_mtx;
                    type_of_effect = 'null';
                    regressor = 'ft_statfun_indepsamplesregrT';
                    k_trials = 100;
                    data_file = 'time_domain_mean_intercept_onsets_2_3_4_5_6_7_8_trial-level.mat';

                    % create the data
                    [data1, participant_order_1] = load_postprocessed_data(main_path, n_participants, ...
                        data_file, partition1);

                    if region_of_interest == 1
                        load('D:/PhD/fieldtrip/roi/two_tailed_roi_28.mat');
                    end

                    all_data = create_data_with_increasing_number_of_trials(data1, k_trials, roi);
                    extract_timeseries_through_time(all_data)
                    [design1, new_participants1] = create_design_matrix_partitions(participant_order_1, data1, ...
                        regression_type, 1, type_of_effect);
                    [all_designs, all_data] = update_design_matrix(all_data, design1);



                elseif strcmp(experiment_type, 'partitions-2-8')
                    partition1.is_partition = 1; % partition 1
                    partition1.partition_number = 1;
                    partition2.is_partition = 1; % partition 2
                    partition2.partition_number = 2;
                    partition3.is_partition = 1; % partition 3
                    partition3.partition_number = 3;
                    regressor = 'ft_statfun_indepsamplesregrT';
                    regression_type = desired_design_mtx;
                    n_participants = 40;

                    if strcmp(type_of_analysis,'time_domain')
                        data_file = 'time_domain_partitions_partitioned_onsets_2_3_4_5_6_7_8_grand-average.mat';


                        [data1, participant_order_1] = load_postprocessed_data(main_path, n_participants, ...
                            data_file, partition1, type_of_analysis);
                        [data2, participant_order_2] = load_postprocessed_data(main_path, n_participants, ...
                            data_file, partition2, type_of_analysis);
                        [data3, participant_order_3] = load_postprocessed_data(main_path, n_participants, ...
                            data_file, partition3, type_of_analysis);

                        partition = 1;
                        [design1, new_participants1] = create_design_matrix_partitions(participant_order_1, data1, ...
                            regression_type, partition, type_of_interaction);
                        partition = 2;
                        [design2, new_participants2] = create_design_matrix_partitions(participant_order_2, data2, ...
                            regression_type, partition, type_of_interaction);
                        partition = 3;
                        [design3, new_participants3] = create_design_matrix_partitions(participant_order_3, data3, ...
                            regression_type, partition, type_of_interaction);


                        data = [new_participants1, new_participants2, new_participants3];
                        design_matrix = [design1;design2;design3];
                        n_part = numel(data);
                        n_part_per_desgin = numel(design1);

                        if strcmp(desired_design_mtx, 'no-factor')
                            design1(1:numel(design1)) = 2.72;
                            design2(1:numel(design1)) = 1.65;
                            design3(1:numel(design1)) = 1.00;
                            design_matrix = [design1, design2, design3];
                        end

                        design_matrix = design_matrix - mean(design_matrix);
                        save_desgin_matrix(design_matrix, n_part_per_desgin, save_path, type_of_interaction)

                        if region_of_interest == 1
                            if strcmp(experiment_type, 'partitions-2-8')
                                if strcmp(roi_applied, 'one-tailed')
                                    load('E:/PhD/fieldtrip/roi/one_tailed_roi_28.mat');
                                elseif strcmp(roi_applied, 'two-tailed')
                                    load('C:\Users\CDoga\Documents\Research\PhD\fieldtrip\roi\two_tailed_roi_28.mat');
                                end
                            end
                            data = create_hacked_roi(data, roi, weight_roi);
                        end

                        all_data{1} = data;
                        all_designs{1} = design_matrix;
                    elseif strcmp(type_of_analysis, 'time_domain_p1')
                        data_file = 'time_domain_partitions_partitioned_onsets_2_3_4_5_6_7_8_grand-average.mat';
                        partition1.is_partition = 1; % partition 1
                        partition1.partition_number = 1;
                        [data1, participant_order_1] = load_postprocessed_data(main_path, n_participants, ...
                            data_file, partition1, type_of_analysis);
                        partition = 1;
                        [design1, new_participants1] = create_design_matrix_partitions(participant_order_1, data1, regression_type, partition, type_of_interaction);
                        design_matrix = design1;
                        design_matrix = design_matrix - mean(design_matrix);
                        data = data1;
                        %save_desgin_matrix(design_matrix, n_part_per_desgin, save_path, 'habituation')

                        plot(design_matrix(1:numel(design_matrix)), 'color', 'r', 'LineWidth', 3.5);
                        hold on;
                        xlabel('Participants', 'FontSize',14);
                        ylabel('Factor Score', 'FontSize',14);
                        set(gcf,'Position',[100 100 500 500])
                        save_dir = strcat(save_path, '/', 'design_matrix.png');
                        exportgraphics(gcf,save_dir,'Resolution',500);
                        close;


                        if region_of_interest == 1
                            if strcmp(experiment_type, 'partitions-2-8')
                                if strcmp(roi_applied, 'one-tailed')
                                    load('E:/PhD/fieldtrip/roi/one_tailed_roi_28.mat');
                                elseif strcmp(roi_applied, 'two-tailed')
                                    load('C:\Users\CDoga\Documents\Research\PhD\fieldtrip\roi\two_tailed_roi_28.mat');
                                end
                            end
                            data = create_hacked_roi(data, roi, weight_roi);
                        end

                        all_designs{1} = design_matrix;
                        all_data{1} = data;

                    end
                elseif strcmp(experiment_type, 'erps-23-45-67')
                    data_file23 = 'time_domain_mean_intercept_onsets_2_3_grand-average.mat';
                    data_file45 = 'time_domain_mean_intercept_onsets_4_5_grand-average.mat';
                    data_file67 = 'time_domain_mean_intercept_onsets_6_7_grand-average.mat';
                    regressor = 'ft_statfun_indepsamplesregrT';
                    type_of_effect = type_of_interaction;

                    n_participants = 40;
                    partitions.is_partition = 0;
                    partitions.partition_number = 0;
                    regression_type = desired_design_mtx;

                    [data1, participant_order_1] = load_postprocessed_data(main_path, n_participants, ...
                        data_file23, partitions, type_of_analysis);
                    [data2, participant_order_2] = load_postprocessed_data(main_path, n_participants, ...
                        data_file45, partitions, type_of_analysis);
                    [data3, participant_order_3] = load_postprocessed_data(main_path, n_participants, ...
                        data_file67, partitions, type_of_analysis);


                    partition = 1;
                    [design1, new_participants1] = create_design_matrix_partitions(participant_order_1, data1, ...
                        regression_type, partition, type_of_interaction);
                    partition = 2;
                    [design2, new_participants2] = create_design_matrix_partitions(participant_order_2, data2, ...
                        regression_type, partition, type_of_interaction);
                    partition = 3;
                    [design3, new_participants3] = create_design_matrix_partitions(participant_order_3, data3, ...
                        regression_type, partition, type_of_interaction);

                    data = [new_participants1, new_participants2, new_participants3];
                    design_matrix = [design1;design2;design3];
                    n_part = numel(data);
                    n_part_per_desgin = numel(design1);

                    if strcmp(desired_design_mtx, 'no-factor')
                        design_matrix_a(1:numel(design1)) = 1.00;
                        design_matrix_b(1:numel(design1)) = 1.65;
                        design_matrix_c(1:numel(design1)) = 2.72;
                        design_matrix = [design_matrix_a, design_matrix_b, design_matrix_c];
                        min_x = min(design_matrix);
                        design_matrix = design_matrix - min_x;
                    end

                    if strcmp(desired_design_mtx, 'headache-scores-entire-median-split') ...
                            || strcmp(desired_design_mtx, 'discomfort-scores-entire-median-split')
                        newdesign1 = design1 * 1.00;
                        newdesign2 = design2 * 1.65;
                        newdesign3 = design3 * 2.72;
                        design_matrix = [newdesign1, newdesign2, newdesign3];
                        min_x = min(design_matrix);
                        design_matrix = design_matrix - min_x;
                    end

                    design_matrix = design_matrix - mean(design_matrix);
                    save_desgin_matrix(design_matrix, n_part_per_desgin, save_path, type_of_interaction)

                    if region_of_interest == 1
                        if strcmp(experiment_type, 'erps-23-45-67') || strcmp(experiment_type,  'erps-23-45-67-no-factor')
                            if strcmp(roi_applied, 'one-tailed')
                                load('E:/PhD/fieldtrip/roi/one_tailed_roi_28.mat');
                            elseif strcmp(roi_applied, 'two-tailed')
                                load('C:\Users\CDoga\Documents\Research\PhD\fieldtrip\roi\two_tailed_roi_28.mat');
                            end
                        end
                        data = create_hacked_roi(data, roi, weight_roi);
                    end

                    all_data{1} = data;
                    all_designs{1} = design_matrix;
                elseif strcmp(experiment_type, 'three-way-interaction')
                    n_participants = 40;
                    partition1.is_partition = 1;
                    partition1.partition_number = 1;
                    partition2.is_partition = 1;
                    partition2.partition_number = 2;
                    partition3.is_partition = 1;
                    partition3.partition_number = 3;

                    onsets_2_3 = 'time_domain_partitions_partitioned_onsets_2_3_grand-average.mat';
                    onsets_4_5 = 'time_domain_partitions_partitioned_onsets_4_5_grand-average.mat';
                    onsets_6_7 = 'time_domain_partitions_partitioned_onsets_6_7_grand-average.mat';

                    regressor = 'ft_statfun_indepsamplesregrT';
                    regression_type = desired_design_mtx;
                    type_of_effect = type_of_interaction;

                    [p1_23, po_p1_23] = load_postprocessed_data(main_path, n_participants, ...
                        onsets_2_3, partition1, type_of_analysis);
                    [p2_23, po_p2_23] = load_postprocessed_data(main_path, n_participants, ...
                        onsets_2_3, partition2, type_of_analysis);
                    [p3_23, po_p3_23] = load_postprocessed_data(main_path, n_participants, ...
                        onsets_2_3, partition3, type_of_analysis);


                    partition = 1;
                    [design_p1_23, p1_23] = create_design_matrix_partitions(po_p1_23, p1_23, ...
                        regression_type, partition, type_of_effect);
                    partition = 2;
                    [design_p2_23, p2_23] = create_design_matrix_partitions(po_p2_23, p2_23, ...
                        regression_type, partition, type_of_effect);
                    partition = 3;
                    [design_p3_23, p3_23] = create_design_matrix_partitions(po_p3_23, p3_23, ...
                        regression_type, partition, type_of_effect);

                    design_p1_23 = design_p1_23 * 0;
                    design_p2_23 = design_p2_23 * 0;
                    design_p3_23 = design_p3_23 * 0;

                    [p1_45, po_p1_45] = load_postprocessed_data(main_path, n_participants, ...
                        onsets_4_5, partition1, type_of_analysis);
                    [p2_45, po_p2_45] = load_postprocessed_data(main_path, n_participants, ...
                        onsets_4_5, partition2, type_of_analysis);
                    [p3_45, po_p3_45] = load_postprocessed_data(main_path, n_participants, ...
                        onsets_4_5, partition3, type_of_analysis);

                    partition = 1;
                    [design_p1_45, p1_45] = create_design_matrix_partitions(po_p1_45, p1_45, ...
                        regression_type, partition, type_of_effect);
                    partition = 2;
                    [design_p2_45, p2_45] = create_design_matrix_partitions(po_p2_45, p2_45, ...
                        regression_type, partition, type_of_effect);
                    partition = 3;
                    [design_p3_45, p3_45] = create_design_matrix_partitions(po_p3_45, p3_45, ...
                        regression_type, partition, type_of_effect);

                    design_p1_45 = design_p1_45 * 1.65;
                    design_p2_45 = design_p2_45 * 1.65;
                    design_p3_45 = design_p3_45 * 1.65;


                    [p1_67, po_p1_67] = load_postprocessed_data(main_path, n_participants, ...
                        onsets_6_7, partition1, type_of_analysis);
                    [p2_67, po_p2_67] = load_postprocessed_data(main_path, n_participants, ...
                        onsets_6_7, partition2, type_of_analysis);
                    [p3_67, po_p3_67] = load_postprocessed_data(main_path, n_participants, ...
                        onsets_6_7, partition3, type_of_analysis);

                    partition = 1;
                    [design_p1_67, p1_67] = create_design_matrix_partitions(po_p1_67, p1_67, ...
                        regression_type, partition, type_of_effect);
                    partition = 2;
                    [design_p2_67, p2_67] = create_design_matrix_partitions(po_p2_67, p2_67, ...
                        regression_type, partition, type_of_effect);
                    partition = 3;
                    [design_p3_67, p3_67] = create_design_matrix_partitions(po_p3_67, p3_67, ...
                        regression_type, partition, type_of_effect);

                    design_p1_67 = design_p1_67 * 2.72;
                    design_p2_67 = design_p2_67 * 2.72;
                    design_p3_67 = design_p3_67 * 2.72;


                    design_matrix = [
                        design_p1_23; design_p1_45; design_p1_67; ...
                        design_p2_23; design_p2_45; design_p2_67; ...
                        design_p3_23; design_p3_45; design_p3_67; ...
                        ];

                    design_matrix = design_matrix - mean(design_matrix);

                    
                    onsets_2_3 = [design_p1_23; design_p2_23; design_p3_23];
                    onsets_4_5 = [design_p1_45; design_p2_45; design_p3_45];
                    onsets_6_7 = [design_p1_67; design_p2_67; design_p3_67];

                    plot(onsets_2_3, "LineWidth",3.5)
                    hold on;
                    plot(onsets_4_5, "LineWidth",3.5)
                    plot(onsets_6_7, "LineWidth",3.5)


                    xline(numel(design_p1_23), '--r', {'Partition 1'}, 'LineWidth', 3.5, 'LabelHorizontalAlignment', 'left')
                    xline(numel(design_p1_23)*2, '--r', {'Partition 2'}, 'LineWidth', 3.5, 'LabelHorizontalAlignment', 'left')       
                    xline(numel(design_p1_23)*3, '--r', {'Partition 3'}, 'LineWidth', 3.5, 'LabelHorizontalAlignment', 'left')

                    legend({'Onsets 2,3', 'Onsets 4,5', 'Onsets 6,7'},'Location','northwest')
                    xlabel('Participants'); 
                    ylabel('Value in Regressor');

                    if strcmp(desired_design_mtx, 'visual_stress')
                        label_factor = "Visual Stress";
                    elseif contains(desired_design_mtx, 'headache')
                        label_factor = 'Headache';
                    elseif contains(desired_design_mtx, 'discomfort')
                        label_factor = 'Discomfort';
                    end

                    plot_title = label_factor + " x Habituation for Partitions x Sensitization for Onsets";

                    title(plot_title);
                    set(gcf,'Position',[100 100 1000 750])
                    path = strcat(save_path, '/', 'design_matrix.png');
                    exportgraphics(gcf,path,'Resolution',500);
                    close;


                    data = [
                        p1_23, p1_45, p1_67, ...
                        p2_23, p2_45, p2_67, ...
                        p3_23, p3_45, p3_67
                        ];

                    if region_of_interest == 1
                        if strcmp(experiment_type, 'partitions-2-8') || strcmp(experiment_type, 'pure-factor-effect') ...
                                || strcmp(experiment_type, 'three-way-interaction')
                            if strcmp(roi_applied, 'one-tailed')
                                load('E:/PhD/fieldtrip/roi/one_tailed_roi_28.mat');
                            elseif strcmp(roi_applied, 'two-tailed')
                               load('C:\Users\CDoga\Documents\Research\PhD\fieldtrip\roi\two_tailed_roi_28.mat');
                            end
                        end
                        data = create_hacked_roi(data, roi, weight_roi);
                    end

                    all_data = {};
                    all_data{1} = data;
                    all_designs{1} = design_matrix;
                end


            elseif contains(type_of_analysis, 'frequency_domain')
                if strcmp(experiment_type, 'erps-23-45-67') && strcmp(type_of_analysis, 'frequency_domain_p1') 
                    n_participants = 40;
                    regression_type = desired_design_mtx;
                    regressor = 'ft_statfun_indepsamplesregrT';
                    partitions.is_partition = 1;
                    partitions.partition_number = 1;
                    type_of_effect = type_of_interaction;
    
                        if strcmp(analysis, 'load')
                            data_file23 = 'frequency_domain_mean_intercept_onsets_2_3_grand-average.mat';
                            data_file45 = 'frequency_domain_mean_intercept_onsets_4_5_grand-average.mat';
                            data_file67 = 'frequency_domain_mean_intercept_onsets_6_7_grand-average.mat';
                        elseif strcmp(analysis, 'preprocess')
                            data_file23 = 'frequency_domain_mean_intercept_onsets_2_3_trial-level.mat';
                            data_file45 = 'frequency_domain_mean_intercept_onsets_4_5_trial-level.mat';
                            data_file67 = 'frequency_domain_mean_intercept_onsets_6_7_trial-level.mat';
                        end
                    

                    end 
                if strcmp(experiment_type, 'erps-23-45-67')
                    n_participants = 40;
                    regression_type = desired_design_mtx;
                    regressor = 'ft_statfun_indepsamplesregrT';
                    partitions.is_partition = 0;
                    partitions.partition_number = 0;
                    type_of_effect = type_of_interaction;

                    if strcmp(analysis, 'load')
                        data_file23 = 'frequency_domain_mean_intercept_onsets_2_3_grand-average.mat';
                        data_file45 = 'frequency_domain_mean_intercept_onsets_4_5_grand-average.mat';
                        data_file67 = 'frequency_domain_mean_intercept_onsets_6_7_grand-average.mat';
                    elseif strcmp(analysis, 'preprocess')
                        data_file23 = 'frequency_domain_mean_intercept_onsets_2_3_trial-level.mat';
                        data_file45 = 'frequency_domain_mean_intercept_onsets_4_5_trial-level.mat';
                        data_file67 = 'frequency_domain_mean_intercept_onsets_6_7_trial-level.mat';
                    end

                    [data1, participant_order1] = load_postprocessed_data(main_path, n_participants, ...
                        data_file23, partitions, type_of_analysis);
                   [data2, participant_order2] = load_postprocessed_data(main_path, n_participants, ...
                        data_file45, partitions, type_of_analysis);
                    [data3, participant_order3] = load_postprocessed_data(main_path, n_participants, ...
                        data_file67, partitions, type_of_analysis);

                    partition.is_partition = 0;
                    partition.partition_number = 0;

                    p1_freq = to_frequency_data(main_path, partition, ...
                        analysis, frequency_level, foi, ...
                        'erp23_', data_file23, n_participants, toi);

                    p2_freq = to_frequency_data(main_path, partition, ...
                        analysis, frequency_level, foi, ...
                        'erp45_', data_file45, n_participants, toi);

                    p3_freq = to_frequency_data(main_path, partition, ...
                        analysis, frequency_level, foi, ...
                        'erp67_', data_file67, n_participants, toi);
                    
                    if strcmp(analysis, 'preprocess')
                        continue;
                    end

                    [design1, new_participants1] = create_design_matrix_partitions(participant_order1, p1_freq, ...
                        regression_type, 1, type_of_effect);
                    [design2, new_participants2] = create_design_matrix_partitions(participant_order2, p2_freq, ...
                        regression_type, 2, type_of_effect);
                    [design3, new_participants3] = create_design_matrix_partitions(participant_order3, p3_freq, ...
                        regression_type, 3, type_of_effect);

                    n_part_per_desgin = numel(design1);

                    vector_of_data = [new_participants1, new_participants2, new_participants3];

                    all_data = {};
                    for i = 1:numel(vector_of_data)
                        participant = vector_of_data(i);
                        all_data{1}{i} = participant{1}.pgi;
                    end

                    if region_of_interest == 1
                        all_data = create_hacked_roi_freq(all_data, roi);
                    end

                    if strcmp(desired_design_mtx, 'no-factor')
                        design1(1:numel(design1)) = 1.00;
                        design2(1:numel(design1)) = 1.65;
                        design3(1:numel(design1)) = 2.72;
                        design_matrix = [design1; design2; design3];
                    else
                        design_matrix = [design1; design2; design3];
                    end

                    design_matrix = design_matrix - mean(design_matrix);
                    all_designs{1} = design_matrix;

                   
                    save_desgin_matrix(design_matrix, 30, save_path, type_of_interaction)

                elseif strcmp(experiment_type, 'factor_effect')
                    n_participants = 40;
                    regression_type = desired_design_mtx;
                    type_of_effect = 'null';
                    regressor = 'ft_statfun_indepsamplesregrT';

                    if strcmp(analysis, 'load')
                        data_file = 'frequency_domain_mean_intercept_onsets_2_3_4_5_6_7_8_grand-average.mat';
                    elseif strcmp(analysis, 'preprocess')
                        data_file = 'frequency_domain_mean_intercept_onsets_2_3_4_5_6_7_8_trial-level.mat';
                    end

                    partition.is_partition = 0;
                    partition.partition_number = 0;

                    [data, participant_order_1] = load_postprocessed_data(main_path, n_participants, ...
                       data_file, partition, type_of_analysis);

                         freq = to_frequency_data(main_path, partition, ...
                        analysis, frequency_level, foi, ...
                        'mean_intercept', data_file, n_participants, toi);
                    n_part = numel(data);

                    [design1, new_participants1] = create_design_matrix_partitions(participant_order_1, freq, ...
                        regression_type, 1, type_of_effect);

                    vector_of_data = [new_participants1];

                    all_data = {};
                    for i = 1:numel(vector_of_data)
                        participant = vector_of_data(i);
                        all_data{1}{i} = participant{1}.pgi;
                    end

                    n_part_per_desgin = numel(design1);
                    design_matrix = design1 - mean(design1);
                    %save_desgin_matrix(design1, n_part_per_desgin, save_path, 'habituation')
                    all_designs{1} = design1;

                elseif strcmp(experiment_type, 'partitions-2-8')
                    n_participants = 40;
                    regression_type = desired_design_mtx;
                    regressor = 'ft_statfun_indepsamplesregrT';
                    type_of_effect = type_of_interaction;

                    partition1.is_partition = 1;
                    partition1.partition_number = 1;
                    partition2.is_partition = 1;
                    partition2.partition_number = 2;
                    partition3.is_partition = 1;
                    partition3.partition_number = 3;

                    if strcmp(analysis, 'load')
                        data_file = 'frequency_domain_partitions_partitioned_onsets_2_3_4_5_6_7_8_grand-average.mat';
                    elseif strcmp(analysis, 'preprocess')
                        data_file = 'frequency_domain_partitions_partitioned_onsets_2_3_4_5_6_7_8_trial-level.mat';
                    end

                  [data1, participant_order1] = load_postprocessed_data(main_path, n_participants, ...
                       data_file, partition1, type_of_analysis);
                   [data2, participant_order2] = load_postprocessed_data(main_path, n_participants, ...
                        data_file, partition2, type_of_analysis);
                   [data3, participant_order3] = load_postprocessed_data(main_path, n_participants, ...
                        data_file, partition3, type_of_analysis);


                    disp('--processing-- Hz')
                    disp(foi)

                    p1_freq = to_frequency_data(main_path, partition1, ...
                        analysis, frequency_level, foi, ...
                        'partition_', data_file, n_participants, toi);

                    p2_freq = to_frequency_data(main_path, partition2, ...
                     analysis, frequency_level, foi, ...
                         'partition_', data_file, n_participants, toi);

                    p3_freq = to_frequency_data(main_path, partition3, ...
                        analysis, frequency_level, foi, ...
                        'partition_', data_file, n_participants, toi);

                    if strcmp(analysis, 'preprocess')
                        continue;
                    end

                    [design1, new_participants1] = create_design_matrix_partitions(participant_order1, p1_freq, ...
                        regression_type, 1, type_of_effect);
                    [design2, new_participants2] = create_design_matrix_partitions(participant_order2, p2_freq, ...
                        regression_type, 2, type_of_effect);
                    [design3, new_participants3] = create_design_matrix_partitions(participant_order3, p3_freq, ...
                        regression_type, 3, type_of_effect);

                    vector_of_data = [new_participants1, new_participants2, new_participants3];

                    all_data = {};
                    for i = 1:numel(vector_of_data)
                        participant = vector_of_data(i);
                        all_data{1}{i} = participant{1}.pgi;
                    end

                    if region_of_interest == 1
                        all_data = create_hacked_roi_freq(all_data, roi);
                    end

                    n_part_per_desgin = numel(design1);



                    if strcmp(desired_design_mtx, 'no-factor')
                        design1(1:numel(design1)) = 2.72;
                        design2(1:numel(design1)) = 1.65;
                        design3(1:numel(design1)) = 1.00;
                        design_matrix = [design1; design2; design3];
                    else
                        design_matrix = [design1; design2; design3];
                    end

                    design_matrix = design_matrix - mean(design_matrix);
                    save_desgin_matrix(design_matrix, n_part_per_desgin, save_path, 'habituation')
                    all_designs{1} = design_matrix;

                elseif strcmp(experiment_type, 'onsets-2-8-explicit')
                    n_participants = 40;
                    regression_type = desired_design_mtx;
                    type_of_effect = 'null';
                    regressor = 'ft_statfun_depsamplesT';

                    if strcmp(analysis, 'load')
                        data_file = 'frequency_domain_mean_intercept_onsets_2_3_4_5_6_7_8_grand-average.mat';
                    elseif strcmp(analysis, 'preprocess')
                        data_file = 'frequency_domain_mean_intercept_onsets_2_3_4_5_6_7_8_trial-level.mat';
                    end

                    partition.is_partition = 0;
                    partition.partition_number = 0;

                    disp('--processing-- Hz')
                    disp(foi);
                    
                    freq = to_frequency_data(main_path, partition, ...
                        analysis, frequency_level, foi, ...
                        'mean_intercept', data_file, n_participants, toi);

                    if strcmp(analysis, 'preprocess')
                        continue;
                    end

                    n_part = numel(freq);

                    null_data = set_values_to_zero_freq(freq);

                    if strcmp(desired_design_mtx, 'aggregated_average')
                        create_agg_average(freq);
                        return;
                    end
                        
                    for i = 1:numel(freq)
                        participant = freq(i);
                        freq{i} = participant{1}.pgi;

                        participant = null_data(i);
                        null_data{i} = participant{1}.pgi;
                    end

                    all_data{1} = freq;
                    all_designs{1} = [1:n_part 1:n_part; ones(1,n_part) 2*ones(1,n_part)];
                
                elseif strcmp(experiment_type, 'three-way-interaction')
                    n_participants = 40;
                    partition1.is_partition = 1;
                    partition1.partition_number = 1;
                    partition2.is_partition = 1;
                    partition2.partition_number = 2;
                    partition3.is_partition = 1;
                    partition3.partition_number = 3;
                    regressor = 'ft_statfun_indepsamplesregrT';
                    regression_type = desired_design_mtx;
                    type_of_effect = type_of_interaction;

                    if strcmp(analysis, 'load')
                        onsets_2_3 = 'frequency_domain_partitions_partitioned_onsets_2_3_grand-average.mat';
                        onsets_4_5 = 'frequency_domain_partitions_partitioned_onsets_4_5_grand-average.mat';
                        onsets_6_7 = 'frequency_domain_partitions_partitioned_onsets_6_7_grand-average.mat';
                    elseif strcmp(analysis, 'preprocess')
                        onsets_2_3 = 'frequency_domain_partitions_partitioned_onsets_2_3_trial-level.mat';
                        onsets_4_5 = 'frequency_domain_partitions_partitioned_onsets_4_5_trial-level.mat';
                        onsets_6_7 = 'frequency_domain_partitions_partitioned_onsets_6_7_trial-level.mat';
                    end

                        % onsets 2,3 for p1,p2,p3

                    [p1_23, po_p1_23] = load_postprocessed_data(main_path, n_participants, ...
                        onsets_2_3, partition1, type_of_analysis);
                    [p2_23, po_p2_23] = load_postprocessed_data(main_path, n_participants, ...
                        onsets_2_3, partition2, type_of_analysis);
                    [p3_23, po_p3_23] = load_postprocessed_data(main_path, n_participants, ...
                        onsets_2_3, partition3, type_of_analysis);

                    p1_23 = to_frequency_data(p1_23, main_path, 1, ...
                        po_p1_23, analysis, frequency_level, foi, ...
                        'p123_', toi);

                    p2_23 = to_frequency_data(p2_23, main_path, 2, ...
                        po_p2_23, analysis, frequency_level, foi, ...
                        'p223_', toi);

                    p3_23 = to_frequency_data(p3_23, main_path, 3, ...
                        po_p3_23, analysis, frequency_level, foi, ...
                        'p323_', toi);

                    partition = 1;
                    [design_p1_23, p1_23] = create_design_matrix_partitions(po_p1_23, p1_23, ...
                        regression_type, partition, type_of_effect);
                    partition = 2;
                    [design_p2_23, p2_23] = create_design_matrix_partitions(po_p2_23, p2_23, ...
                        regression_type, partition, type_of_effect);
                    partition = 3;
                    [design_p3_23, p3_23] = create_design_matrix_partitions(po_p3_23, p3_23, ...
                        regression_type, partition, type_of_effect);

                    design_p1_23 = design_p1_23 * 0;
                    design_p2_23 = design_p2_23 * 0;
                    design_p3_23 = design_p3_23 * 0;

                    % onsets 4,5 for p1,p2,p3
                    [p1_45, po_p1_45] = load_postprocessed_data(main_path, n_participants, ...
                        onsets_4_5, partition1, type_of_analysis);
                    [p2_45, po_p2_45] = load_postprocessed_data(main_path, n_participants, ...
                        onsets_4_5, partition2, type_of_analysis);
                    [p3_45, po_p3_45] = load_postprocessed_data(main_path, n_participants, ...
                        onsets_4_5, partition3, type_of_analysis);

                    p1_45 = to_frequency_data(p1_45, main_path, 1, ...
                        po_p1_45, analysis, frequency_level, foi, ...
                        'p145', toi);

                    p2_45 = to_frequency_data(p2_45, main_path, 2, ...
                        po_p2_45, analysis, frequency_level, foi, ...
                        'p245_', toi);

                    p3_45 = to_frequency_data(p3_45, main_path, 3, ...
                        po_p3_45, analysis, frequency_level, foi, ...
                        'p345_', toi);

                    partition = 1;
                    [design_p1_45, p1_45] = create_design_matrix_partitions(po_p1_45, p1_45, ...
                        regression_type, partition, type_of_effect);
                    partition = 2;
                    [design_p2_45, p2_45] = create_design_matrix_partitions(po_p2_45, p2_45, ...
                        regression_type, partition, type_of_effect);
                    partition = 3;
                    [design_p3_45, p3_45] = create_design_matrix_partitions(po_p3_45, p3_45, ...
                        regression_type, partition, type_of_effect);

                    design_p1_45 = design_p1_45 * 1.65;
                    design_p2_45 = design_p2_45 * 1.65;
                    design_p3_45 = design_p3_45 * 1.65;

                    % onsets 6,7 for p1,p2,p3
                    [p1_67, po_p1_67] = load_postprocessed_data(main_path, n_participants, ...
                        onsets_6_7, partition1, type_of_analysis);
                    [p2_67, po_p2_67] = load_postprocessed_data(main_path, n_participants, ...
                        onsets_6_7, partition2, type_of_analysis);
                    [p3_67, po_p3_67] = load_postprocessed_data(main_path, n_participants, ...
                        onsets_6_7, partition3, type_of_analysis);

                    p1_67 = to_frequency_data(p1_67, main_path, 1, ...
                        po_p1_45, analysis, frequency_level, foi, ...
                        'p167', toi);

                    p2_67 = to_frequency_data(p2_67, main_path, 2, ...
                        po_p2_45, analysis, frequency_level, foi, ...
                        'p267_', toi);

                    p3_67 = to_frequency_data(p3_67, main_path, 3, ...
                        po_p3_45, analysis, frequency_level, foi, ...
                        'p367_', toi);

                    partition = 1;
                    [design_p1_67, p1_67] = create_design_matrix_partitions(po_p1_67, p1_67, ...
                        regression_type, partition, type_of_effect);
                    partition = 2;
                    [design_p2_67, p2_67] = create_design_matrix_partitions(po_p2_67, p2_67, ...
                        regression_type, partition, type_of_effect);
                    partition = 3;
                    [design_p3_67, p3_67] = create_design_matrix_partitions(po_p3_67, p3_67, ...
                        regression_type, partition, type_of_effect);

                    design_p1_67 = design_p1_67 * 2.72;
                    design_p2_67 = design_p2_67 * 2.72;
                    design_p3_67 = design_p3_67 * 2.72;


                    design_matrix = [
                        design_p1_23; design_p1_45; design_p1_67; ...
                        design_p2_23; design_p2_45; design_p2_67; ...
                        design_p3_23; design_p3_45; design_p3_67; ...
                        ];

                    design_matrix = design_matrix - mean(design_matrix);

                    
                    onsets_2_3 = [design_p1_23; design_p2_23; design_p3_23];
                    onsets_4_5 = [design_p1_45; design_p2_45; design_p3_45];
                    onsets_6_7 = [design_p1_67; design_p2_67; design_p3_67];

                    plot(onsets_2_3, "LineWidth",3.5)
                    hold on;
                    plot(onsets_4_5, "LineWidth",3.5)
                    plot(onsets_6_7, "LineWidth",3.5)


                    xline(numel(design_p1_23), '--r', {'Partition 1'}, 'LineWidth', plot_line_width)
                    xline(numel(design_p1_23)*2, '--r', {'Partition 2'}, 'LineWidth', plot_line_width)       
                    xline(numel(design_p1_23)*3, '--r', {'Partition 3'}, 'LineWidth', plot_line_width)

                    legend({'Onsets 2,3', 'Onsets 4,5', 'Onsets 6,7'},'Location','northwest')
                    xlabel('Participants');
                    ylabel('3-way Interaction');

                    if strcmp(desired_design_mtx, 'visual_stress')
                        label_factor = "Visual Stress";
                    elseif contains(desired_design_mtx, 'headache')
                        label_factor = 'Headache';
                    elseif contains(desired_design_mtx, 'discomfort')
                        label_factor = 'Discomfort';
                    end

                    plot_title = label_factor + " x Habituation for Partitions x Sensitization for Onsets";

                    title(plot_title);
                    set(gcf,'Position',[100 100 1000 750])
                    path = strcat(save_path, '/', 'design_matrix.png');
                    exportgraphics(gcf,path,'Resolution',500);
                    close;

                    dataset = [
                        p1_23, p1_45, p1_67, ...
                        p2_23, p2_45, p2_67, ...
                        p3_23, p3_45, p3_67
                        ];

                    if region_of_interest == 1
                        if strcmp(experiment_type, 'partitions-2-8') || strcmp(experiment_type, 'pure-factor-effect') ...
                                || strcmp(experiment_type, 'three-way-interaction')
                            if strcmp(roi_applied, 'one-tailed')
                                load('D:/PhD/fieldtrip/roi/one_tailed_roi_28.mat');
                            elseif strcmp(roi_applied, 'two-tailed')
                                load('D:/PhD/fieldtrip/roi/two_tailed_roi_28.mat');
                            end
                        end
                        dataset = create_hacked_roi(dataset, roi, weight_roi);
                    end

                    all_data = {};
                    for i = 1:numel(dataset)
                        participant = dataset(i);
                        all_data{1}{i} = participant{1}.pgi;
                    end
                    
                    all_designs{1} = design_matrix;
                end
            end


            %% loop all the experiments / designs
            for i =1:numel(all_data)
                data = all_data{i};
                design_matrix = all_designs{i};

                % check that we have atleast 2 elements to perform some
                % analysis
                if numel(data) < 2
                    continue;
                end

                %% check the type of experiment, if its a specific one we should
                %% create a new folder every time
                if strcmp(experiment_type, 'trial-level-2-8')
                    new_save_path = save_path + "_" + num2str(i);
                    mkdir([new_save_path]);
                else
                    new_save_path = save_path;
                end

                %% save the number of participants which went into the analysis
                num_part = numel(data);
                save(strcat(new_save_path, '/number_of_participants.mat'), 'num_part')

                %% setup FT analysis
                % we have to switch to SPM8 to use some of the functions in FT
                %addpath '/Users/cihandogan/Documents/Research/spm8';
                %addpath '/Users/cihandogan/Documents/Research/spm12';
                %rmpath '/Users/cihandogan/Documents/Research/fieldtrip-20230118/external/spm12';
                %rmpath '/Users/cihandogan/Documents/Research/fieldtrip-20230118/external/spm8';

                % we need to tell fieldtrip how our electrodes are structured
                cfg = [];
                cfg.feedback = 'no';
                cfg.method = 'distance';
                cfg.elec = data{1}.elec;
                neighbours = ft_prepare_neighbours(cfg);

                % all experiment configuraitons:
                cfg = [];
                cfg.latency = [start_latency, end_latency];
                cfg.channel = 'eeg';
                cfg.statistic = regressor;
                cfg.method = 'montecarlo';
                cfg.correctm = 'cluster';
                cfg.neighbours = neighbours;
                cfg.clusteralpha = 0.025;
                cfg.numrandomization = 3000;
                cfg.tail = 0;
                cfg.design = design_matrix;
                cfg.computeprob = 'yes';
                cfg.alpha = 0.05;
                cfg.correcttail = 'alpha';


                %% run the fieldtrip analyses
                if contains(type_of_analysis, 'time_domain')
                    if contains(experiment_type, 'onsets-2-8-explicit') && strcmp(regression_type, 'no-factor') || contains(regression_type, 'eye')
                        cfg.clusterthreshold = 'nonparametric_individual';
                        cfg.uvar = 1;
                        cfg.ivar = 2;
                        null_data = set_values_to_zero(data); % create null data to hack a t-test
                        stat = ft_timelockstatistics(cfg, data{:}, null_data{:});
                        save(strcat(new_save_path, '/stat.mat'), 'stat')
                        desired_cluster =1;
                        get_region_of_interest_electrodes(stat, desired_cluster, "roi");
                    elseif contains(experiment_type, 'partitions') || contains(experiment_type, 'onsets-2-8-explicit') ...
                            || contains(experiment_type, 'onsets-1-factor') || contains(experiment_type, 'erps-23-45-67') ...
                            || contains(experiment_type, 'three-way-interaction') || contains(experiment_type, 'Partitions') ...
                            || contains(experiment_type, 'trial-level-2-8') || contains(experiment_type, 'pure-factor-effect')
                        cfg.ivar = 1;
                        cfg.clusterthreshold = 'nonparametric_individual';
                        stat = ft_timelockstatistics(cfg, data{:});

                        %disp(stat.posclusters(1).prob)

                        save(strcat(new_save_path, '/stat.mat'), 'stat')

                        if ~isfield(stat, 'posclusters') || ~isfield(stat, 'negclusters')
                            continue;
                        end
                    end
                elseif strcmp(type_of_analysis, 'frequency_domain')
                    if contains(experiment_type, 'onsets-2-8-explicit')     
                        desired_cluster = 1;
                        cfg.uvar = 1;
                        cfg.ivar = 2;
                        cfg.avgoverfreq = 'yes';
                        cfg.frequency = [foi(1) foi(2)];
                        cfg.clusterthreshold = 'nonparametric_individual';
                        stat = ft_freqstatistics(cfg, data{:}, null_data{:});
                        save(strcat(new_save_path, '/stat.mat'), 'stat')
                        roi_name = "roi_" + "freq_" +  int2str(foi(1)) + "_" + int2str(foi(2));
                        %get_region_of_interest_electrodes(stat, desired_cluster, roi_name);
                    else
                        cfg.clusterthreshold = 'nonparametric_individual';
                        cfg.frequency = [foi(1) foi(2)];
                        cfg.ivar = 1;
                        cfg.avgoverfreq = 'yes';
                        cfg.clusterthreshold = 'nonparametric_individual';
                        stat = ft_freqstatistics(cfg, data{:});
                        save(strcat(new_save_path, '/stat.mat'), 'stat')
                        if ~isfield(stat, 'posclusters') || ~isfield(stat, 'negclusters')
                            continue;
                        end
                    end
                end

                %% check if we have anything
                if size(stat.posclusters, 2) == 0
                    continue;
                end

                %% get peak level stats
                pos_cluster_peak_level_data = {};
                if numel(stat.posclusters) > 0
                    for i=1:numel(stat.posclusters)
                        [pos_peak_level_stats, pos_all_stats] = get_peak_level_stats(stat, i, 'positive');
                        fname = "/pos_peak_level_stats_c_" + num2str(i) + ".mat";
                        save(strcat(new_save_path, fname), 'pos_all_stats')
                        if i == 1
                            first_pos_peak_level_stats = pos_peak_level_stats;
                            first_pos_all_stats = pos_all_stats;
                        end
                        pos_cluster_peak_level_data{i} = pos_peak_level_stats;
                    end
                end

                neg_cluster_peak_level_data = {};
                if numel(stat.negclusters) > 0
                    for i=1:numel(stat.negclusters)
                        [neg_peak_level_stats, neg_all_stats] = get_peak_level_stats(stat, i, 'negative');
                        fname = "/neg_peak_level_stats_c_" + num2str(i) + ".mat";
                        save(strcat(new_save_path, '/neg_peak_level_stats.mat'), 'neg_all_stats')
                        if i == 1
                            first_neg_peak_level_stats = neg_peak_level_stats;
                            first_neg_all_stats = neg_all_stats;
                        end
                        neg_cluster_peak_level_data{i} = neg_peak_level_stats;
                    end
                end

                %% function that plots the t values through time and decides whcih electrode to plot
                if numel(stat.posclusters) > 0
                    pos_peak_level_stats = compute_best_electrode_from_t_values(stat,first_pos_all_stats,new_save_path, 'positive', first_pos_peak_level_stats);
                end
                if numel(stat.negclusters) > 0
                    neg_peak_level_stats = compute_best_electrode_from_t_values(stat,first_neg_all_stats,new_save_path, 'negative', first_neg_peak_level_stats);
                end


                %% generate ERPs using the stat information
                number_of_clusters_to_plot = 2; % plots top k clusters
                plot_thin_med_thick = 0;
                if generate_erps == 1
                    num_pos_clusters = numel(stat.posclusters);
                    if num_pos_clusters > 0
                        for i = 1:1
                            if i <= number_of_clusters_to_plot
                                generate_peak_erps(master_dir, main_path, experiment_type, ...
                                    stat, pos_cluster_peak_level_data{i}, 'positive', desired_design_mtx, i, ...
                                    new_save_path, weight_erps, weighting_factor, ...
                                    type_of_analysis, foi, plot_thin_med_thick, toi);
                            end
                        end
                    end
                    %num_neg_clusters = numel(stat.negclusters);
                    %if num_neg_clusters > 0
                    %    for i = 1:1
                    %        if i <= number_of_clusters_to_plot
                    %            generate_peak_erps(master_dir, main_path, experiment_type, ...
                    %                stat, neg_cluster_peak_level_data{i}, 'negative', desired_design_mtx, i, ...
                    %                new_save_path, weight_erps, weighting_factor, ...
                    %                type_of_analysis, foi);
                    %        end
                    %    end
                    %end
                end

                %% get cluster level percentage through time
                if region_of_interest == 1
                    volume_descriptor = " of mean/intercept ROI";
                else
                    volume_descriptor = " of entire volume";
                end

                % 1 for the most positive going cluster
                if numel(stat.posclusters) > 0
                    title = "Positive going clusters through time as a %" + volume_descriptor;
                    calculate_cluster_size(stat, title, 'positive', new_save_path);
                end

                %if numel(stat.negclusters) > 0
                %    title = "Negative going clusters through time as a %" + volume_descriptor;
                %    calculate_cluster_size(stat, title, 'negative', new_save_path);
                %end

                %% make pretty plots
               if create_topographic_maps == 1 &&  numel(stat.posclusters) > 0
                    create_viz_topographic_maps(data, stat, start_latency, end_latency, ...
                        0.05, 'positive', new_save_path, type_of_analysis, toi)

                    %% plot the peak electrode
                    create_viz_peak_electrode(stat, first_pos_peak_level_stats, new_save_path)

                    %create_viz_topographic_maps(data, stat, start_latency, end_latency, ...
                    %    0.05, 'negative', new_save_path, type_of_analysis)
                end
            end
        end
    end
end

%% used to set frequency data values to 0 for time
function data = set_values_to_zero(data)
for idx = 1:numel(data)
    participant_data = data{1,idx};
    avg = participant_data.avg;
    avg(:) = 0;
    participant_data.avg = avg;
    data{1,idx} = participant_data;
end
end

%% used to set frequency data values to 0 for freq
function data = set_values_to_zero_frequency(data)
for idx = 1:numel(data)
    participant_data = data{1,idx};
    spectrum = participant_data.powspctrm;
    spectrum(:) = 0;
    participant_data.powspctrm = spectrum;
    data{1,idx} = participant_data;
end
end

%% creates a topographic map highlighting the peak electrode
function create_viz_peak_electrode(stat, pos_peak_level_stats, save_dir)


elecs = zeros(size(stat.elec.chanpos,1), 1);
peak_electrode = pos_peak_level_stats.electrode;
e_idx = find(contains(stat.label,peak_electrode));
elecs(e_idx)=1;

save_dir = save_dir + "/" + "highlighted_electrode.png";

cfg = [];
cfg.parameter = 'stat';
cfg.zlim = [-5, 5];
cg.colorbar = 'yes';
cfg.marker = 'off';
cfg.markersize = 1;
cfg.highlight = 'on';
cfg.highlightchannel = find(elecs);
cfg.highlightcolor = {'r', [0 0 1]};
cfg.highlightsize = 10;
cfg.comment = 'no';
cfg.style = 'blank';


ft_topoplotER(cfg, stat);

set(gcf,'Position',[100 100 250 250])
exportgraphics(gcf,save_dir,'Resolution',500);
close;

end

%% update design to remove missing participants
function [all_designs, all_data] = update_design_matrix(data, design)
[all_designs, all_data] = deal({}, {});

num_trials = size(data, 2);
for i = 1:num_trials
    subset = data{i};
    num_participants = size(subset, 2);
    design_at_trial_t = [];
    participants_at_trial_t = {};
    for j = 1:num_participants
        participant = data{i}{j};
        if isstruct(participant)
            participants_at_trial_t{end+1} = participant;
            design_at_trial_t(end+1) = design(j);
        end
    end
    all_designs{i} = design_at_trial_t;
    all_data{i} = participants_at_trial_t;
end
end

%% create a dataset with increasing number of trials
function new_data = create_data_with_increasing_number_of_trials(data, n_trials, roi)

    function pass = check_conditions(n, kth, thin,med,thick)
        thin_found = 0;
        thick_found = 0;
        med_found = 0;

        if kth == 1
            k_start = 1;
        else
            k_start = kth - n;
        end

        if sum(ismember(k_start:kth, thin)) >= 1 && ...
                sum(ismember(k_start:kth, thick)) >= 1 && ...
                sum(ismember(k_start:kth, med)) >= 1
            pass = 1;
        else
            pass = 0;
        end
    end


    function new_data = return_average(data)
        n = numel(data);
        new_data = [];
        for k=1:n
            new_data(:,:,k) = data{k};
        end

        if numel(size(new_data)) == 2
            return;
        else
            new_data = mean(new_data,3);
        end
    end

new_data = {};
for kth=1:n_trials
    disp("processing " + int2str(kth))
    for i=1:numel(data)
        participant = data{i};

        thin = participant.thin;
        thick = participant.thick;
        med = participant.med;

        % get the orders, make sure all three of the same trial exist
        thin_order = squeeze(participant.thin_order) + 1;
        thick_order = squeeze(participant.thick_order) + 1;
        med_order = squeeze(participant.med_order) + 1;

        if check_conditions(100, kth, thin_order, med_order, thick_order)
            if numel(thin) < kth
                thin_cpy = return_average(thin(1:numel(thin)));
            else
                thin_cpy = return_average(thin(1:kth));
            end

            if numel(med) < kth
                med_cpy = return_average(med(1:numel(med)));
            else
                med_cpy = return_average(med(1:kth));
            end

            if numel(thick) < kth
                thick_cpy = return_average(thick(1:numel(thick)));
            else
                thick_cpy = return_average(thick(1:kth));
            end

            pgi = med_cpy - (thick_cpy + thin_cpy)/2;
            participant_cpy = participant;
            participant_cpy.avg = pgi;
            participant_cpy = rmfield(participant_cpy, 'med');
            participant_cpy = rmfield(participant_cpy, 'thin');
            participant_cpy = rmfield(participant_cpy, 'thick');
            participant_cpy = create_hacked_roi(participant_cpy, roi, 0);
            participant_cpy = participant_cpy{1};
            new_data{kth}{i} = participant_cpy;
        else
            new_data{kth}{i} = nan;
        end
    end
end
end

%% applies dummy coordinates at the top of the scalp to eye electrodes
function data = apply_dummy_coordinates_to_eye_electrodes(data)
for i=1:numel(data)
    participant = data{i};
    elec = participant.elec;
    elec.label = participant.label;
    elec.chantype = {'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg'};
    elec.chanunit = {'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V'};
    dummy_coordinates = [
        [39.0041,73.3611,14.3713], % C16
        [66.289,55.4377,6.436], % C8
        [-30.8659,74.2231,18.9699], % C29
        [-61.033,56.6667,14.7309], % C30
        [3.4209,79.7912,16.9555], % C17
        [4.6444,72.0457,39.0973], % C18
        ];
    elec.chanpos = dummy_coordinates;
    elec.elecpos = dummy_coordinates;
    participant.elec = elec;
    data{i} = participant;
end
end


%% Extract the frequency where power is highest
function extract_frequency_from_highest_power(data, foi, toi, electrode)
frequencies = data{1}.med.freq;
time = data{1}.med.time;
electrodes = data{1}.med.label;

electrode_idx = find(strcmp(electrodes,electrode));
[~, start_time] = min(abs(time-toi(1)));
[~, end_time] = min(abs(time-toi(2)));
[~, start_freq] = min(abs(frequencies-foi(1)));
[~, end_freq] = min(abs(frequencies-foi(2)));

freq_of_max_pow = [];
for k = 1:numel(data)
    thin = data{k}.thin.powspctrm;
    med = data{k}.med.powspctrm;
    thick = data{k}.thick.powspctrm;
    pgi = med - (thin + thick)/2;
    participant_number = data{k}.participant_number;

    pgi_at_electrode = squeeze(pgi(electrode_idx,...
        start_freq:end_freq, ...
        start_time:end_time ...
        ));

    max_power = max(pgi_at_electrode, [], 'all');
    [f, ~] = find(pgi_at_electrode==max_power);
    freq_of_maximum_power = frequencies(f);

    freq_of_max_pow(k,1) = freq_of_maximum_power;
    freq_of_max_pow(k,2) = participant_number;
end
end
%% extract specific timeseries values
function extract_time_series_values(data, participants, roi, electrode)
electrode_idx = find(strcmp(data{1}.label,electrode));

datas = [];
for k = 1:numel(data)
    pgi = data{k}.avg;
    time = data{k}.time;
    [~, start_idx] = min(abs(time-roi(1)));
    [~, end_idx] = min(abs(time-roi(2)));
    pgi_at_electrode = pgi(electrode_idx,start_idx:end_idx);
    m_pgi = max(pgi_at_electrode);
    participant_number = participants{k};
    datas(k,1) = m_pgi;
    datas(k,2) = participant_number;
end
end



%% plot the t values through time and select the best electrode
function peak_stat_info = compute_best_electrode_from_t_values(stat, electrode_stats, save_dir, tail, peak_stat_info)

% updates the mask matrix with 0s if not in any of the clusters found
% in the label matrix
    function significant_masks = update_mask_matrix(stat, tail)
        significant_masks = stat.mask;

        if strcmp(tail, 'positive')
            cla = stat.posclusterslabelmat;
        else
            cla = stat.negclusterslabelmat;
        end

        for k = 1:size(cla, 1)
            for j = 1:size(cla, 2)
                cluster_assgn = cla(k, j);
                if cluster_assgn == 0
                    significant_masks(k, j) = 0;
                end
            end
        end
    end

significant_masks = update_mask_matrix(stat, tail);
electrodes = electrode_stats.electrodes;

if strcmp(tail, 'positive')
    cluster_labels = stat.posclusterslabelmat;
    fname = '/positive_t_values_through_time.png';
else
    cluster_labels = stat.negclusterslabelmat;
    fname = '/negative_t_values_through_time.png';
end

time = stat.time;
most_significant_electrode = electrode_stats.electrodes{1};
previous_best = 0;

for i=1:numel(electrodes)
    electrode_idx = find(strcmp(stat.label,electrodes(i)));
    significant_time_series = significant_masks(electrode_idx,:);
    cluster_time_series = cluster_labels(electrode_idx,:);
    raw_time_series = zeros(1,numel(cluster_time_series));
    significance_count = 0;
    for j=1:numel(significant_time_series)
        if cluster_time_series(j)==1
            is_significant = significant_time_series(j);
            significance_count = significance_count + is_significant;
            raw_time_series(j) = significance_count;
        else
            raw_time_series(j) = significance_count;
        end
    end
    if previous_best < significance_count && strcmp(tail, 'positive')
        e = electrodes(i);
        most_significant_electrode = e;
        previous_best = significance_count;
    elseif previous_best > significance_count && strcmp(tail, 'negative')
        e = electrodes(i);
        most_significant_electrode = e;
        previous_best = significance_count;
    end
    if sum(raw_time_series) > 0
        plot(time, raw_time_series);
        hold on;
    end
end

[col, ~] = size(electrode_stats);

for i = 1:col
    if strcmp(electrode_stats.electrodes{i}, most_significant_electrode)
        peak_stat_info.electrode = most_significant_electrode;
        peak_stat_info.time = electrode_stats.time(i);
        peak_stat_info.t_value = electrode_stats.t_value(i);
    end
end


xlabel('Time (ms)');
ylabel('Cumulative Frequency (If T-value is significant');
legend(electrodes,'Location','northwest')
title(strcat('Sustained Significant Electrode:',most_significant_electrode));
save_dir = strcat(save_dir, fname);
set(gcf,'Position',[100 100 1000 350])
exportgraphics(gcf,save_dir,'Resolution',500);
close;
end

%% plot and save the design matrix
function save_desgin_matrix(design_matrix, n_participants, save_path, experiment_type)
plot(design_matrix(1:n_participants), 'color', 'r', 'LineWidth', 3.5);
hold on;
plot(design_matrix(n_participants+1:n_participants*2), 'color', 'g', 'LineWidth', 3.5);
plot(design_matrix((n_participants*2)+1:n_participants*3), 'color', 'b', 'LineWidth', 3.5);
xlabel('Participants','FontSize', 14);
ylabel('Value in Regressor','FontSize', 14);
if strcmp(experiment_type, 'habituation')
    legend({'P1', 'P2', 'P3'},'Location','bestoutside','FontSize', 14)
else
    legend({'Onsets 2:3', 'Onsets 4:5', 'Onsets 6:7'},'Location','bestoutside','FontSize', 14)
end
set(gcf,'Position',[100 100 500 500])
save_dir = strcat(save_path, '/', 'design_matrix.png');
exportgraphics(gcf,save_dir,'Resolution',500);
close;
end

%% generate ERPs
    function generate_peak_erps(master_dir, main_path, experiment_type, ...
    stat, peak_information, effect_type, regression_type, desired_cluster, ...
    save_dir, weight_erps, weighting_factor, type_of_analysis, foi, plot_thin_med_thick, toi)

df = stat.df;
time = stat.time;
peak_electrode = peak_information.electrode;
peak_time = peak_information.time;
peak_t_value = peak_information.t_value;

if strcmp(effect_type, 'positive')
    if numel(stat.posclusters) < 1 || strcmp(peak_electrode, '')
        return ;
    end

    labels = stat.posclusterslabelmat;
    labels(labels>desired_cluster) =0;
    pvalue = round(stat.posclusters(desired_cluster).prob,4);
    cluster_size = stat.posclusters(desired_cluster).clusterstat;
    plot_desc = "positive_peak_erp_" + num2str(desired_cluster) + ".png";
else
    if numel(stat.negclusters) < 1 || strcmp(peak_electrode, '')
        return ;
    end

    labels = stat.negclusterslabelmat;
    peak_t_value = abs(peak_t_value);
    labels(labels>desired_cluster) =0;
    pvalue = round(stat.negclusters(desired_cluster).prob,4);
    cluster_size = stat.negclusters(desired_cluster).clusterstat;
    plot_desc = "negative_peak_erp_" + num2str(desired_cluster) + ".png";
end

through_time = sum(labels);
start_idx = find(through_time(:), desired_cluster, 'first');
end_idx = find(through_time(:), desired_cluster, 'last');
start_of_effect = time(start_idx);
end_of_effect = time(end_idx);

save_dir = strcat(save_dir, '/', plot_desc);

generate_plots(master_dir, main_path, experiment_type, start_of_effect,...
    end_of_effect, peak_electrode, peak_time, peak_t_value, df, ...
    regression_type, pvalue, cluster_size, save_dir, effect_type, ...
    weight_erps, weighting_factor, type_of_analysis, foi, desired_cluster, plot_thin_med_thick, ...
    toi)

close;
end

%% Create a hacked ROI for freq data
%% psudeo code it tomorrow
function all_new_data = create_hacked_roi_freq(data, roi)
    roi_clusters = squeeze(roi.clusters);
    roi_time = roi.time;
    participants = data{1};

    new_participant_data = {};
    for i = 1:numel(participants)
        participant_data = participants{i};
        data = participant_data.powspctrm;
        data_time = participant_data.time;
   
        new_data = NaN(size(data));
    
        for electrode = 1:size(roi_clusters, 1)
            for t = 1:size(roi_clusters, 2)
                cluster_at_elec_time = roi_clusters(electrode, t);
                if cluster_at_elec_time == 1
                    cluster_time_at_t = roi_time(t);
                    
                     [~,idx]=min(abs(cluster_time_at_t-data_time));

                    data_points_all_freq = data(electrode, :, idx);
                    new_data(electrode, :, idx) = data_points_all_freq;
                end 
            end
        end
        participant_data.powspctrm = new_data;
        new_participant_data{i} = participant_data;
    end

all_new_data = {new_participant_data};
end 

%% Create a hacked ROI in the data
function new_data = xcreate_hacked_roi(data, roi, weight_roi)
roi_clusters = roi.clusters;
roi_time = roi.time;

start_latency = NaN;
end_latency = NaN;
new_data = {};
for idx_i = 1:numel(data)
    each_participant = data{idx_i};
    participant_data = each_participant.avg;
    participant_time = each_participant.time;
    [electrodes, time] = size(participant_data);
    new_participant_data = NaN(electrodes, time);
    time_x = 1:1:time;
    time_x = time_x * 2;

    for roi_idx = 1:numel(roi_time)
        t = roi_time(roi_idx)*1000;
        [~,idx]=min(abs(time_x-t));
        idx = idx + 100; % for baselining period
        clusters_at_t = roi_clusters(:,roi_idx);

        if isnan(start_latency) && sum(clusters_at_t)>=1
            start_latency =t;
        elseif roi_idx == numel(roi_time)
            end_latency = t;
        end

        for electrode_idx=1:numel(clusters_at_t)
            if clusters_at_t(electrode_idx) == 1
                new_participant_data(electrode_idx, idx) = participant_data(electrode_idx, idx);
            end
        end

    end




    %[~,start_idx]=min(abs(time_x-start_latency));
    %[~,end_idx]=min(abs(time_x-end_latency));
    %new_participant_data = new_participant_data(:,start_idx:end_idx);
    each_participant.avg = new_participant_data;

    %time = each_participant.time;
    %[~,start_idx]=min(abs(time-start_latency/1000));
    %[~,end_idx]=min(abs(time-end_latency/1000));

    new_data{idx_i} = each_participant;
end
end

function new_data = create_hacked_roi(data, roi, weight_roi)
    % Extract cluster and time information from the ROI
    roi_clusters = roi.clusters;
    roi_time = roi.time;

    % Initialize latency and result variables
    start_latency = NaN;
    end_latency = NaN;
    new_data = {}; % Initialize empty cell array for the modified data

    % Loop through each participant's data
    for participant_idx = 1:numel(data)
        participant = data{participant_idx}; % Current participant's data
        original_data = participant.avg;    % Extract EEG data
        participant_time = participant.time; % Extract time vector
        [num_electrodes, num_timepoints] = size(original_data); % Data dimensions
        modified_data = NaN(num_electrodes, num_timepoints); % Pre-allocate new participant data

        % Loop through ROI time points
        for roi_idx = 1:numel(roi_time)
            roi_time_ms = roi_time(roi_idx);
            [~, closest_time_idx] = min(abs(participant_time - roi_time_ms)); % Find closest time point in participant data

            % Get clusters corresponding to the current ROI time point
            current_clusters = roi_clusters(:, roi_idx);

            % Set start latency at the first detected cluster activation
            if isnan(start_latency) && sum(current_clusters) >= 1
                start_latency = roi_time_ms;
            elseif roi_idx == numel(roi_time)
                % Set end latency at the last ROI time point
                end_latency = roi_time_ms;
            end

            % Modify participant data for active electrodes at this time point
            for electrode_idx = 1:numel(current_clusters)
                if current_clusters(electrode_idx) == 1
                    modified_data(electrode_idx, closest_time_idx) = original_data(electrode_idx, closest_time_idx);
                end
            end
        end

        % Store the modified data for this participant
        participant.avg = modified_data;
        new_data{participant_idx} = participant;
    end
end


%% baed on onsets 2-8 mean intercept effect, get the start and end latency
%% based on our ROI
function get_start_and_end_latency_of_mean_intercept(stat)
load(roi_fn);
roi_labels = roi.label;
stat_labels = stat.label;
time = stat.time;
cluster_labels = stat.posclusterslabelmat;
cluster_labels(cluster_labels>1) = 0;
cluster_sum_by_time = sum(cluster_labels);

start_found = 0;
end_found=0;
start_latecy = 0;
end_latency = 0;

for i = 1:numel(cluster_sum_by_time)
    if cluster_sum_by_time(i) > 0
        electrodes =  cluster_labels(:,i);
        electrodes_with_effect = find(electrodes==1);
        t = time(i);

        for electrode_idx = 1:numel(stat_labels)
            if any(electrodes_with_effect(:)==electrode_idx)
                electrode_name = stat_labels(electrode_idx);
                if numel(intersect(electrode_name, roi_labels)) ==1
                    if start_found == 0
                        start_latency =  t;
                        start_found =1;
                    end
                end
            end
        end
    end
end
end

%% based on onsets 2-8 mean intercept effect, save the electrodes
% which are at the peak level - used for ROI analysis
% saves the neighbourhood of electrodes for use later on
function get_region_of_interest_electrodes(stat, desired_cluster, roi_name)
roi_clusters = stat.posclusterslabelmat;
roi_clusters(roi_clusters>desired_cluster) = 0;
time = stat.time;
roi.clusters = roi_clusters;
roi.time = time;

filename = strcat("C:\Users\CDoga\Documents\Research\PhD\fieldtrip\roi\", roi_name);

save(filename, 'roi');

end

%% plots the cluster effect through time
function create_viz_cluster_effect(stat, alpha)
cfg = [];
cfg.alpha = alpha;
cfg.parameter = 'stat';
cfg.zlim = [-4 4];
cfg.xlim = [0.00, 0.800];
cfg.highlightcolorpos = [1,0,0];
cfg.highlightsymbolseries =  ['*', '*','*','*','*'];
%cfg.toi = [0.56, 0.256];
ft_clusterplot(cfg, stat);
end

%% used to calculate the cluster size through time
function calculate_cluster_size(stat, ptitle, type, save_dir)
if contains(type, 'positive')
    cluster_labelling_mtx = stat.posclusterslabelmat;
    number_of_formed_clusters = unique(stat.posclusterslabelmat);
    number_of_formed_clusters = number_of_formed_clusters(number_of_formed_clusters~=0);
    significance_clusters = stat.posclusters;
    save_dir = strcat(save_dir, '/', 'positive_cluster.png');
elseif contains(type, 'negative')
    cluster_labelling_mtx = stat.negclusterslabelmat;
    number_of_formed_clusters = unique(stat.negclusterslabelmat);
    number_of_formed_clusters = number_of_formed_clusters(number_of_formed_clusters~=0);
    significance_clusters = stat.negclusters;
    save_dir = strcat(save_dir, '/', 'negative_cluster.png');
end

time_mtx = stat.time;
[electrodes, time] = size(cluster_labelling_mtx);
colours = ['r', 'g', 'b', 'm', 'y', 'c', 'k', 'w', 'w', 'w', 'w', 'w', 'w', 'w'];
legend_to_use = {};

ylim_max = 0;

if numel(number_of_formed_clusters) > 5
    n = 5;
else
    n = numel(number_of_formed_clusters);
end

for k=1:n
    desired_cluster = number_of_formed_clusters(k);
    c = colours(k);

    description = "Cluster: " + num2str(desired_cluster) + " p-value: " + num2str(round(significance_clusters(desired_cluster).prob, 4));
    legend_to_use{k} = description;

    cluster_size_through_time = zeros(2,time);
    for i = 1:time
        t = time_mtx(i);
        electrodes_in_time = cluster_labelling_mtx(:,i);
        clusters_in_time = find(electrodes_in_time==desired_cluster);
        cluster_size_through_time(1,i) = t;
        cluster_size_through_time(2,i) = numel(clusters_in_time)/electrodes;
    end
    area(cluster_size_through_time(1,:)*1000, cluster_size_through_time(2,:)*100);
    curr_max = max(cluster_size_through_time(2,:)*100, [], 'all') + 5;
    if curr_max > ylim_max
        ylim_max = curr_max;
    end
    hold on;
end
ylim([0 ylim_max]);
grid on;
xlabel('Time (ms)', 'FontSize', 14);
ylabel('Percentage of cluster', 'FontSize', 14);
xlim([0,400])
%xlim([0,800])
%xlim([0,3000])
title(ptitle, 'FontSize', 14);
legend(legend_to_use, 'Location', 'northwest', 'FontSize',14);


set(gcf,'Position',[100 100 750 350])
exportgraphics(gcf,save_dir,'Resolution',500);
close;
end

%% used to generate 'peak' level stats
 function [peak_level_electrode, all_electrode_stats] = get_peak_level_stats(stat, cluster, type)
peak_level_stats = containers.Map;

time = stat.time;
electrodes = stat.label;
t_values = stat.stat;
if contains(type, 'positive')
    cluster_matrix_locations = stat.posclusterslabelmat;
elseif contains(type, 'negative')
    cluster_matrix_locations = stat.negclusterslabelmat;
end


[rows, columns] = size(cluster_matrix_locations);
for col = 1:columns
    for row = 1:rows

        t_value = t_values(row, col);
        t = time(col);
        electrode = electrodes{row};

        which_cluster = cluster_matrix_locations(row, col);
        if which_cluster == cluster
            keys = peak_level_stats.keys;
            if any(strcmp(keys,electrode))
                previous_values = peak_level_stats(electrode);
                previous_t = previous_values{2};
                if t_value > previous_t && contains(type, 'positive')
                    peak_level_stats(electrode) = {t, t_value};
                elseif t_value < previous_t && contains(type, 'negative')
                    peak_level_stats(electrode) = {t, t_value};
                end
            else
                peak_level_stats(electrode) = {0, 0};
            end
        end
    end
end


peak_level_electrode = get_most_significant_electrode(peak_level_stats, type);
all_electrode_stats = get_all_electrode_stats(peak_level_stats, type);
end

function mtx = get_all_electrode_stats(peak_level_stats, type)
keys = peak_level_stats.keys;
electrodes = cell(numel(keys), 1);
time = zeros(numel(keys), 1);
t_value = zeros(numel(keys), 1);

for i = 1:numel(keys)
    electrode = keys{i};
    stats = peak_level_stats(electrode);
    tv = stats{2};
    t = stats{1};

    electrodes{i} = electrode;
    time(i) = t;
    t_value(i) = tv;
end

mtx = table(electrodes, time, t_value);
if strcmp(type, 'positive')
    mtx = sortrows(mtx, {'t_value'}, {'descend'});
elseif strcmp(type, 'negative')
    mtx = sortrows(mtx, {'t_value'}, {'ascend'});
end
end

function most_significant = get_most_significant_electrode(peak_level_stats, type)
keys = peak_level_stats.keys;

most_significant.electrode = '';
most_significant.time = 0;
most_significant.t_value = 0;
for i = 1:numel(keys)
    electrode = keys{i};
    stats = peak_level_stats(electrode);
    t_value = stats{2};
    time = stats{1};
    if t_value > most_significant.t_value && contains(type, 'positive')
        most_significant.electrode = electrode;
        most_significant.time = time;
        most_significant.t_value = t_value;
    elseif t_value < most_significant.t_value && contains(type, 'negative')
        most_significant.electrode = electrode;
        most_significant.time = time;
        most_significant.t_value = t_value;
    end
end
end
%% used to create the topographic maps
function create_viz_topographic_maps(data1, stat, start_latency, ...
    end_latency, alpha, type, save_dir, analysis_type, toi)


if numel(stat.negclusters) < 1 && strcmp(type, 'negative')
    return;
elseif numel(stat.posclusters) < 1 && strcmp(type,'positive')
    return;
end

% fixes the interpolation issue of NaNs
stat.stat(isnan(stat.stat))=0;

if strcmp(analysis_type, 'frequency_domain')
    stat.stat(isnan(stat.stat))=0;
    timestep = 0.25; % in seconds (50ms)

    j = [start_latency:timestep:end_latency]; % Temporal endpoints (in seconds) of the ERP average computed in each subplot
    [i1,i2] = match_str(data1{1}.label, stat.label);

    pos_cluster_pvals = [stat.posclusters(:).prob];
    pos_clust = find(pos_cluster_pvals < 0.05);
    pos = ismember(stat.posclusterslabelmat, pos_clust);
    save_dir = strcat(save_dir, '/', 'positive_topographic.png');
    if isempty(pos_clust)
        clust = zeros(size(stat.stat)); %401 before
    else
        clust = pos;
    end


    max_t = max(stat.stat, [], 'all');
    max_t = round(max_t, 2);

    max_iter = numel(j)-1;
    for k = 1:max_iter
        subplot(4,4,k);
        cfg = [];
        cfg.parameter = 'stat';
        cfg.xlim = [j(k) j(k+1)];
        cfg.zlim = [-max_t, max_t];
        clust_int = zeros(128,1);
        t_idx = find(stat.time>=j(k) & stat.time<=j(k+1));
        clust_int(i1) = all(clust(i2,1,t_idx),3);
        %cfg.colorbar = 'SouthOutside';
        cfg.marker ='on';
        cfg.markersize = 3;
        cfg.highlight = 'on';
        cfg.highlightchannel = find(clust_int);
        cfg.highlightcolor = {'r',[0 0 1]};
        cfg.highlightsize = 14;
        cfg.fontsize = 14;
        cfg.comment = 'xlim';
        cfg.commentpos = 'title';
        ft_topoplotTFR(cfg, stat);
    end

else
    cfg = [];
    cfg.channel   = 'all';
    cfg.latency   = 'all';
    cfg.parameter = 'avg';

    grand_avg1 = ft_timelockgrandaverage(cfg, data1{:});
    grand_avg1.elec = data1{1}.elec;
    [i1,i2] = match_str(grand_avg1.label, stat.label);

    figure;
    timestep = 0.025; % 25ms
    sampling_rate = 512;
    sample_count  = length(stat.time);

    j = [start_latency:timestep:end_latency]; % Temporal endpoints (in seconds) of the ERP average computed in each subplot

    %$m = zeros(1,size(j,2));
    %for ith = 1:size(j,2)
    %end

    m = [1:timestep*sampling_rate:sample_count];  % temporal endpoints in M/EEG samples
    m(end+1) = sample_count;

    if contains(type, 'positive')
        pos_cluster_pvals = [stat.posclusters(:).prob];
        pos_clust = find(pos_cluster_pvals < alpha);
        clust = ismember(stat.posclusterslabelmat, pos_clust);
        save_dir = strcat(save_dir, '/', 'positive_topographic.png');
    elseif contains(type, 'negative')
        neg_cluster_pvals = [stat.negclusters(:).prob];
        neg_clust = find(neg_cluster_pvals < alpha);
        clust = ismember(stat.negclusterslabelmat, neg_clust);
        save_dir = strcat(save_dir, '/', 'negative_topographic.png');
    end
    max_iter = numel(m)-1;

    max_t = max(stat.stat, [], 'all');
    max_t = round(max_t, 2);

    for k = 1:max_iter
        subplot(1,8,k);
        cfg = [];
        cfg.xlim = [j(k) j(k+1)];
        cfg.zlim = [-5e-14 5e-14];
        pos_int = zeros(numel(grand_avg1.label),1);
        pos_int(i1) = all(clust(i2, m(k):m(k+1)), 2);
        cfg.highlight = 'on';
        cfg.highlightchannel = find(pos_int);
        cfg.highlightcolor = {'r',[0 0 1]};
        cfg.comment = 'xlim';
        cfg.commentpos = 'title';
        cfg.parameter = 'stat';
        cfg.zlim=[-max_t,max_t];
        cfg.colorbar = 'SouthOutside';
        cfg.fontsize = 11;
        cfg.markerfontsize = 14;
        cfg.highlightfontsize = 14;
        cfg.style = 'straight';
        cfg.labels = 'off';
        %cfg.layout = 'biosemi128.lay';
        ft_topoplotER(cfg, stat);
        %ft_clusterplot(cfg, stat)
    end
end
set(gcf,'Position',[100 100 2000 1500])
exportgraphics(gcf,save_dir,'Resolution',500);

%topo = imread(save_dir);

%timings = topo(1:200, :, :);
%topographic_maps = topo(700:1820, :, :);
%legends = topo(2400:2648, :, :);
%new_topo = [timings; topographic_maps; legends];

%imwrite(new_topo,save_dir,'PNG');
close;
end

%% set all my values to 0 to hack a T test
function data = set_values_to_zero_freq(data)
for idx = 1:numel(data)
    participant_data = data{idx};
    participant_data.pgi.powspctrm(:) = 0;
    data{idx} = participant_data;
end
end

%% a function to remove and update the design matrix for more complex operations
function [scores,new_participants] = tweak_design_matrix(dataset, participant_order, participants, type_of_effect)
% since not all participants have factor scores this part lines up and discards participants with no scores
score = dataset;
calc(:, 1) = score(:, 2);
calc(:, 2) = score(:, 1);
calc = sortrows(calc);
[rows, ~] = size(calc);
ratings(:, 1) = calc(:, 2);
ratings(:, 2) = calc(:, 1);

new_participants = {};
cnt = 1;

for j = 1:numel(participant_order)
    participant = participant_order(j);
    score = ratings(find(ratings(:, 1) == participant{1}), 2);

    if numel(score) > 0
        datas(cnt, 2) = score;
        datas(cnt, 1) = participant{1};
        new_participants{cnt} = participants{j};
        cnt = cnt + 1;
    end

end


scores.one = datas;
scores.two = datas;
scores.three = datas;

min_n = min(scores.one);
scores.one(:,2) = scores.one(:,2) - min_n(2);
scores.two(:,2) = scores.two(:,2) - min_n(2);
scores.three(:,2) = scores.three(:,2) - min_n(2);

if strcmp(type_of_effect, 'habituation')
    scores.one(:,2) = scores.one(:,2) * 2.72;
    scores.two(:,2) = scores.two(:,2) * 1.65;
    scores.three(:,2) = scores.three(:,2) * 1.00;
elseif strcmp(type_of_effect, 'sensitization')
    scores.one(:,2) = scores.one(:,2) * 1.00;
    scores.two(:,2) = scores.two(:,2) * 1.65;
    scores.three(:,2) = scores.three(:,2) * 2.72;
else
    error('Type of experiment not properly specified');
end

[n_participants, ~] = size(datas);

for k=1:n_participants
    p1 = scores.one(k,1);
    p2 = scores.two(k,1);
    p3 = scores.three(k,1);

    if p1 == p2 && p2 == p3
        if strcmp(type_of_effect, 'habituation')
            to_remove = scores.three(k,2);
        elseif strcmp(type_of_effect, 'sensitization')
            to_remove = scores.one(k,2);
        end

        scores.one(k,2) = scores.one(k,2) - to_remove;
        scores.two(k,2) = scores.two(k,2) - to_remove;
        scores.three(k,2) = scores.three(k,2) - to_remove;
    else
        error('Participants do not align...')
    end
end
end

%% gram schmitt algorithm
%% gram-schitt process for orthoganilzation
function y = gschmidt(x,v)

[~,n]=size(x);
y=x(:,1);
for k = 2:n  % orthogonalization process
    z=0;
    for j=1:k-1
        p=y(:,j);
        z=z+((p'*x(:,k))/(p'*p))*p;
    end
    y=[y x(:,k)-z];
end

if nargin == 1
    for j=1:n, y(:,j) = y(:,j)/norm(y(:,j));end % normalizing
end
end



%% function orthog data
function scores = orthog_data(VS, DS, HD, type_of_effect)

num_per_partition = size(VS.one, 1);
VS_combined = [VS.one(:,2); VS.two(:,2); VS.three(:,2)];
HD_combined = [HD.one(:,2); HD.two(:,2); HD.three(:,2)];
DS_combined = [DS.one(:,2); DS.two(:,2); DS.three(:,2)];



X1(:,1) = VS_combined;
X1(:,2) = HD_combined;
X1(:,3) = DS_combined;
X_orthog = gschmidt(X1, 1);
ds_o = X_orthog(:,3);
hd_o = X_orthog(:,2);


if contains(type_of_effect, 'visual_stress')
    scores = VS;
elseif contains(type_of_effect, 'discomfort')
    ds1(:,1) = DS.one(:,1);
    ds2(:,1) = DS.two(:,1);
    ds3(:,1) = DS.three(:,1);

    ds1(:,2) = ds_o(1:num_per_partition);
    ds2(:,2) = ds_o(num_per_partition+1:num_per_partition*2);
    ds3(:,2) = ds_o((num_per_partition*2)+1:num_per_partition*3);

    scores.one = ds1;
    scores.two = ds2;
    scores.three = ds3;
elseif contains(type_of_effect, 'headache')
    hd1(:,1) = HD.one(:,1);
    hd2(:,1) = HD.two(:,1);
    hd3(:,1) = HD.three(:,1);

    hd1(:,2) = hd_o(1:num_per_partition);
    hd2(:,2) = hd_o(num_per_partition+1:num_per_partition*2);
    hd3(:,2) = hd_o((num_per_partition*2)+1:num_per_partition*3);

    scores.one = hd1;
    scores.two = hd2;
    scores.three = hd3;
end

end


%% function to create the design matrix for all experiments
function [design, new_participants] = create_design_matrix_partitions(participant_order, participants, ...
    regression_type, partition, type_of_effect)

dataset = return_scores(regression_type, type_of_effect);

if ~strcmp(type_of_effect, 'null')

    [scores, new_participants] = tweak_design_matrix(dataset, participant_order, participants, type_of_effect);

    if contains(regression_type, 'orthog')
        VS = return_scores('visual_stress', type_of_effect);
        DS = return_scores('discomfort', type_of_effect);
        HD = return_scores('headache', type_of_effect);

        [VS, ~] = tweak_design_matrix(VS, participant_order, participants, type_of_effect);
        [DS, ~] = tweak_design_matrix(DS, participant_order, participants, type_of_effect);
        [HD, ~] = tweak_design_matrix(HD, participant_order, participants, type_of_effect);

        scores = orthog_data(VS, DS, HD, regression_type);
    end

    if partition == 1
        ratings = scores.one;
    elseif partition == 2
        ratings = scores.two;
    elseif partition == 3
        ratings = scores.three;
    elseif partition == 0
        ratings = scores.one;
    end

    design = ratings(:,2);
else
    design = dataset(:, 2);

    % since not all participants have factor scores this part lines up and discards participants with no scores
    score = dataset;
    calc(:, 1) = score(:, 2);
    calc(:, 2) = score(:, 1);
    calc = sortrows(calc);
    [rows, ~] = size(calc);
    ratings(:, 1) = calc(:, 2);
    ratings(:, 2) = calc(:, 1);

    new_participants = {};
    cnt = 1;

    for j = 1:numel(participant_order)
        participant = participant_order(j);
        score = ratings(find(ratings(:, 1) == participant{1}), 2);

        if numel(score) > 0
            datas(cnt, 2) = score;
            datas(cnt, 1) = participant{1};
            new_participants{cnt} = participants{j};
            cnt = cnt + 1;
        end

    end
    design = datas(:, 2);
end
end

%% return scores
function dataset = return_scores(regression_type, type_of_effect)
if strcmp(regression_type, 'no-factor') || contains(regression_type, 'eye')
    dataset = [
        1, 1; 2, 1; 3, 1; 4, 1; 5, 1; 6, 1; 7, 1; 8, 1; 9, 1; 10, 1; 11, 1; 12, 1; 13, 1; 14, 1; 15, 1;
        16, 1; 17, 1; 18, 1; 19, 1; 20, 1; 21, 1; 22, 1; 23, 1; 24, 1; 25, 1; 26, 1; 27, 1; 28, 1;
        29, 1; 30, 1; 31, 1; 32, 1; 33, 1; 34, 1; 35, 1; 36, 1; 37, 1; 38, 1; 39, 1; 40, 1;
        ];


elseif contains(regression_type, 'headache')
    dataset = [
        1, -0.22667; 2, -0.05198; 3, -0.72116; 4, 0.53139; 5, 1.72021; 6, -1.17636; 7, -0.79706; 8, 0.19942; 9, -0.6924;
        10, -0.35826; 11, -0.58533; 12, 2.04136; 13, -0.26573; 14, 1.30963; 15, 3.4497; 16, 0.00172; 17, -0.15026; 19, -0.55639;
        20, -0.41626; 21, 1.14373; 22, -0.56513; 23, -0.72755; 24, -0.43472; 25, -0.69897; 26, -1.34952; 27, 1.40986; 28, 0.36296; 29, 0.04162;
        30, -0.4697; 31, -0.70362; 32, -0.73219; 33, -0.88081; 34, -0.79623; 35, -0.75114; 36, 0.09594; 37, 1.22665; 38, -0.3365; 39, 1.07651;
        40, -0.16678;
        ];


elseif contains(regression_type, 'visual_stress')
    dataset = [
        1, 0.3227; 2, -0.10861; 3, -0.51018; 4, 1.1336; 5, -0.63947; 6, -1.21472; 7, -0.33005; 8, 0.75238; 9, -0.39025; 10, -0.72205;
        11, -0.76904; 12, -1.06297; 13, -0.89853; 14, -1.46715; 15, -1.87343; 16, -1.19871; 17, 0.15415; 19, -0.43427; 20, 0.5867;
        21, 1.0008; 22, -0.11689; 23, 1.72091; 24, 0.14105; 25, 0.62214; 26, -0.74829; 27, 2.02421; 28, 0.67386; 29, -0.02367;
        30, 0.03638; 31, 0.6996; 32, -0.29977; 33, -0.64998; 34, 0.02624; 35, -0.82177; 36, -0.42512; 37, 0.79861; 38, -0.58832; 39, 2.33323;
        40, 2.26667;
        ];

elseif contains(regression_type, 'discomfort')
    dataset = [
        1, -0.264; 2, 0.4459; 3, -0.49781; 4, 1.77666; 5, -0.55638; 6, 0.87174; 7, -0.68504; 8, 0.92835; 9, -0.80581; 10, -0.87505;
        11, 0.39111; 12, -0.76054; 13, -0.68987; 14, 1.60776; 15, -0.19637; 16, 1.13956; 17, 1.53606; 19, -0.08254; 20, 0.12186;
        21, 0.08428; 22, 0.61663; 23, -1.47958; 24, 2.28422; 25, -0.80891; 26, -0.55738; 27, 0.2238; 28, -0.93291; 29, 0.3791; 30, -0.63074;
        31, 2.14683; 32, -1.49948; 33, 1.21954; 34, -0.79734; 35, -0.51303; 36, -1.0687; 37, -0.61345; 38, -1.02592; 39, -0.87653; 40, 0.444;
        ];

end
end

%% load post-processed fildtrip data
function [labels, ft_regression_data, participant_order] = ...
    load_data_for_plotting(main_path, n_participants, filename, partition, peak_electrode, analysis_window)


ft_regression_data = {};
participant_order = {};

idx_used_for_saving_data = 1;
for i=1:n_participants
    disp(strcat('LOADING PARTICIPANT...', int2str(i)));
    participant_main_path = strcat(main_path, int2str(i));

    if exist(participant_main_path, 'dir')
        cd(participant_main_path);

        if isfile(filename)
            load(filename);
        else
            continue;
        end

        e_idx = find(contains(data.label,peak_electrode));

        if partition.is_partition
            if partition.partition_number == 1
                med = data.p1_med(e_idx,:,:);
                thick = data.p1_thick(e_idx,:,:);
                thin = data.p1_thin(e_idx,:,:);
            elseif partition.partition_number == 2
                med = data.p2_med(e_idx,:,:);
                thick = data.p2_thick(e_idx,:,:);
                thin = data.p2_thin(e_idx,:,:);
            elseif partition.partition_number == 3
                med = data.p3_med(e_idx,:,:);
                thick = data.p3_thick(e_idx,:,:);
                thin = data.p3_thin(e_idx,:,:);
            end
        elseif ~partition.is_partition
            med = data.med(e_idx,:,:);
            thin = data.thin(e_idx,:,:);
            thick = data.thick(e_idx,:,:);
        end

        ft.med_t = med;
        ft.thin_t = thin;
        ft.thick_t = thick;
        ft.time = data.time{1};

        ft.med = mean(med,3);
        ft.thick = mean(thick, 3);
        ft.thin = mean(thin, 3);
        ft.avg = ft.med - (ft.thick + ft.thin)/2;

        ft = cut_data_using_analysis_window(ft, analysis_window);


        ft_regression_data{idx_used_for_saving_data} = ft;
        participant_order{idx_used_for_saving_data} = i;
        idx_used_for_saving_data = idx_used_for_saving_data + 1;

        labels = data.label;
    end
end
end

%% load post-processed fildtrip data
function [ft_regression_data, participant_order] = ...
    load_postprocessed_data(main_path, n_participants, filename, partition, domain)

ft_regression_data = {};
participant_order = {};

idx_used_for_saving_data = 1;
for i=1:40
    disp(strcat('LOADING PARTICIPANT...', int2str(i)));
    participant_main_path = strcat(main_path, int2str(i));

    %exp_type = 'mean_intercept';

    %if strcmp(exp_type, 'mean_intercept')
    %   parts_to_filter = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,36,38,39,40];
    %else
    %   parts_to_filter = [1,2,3,4,6,7,8,9,10,12,13,14,17,20,21,22,23,25,26,30,31,32,33,34,38,39,40];
    %end
    
    %if ismember(i,parts_to_filter)



    if exist(participant_main_path, 'dir')
        cd(participant_main_path);
        
        if strcmp(domain, "time_domain_p1")
            domain = "time_domain";
        end

        if exist(domain, 'dir')
             cd(domain)
        else
            continue;
        end

       

        if isfile(filename)
            load(filename);
        else
            continue;
        end

        ft.label = data.label;
        ft.time = data.time{1};
        ft.trialinfo = [1];
        ft.elec = data.elec;
        ft.dimord = 'chan_time';

        % find the condition labels used to match up the data

        if partition.is_partition
            if partition.partition_number == 1
                if isfield(data, 'p1_pgi')
                    pgi = data.p1_pgi;
                end
                thin = data.p1_thin;
                med = data.p1_med;
                thick = data.p1_thick;
            elseif partition.partition_number == 2
                if isfield(data, 'p2_pgi')
                    pgi = data.p2_pgi;
                end
                thin = data.p2_thin;
                med = data.p2_med;
                thick = data.p2_thick;
            elseif partition.partition_number == 3
                if isfield(data, 'p3_pgi')
                    pgi = data.p3_pgi;
                end
                thin = data.p3_thin;
                med = data.p3_med;
                thick = data.p3_thick;
            else
                thin = data.thin;
                med = data.med;
                thick = data.thick;
            end
        elseif ~partition.is_partition
            try
                pgi = data.med - (data.thin + data.thick)/2;
                ft.avg = pgi;
            catch
                disp('freq')
            end
            thin = data.thin;
            med = data.med;
            thick = data.thick;

        end

        if isfield(data, 'p1_pgi') || isfield(data, 'p2_pgi') || isfield(data, 'p3_pgi')
            ft.avg = pgi;
        end

        ft.thin = thin;
        ft.med = med;
        ft.thick = thick;

        if isfield(data, 'thick_order') || isfield(data, 'thin_order') ...
                || isfield(data, 'med_order')
            ft.thick_order = data.thick_order;
            ft.thin_order = data.thin_order;
            ft.med_order = data.med_order;
        end

        ft = rmfield(ft, "trialinfo");

        ft_regression_data{idx_used_for_saving_data} = ft;
        participant_order{idx_used_for_saving_data} = i;
        idx_used_for_saving_data = idx_used_for_saving_data + 1;
    end
    end
end
%end

%% spm loading fn
%% load post-processed fildtrip data
%% return the SPM data in a fieldtrip format
function [ft_regression_data, participant_order] = ...
    load_postprocessed_spm_data(main_path, n_participants, filename, partition)

ft_regression_data = {};
participant_order = {};

partition.partition_number = int2str(partition.partition_number);


idx_used_for_saving_data = 1;
for participant = 1:n_participants


    disp(strcat('LOADING PARTICIPANT...', int2str(participant)));

    participant_main_path = strcat(main_path, int2str(participant));

    if exist(participant_main_path, 'dir')
        cd(participant_main_path);


        med = {};
        thin = {};
        thick = {};


        if participant < 10
            p = strcat('0', int2str(participant));
        else
            p = int2str(participant);
        end

        cd("SPM_ARCHIVE/")

        data_structure = strcat('spmeeg_P', p);
        data_structure = strcat(data_structure, '_075_80hz_rejected_tempesta.mat');
        data_structure = strcat(filename, data_structure);

        if isfile(data_structure)
            load(data_structure);
        else
            continue;
        end

        spm_eeg = meeg(D);
        fieldtrip_raw = spm_eeg.ftraw;

        n_trials = size(D.trials);
        n_trials = n_trials(2);
        mt=1; tht=1; tt=1;
        for index_i = 1:n_trials
            label = D.trials(index_i).label;
            if partition.is_partition
                if contains(label, partition.partition_number) && contains(label, 'medium')
                    med(mt) = fieldtrip_raw.trial(index_i);
                    mt=mt+1;
                elseif contains(label, partition.partition_number) && contains(label, 'thin')
                    thin(tht) = fieldtrip_raw.trial(index_i);
                    tht=tht+1;
                elseif contains(label, partition.partition_number) && contains(label, 'thick')
                    thick(tt) = fieldtrip_raw.trial(index_i);
                    tt=tt+1;
                end
            else
                if contains(label, 'medium')
                    med(mt) = fieldtrip_raw.trial(index_i);
                    mt=mt+1;
                elseif contains(label, 'thin')
                    thin(tht) = fieldtrip_raw.trial(index_i);
                    tht=tht+1;
                elseif contains(label, 'thick')
                    thick(tt) = fieldtrip_raw.trial(index_i);
                    tt=tt+1;
                end
            end
        end

        if isempty(med) || isempty(thin) || isempty(thick)
            continue;
        end

        ft.label = fieldtrip_raw.label;
        ft.elec = fieldtrip_raw.elec;
        ft.med = med;
        ft.thick = thick;
        ft.thin = thin;
        ft.time = fieldtrip_raw.time{1};
        ft.dimord = 'chan_time';

        ft_regression_data{idx_used_for_saving_data} = ft;
        participant_order{idx_used_for_saving_data} = participant;
        idx_used_for_saving_data = idx_used_for_saving_data + 1;
    end
end
end

%% generate erp plots
function generate_plots(master_dir, main_path, experiment_type, start_peak, ...
    end_peak, peak_electrode, peak_effect, t_value, df, regression_type, ...
    pvalue, cluster_size, save_dir, effect_type, weight_erps, weighting_factor, ...
    type_of_analysis, foi, desired_cluster, plot_thin_med_thick, toi)


start_plotting = toi(1) * 1000;
start_plotting = start_plotting - 200;
end_plotting = toi(2) * 1000;
end_plotting = end_plotting + 200;

start_plotting = -200;
end_plotting = 300;


plotting_window = [start_plotting,  end_plotting];
%rmpath C:/ProgramFiles/spm8;
%addpath C:/ProgramFiles/spm12;
cd(master_dir);

%% Are we looking at onsets 2-8 or partitions
% set up the experiment as needed
if strcmp(experiment_type, 'onsets-2-8-explicit')
    n_participants = 40;

    partition.is_partition = 0;
    partition.partition_number = 0;

    if strcmp(type_of_analysis, 'time_domain')
        data_file = 'time_domain_mean_intercept_onsets_2_3_4_5_6_7_8_grand-average.mat';
        [data, ~] = load_postprocessed_data(main_path, n_participants, ...
            data_file, partition, type_of_analysis);
        e_idx = find(contains(data{1}.label,peak_electrode));
    elseif strcmp(type_of_analysis, 'frequency_domain')
        data_file = 'frequency_domain_mean_intercept_onsets_2_3_4_5_6_7_8_grand-average.mat';
        freq = to_frequency_data(main_path, partition, ...
            'load', 'trial-level', foi, ...
            'mean_intercept', data_file, n_participants, toi);
        n_part = numel(freq);

        e_idx = find(contains(freq{1}.pgi.label,peak_electrode));
        data = format_for_plotting_functions(freq);
    end

    ci = bootstrap_erps(data, e_idx);

elseif strcmp(experiment_type, 'pure-factor-effect') || strcmp(experiment_type, 'factor_effect')
    
    if strcmp(type_of_analysis, 'time_domain')
        n_participants = 40;
    
        partition.is_partition = 0;
        partition.partition_number = 0;
        type_of_effect = 'habituation';
    
        data_file = 'time_domain_mean_intercept_onsets_2_3_4_5_6_7_8_grand-average.mat';
        [data, participant_order] = load_postprocessed_data(main_path, n_participants, ...
            data_file, partition, type_of_analysis);
        e_idx = find(contains(data{1}.label,peak_electrode));
    
        [data_h, data_l] = get_partitions_medium_split(data, participant_order,...
            regression_type, 1, type_of_effect, weight_erps, weighting_factor);
    
        ci1_h = bootstrap_erps(data_h, e_idx);
        ci1_l = bootstrap_erps(data_l, e_idx);
    elseif strcmp(type_of_analysis, 'frequency_domain')
        data_file = 'frequency_domain_mean_intercept_onsets_2_3_4_5_6_7_8_grand-average.mat';
        n_participants = 40;
    
        partition.is_partition = 0;
        partition.partition_number = 0;
        type_of_effect = 'habituation';

        [data, participant_order] = load_postprocessed_data(main_path, n_participants, ...
            data_file, partition, type_of_analysis);
        e_idx = find(contains(data{1}.label,peak_electrode));
       
        freq = to_frequency_data(main_path, partition, ...
            'load', 'trial-level', foi, ...
            'mean_intercept', data_file, n_participants, toi);

        data1 = format_for_plotting_functions(freq);

        [data1_h, data1_l] = get_partitions_medium_split(data1, participant_order,...
            regression_type, 1, type_of_effect, weight_erps, weighting_factor);
        ci1_h = bootstrap_erps(data1_h, e_idx);
        ci1_l = bootstrap_erps(data1_l, e_idx);

        data = data1;
    end

elseif strcmp(experiment_type, 'partitions-2-8') && contains(type_of_analysis, 'p1')
    n_participants = 40;
    partition1.is_partition = 1; % partition 1
    partition1.partition_number = 1;
    data_file = 'time_domain_partitions_partitioned_onsets_2_3_4_5_6_7_8_grand-average.mat';
    type_of_effect = 'habituation';

    [data, participant_order_1] = load_postprocessed_data(main_path, n_participants, ...
        data_file, partition1, type_of_analysis);
    e_idx = find(contains(data{1}.label,peak_electrode));
    ci = bootstrap_erps(data, e_idx);

    [data_h, data_l] = get_partitions_medium_split(data, participant_order_1,...
        regression_type, 1, type_of_effect, weight_erps, weighting_factor);
    ci1_h = bootstrap_erps(data_h, e_idx);
    ci1_l = bootstrap_erps(data_l, e_idx);


    %[data1_h, data1_l] = get_partitions_medium_split(data, participant_order_1,...
    % regression_type, 1, type_of_effect, weight_erps, weighting_factor);
    %ci1_h = bootstrap_erps(data1_h, e_idx);
    %ci1_l = bootstrap_erps(data1_l, e_idx);
elseif strcmp(experiment_type, 'partitions (no factor)') || strcmp(experiment_type, 'partitions-1') ||  strcmp(experiment_type, 'partitions-2-8')
    n_participants = 40;

    partition1.is_partition = 1;
    partition1.partition_number = 1;
    partition2.is_partition = 1;
    partition2.partition_number = 2;
    partition3.is_partition = 1;
    partition3.partition_number = 3;


    if strcmp(type_of_analysis, 'time_domain')
        data_file = 'time_domain_partitions_partitioned_onsets_2_3_4_5_6_7_8_grand-average.mat';
        [data1, participant_order_1] = load_postprocessed_data(main_path, n_participants, ...
            data_file, partition1, type_of_analysis);
        e_idx = find(contains(data1{1}.label,peak_electrode));
        [data2, participant_order_2] = load_postprocessed_data(main_path, n_participants, ...
            data_file, partition2, type_of_analysis);
        [data3, participant_order_3] = load_postprocessed_data(main_path, n_participants, ...
            data_file, partition3, type_of_analysis);

        data = [data1,data2,data3];

        type_of_effect = 'habituation';
        [data1_h, data1_l] = get_partitions_medium_split(data1, participant_order_1,...
            regression_type, 1, type_of_effect, weight_erps, weighting_factor);
        ci1_h = bootstrap_erps(data1_h, e_idx);
        ci1_l = bootstrap_erps(data1_l, e_idx);
        [data2_h, data2_l] = get_partitions_medium_split(data2, participant_order_2,...
            regression_type, 2, type_of_effect, weight_erps, weighting_factor);
        ci2_h = bootstrap_erps(data2_h, e_idx);
        ci2_l = bootstrap_erps(data2_l, e_idx);
        [data3_h, data3_l] = get_partitions_medium_split(data3, participant_order_3,...
            regression_type, 3, type_of_effect, weight_erps, weighting_factor);
        ci3_h = bootstrap_erps(data3_h, e_idx);
        ci3_l = bootstrap_erps(data3_l, e_idx);
    elseif strcmp(type_of_analysis, 'frequency_domain')
        data_file = 'frequency_domain_partitions_partitioned_onsets_2_3_4_5_6_7_8_grand-average.mat';
        [data1, participant_order_1] = load_postprocessed_data(main_path, n_participants, ...
            data_file, partition1, type_of_analysis);

        e_idx = find(contains(data1{1}.label,peak_electrode));


        [data2, participant_order_2] = load_postprocessed_data(main_path, n_participants, ...
            data_file, partition2, type_of_analysis);
        [data3, participant_order_3] = load_postprocessed_data(main_path, n_participants, ...
           data_file, partition3, type_of_analysis);
        type_of_effect = 'habituation';

        p1_freq = to_frequency_data(main_path, partition1, ...
            'load', 'trial-level', foi, ...
            'partition_', data_file, n_participants, toi);

        p2_freq = to_frequency_data(main_path, partition2, ...
            'load', 'trial-level', foi, ...
            'partition_', data_file, n_participants, toi);

        p3_freq = to_frequency_data(main_path, partition3, ...
            'load', 'trial-level', foi, ...
            'partition_', data_file, n_participants, toi);
        
        data1 = format_for_plotting_functions(p1_freq);
        data2 = format_for_plotting_functions(p2_freq);
        data3 = format_for_plotting_functions(p3_freq);

        [data1_h, data1_l] = get_partitions_medium_split(data1, participant_order_1,...
            regression_type, 1, type_of_effect, weight_erps, weighting_factor);
        ci1_h = bootstrap_erps(data1_h, e_idx);
        ci1_l = bootstrap_erps(data1_l, e_idx);
        [data2_h, data2_l] = get_partitions_medium_split(data2, participant_order_2,...
            regression_type, 2, type_of_effect, weight_erps, weighting_factor);
        ci2_h = bootstrap_erps(data2_h, e_idx);
        ci2_l = bootstrap_erps(data2_l, e_idx);
        [data3_h, data3_l] = get_partitions_medium_split(data3, participant_order_3,...
            regression_type, 3, type_of_effect, weight_erps, weighting_factor);
        ci3_h = bootstrap_erps(data3_h, e_idx);
        ci3_l = bootstrap_erps(data3_l, e_idx);

        data = [data1, data2, data3];
    end
 elseif strcmp(experiment_type, 'three-way-interaction')
        n_participants = 40;
        partition1.is_partition = 1;
        partition1.partition_number = 1;
        partition2.is_partition = 1;
        partition2.partition_number = 2;
        partition3.is_partition = 1;
        partition3.partition_number = 3;

        onsets_2_3 = 'time_domain_partitions_partitioned_onsets_2_3_grand-average.mat';
        onsets_4_5 = 'time_domain_partitions_partitioned_onsets_4_5_grand-average.mat';
        onsets_6_7 = 'time_domain_partitions_partitioned_onsets_6_7_grand-average.mat';

        type_of_effect = 'habituation';


        % p1,p2,p3 for onsets 2;3

        [p1_23, po_p1_23] = load_postprocessed_data(main_path, n_participants, ...
            onsets_2_3, partition1, type_of_analysis);
        [p2_23, po_p2_23] = load_postprocessed_data(main_path, n_participants, ...
            onsets_2_3, partition2, type_of_analysis);
        [p3_23, po_p3_23] = load_postprocessed_data(main_path, n_participants, ...
            onsets_2_3, partition3, type_of_analysis);
        
        [p1_23_h, p1_23_l] = get_partitions_medium_split(p1_23, po_p1_23,...
            regression_type, 1, type_of_effect, weight_erps, weighting_factor);
        [p2_23_h, p2_23_l] = get_partitions_medium_split(p2_23, po_p2_23,...
            regression_type, 2, type_of_effect, weight_erps, weighting_factor);
        [p3_23_h, p3_23_l] = get_partitions_medium_split(p3_23, po_p3_23,...
            regression_type, 3, type_of_effect, weight_erps, weighting_factor);
        
        
        e_idx = find(contains(p1_23{1}.label,peak_electrode));

        p1_23_h = bootstrap_erps(p1_23_h, e_idx);
        p1_23_l = bootstrap_erps(p1_23_l, e_idx);

        p2_23_h = bootstrap_erps(p2_23_h, e_idx);
        p2_23_l = bootstrap_erps(p2_23_l, e_idx);

        p3_23_h = bootstrap_erps(p3_23_h, e_idx);
        p3_23_l = bootstrap_erps(p3_23_l, e_idx);

        % p1,p2,p3 for onsets 4;5

        [p1_45, po_p1_45] = load_postprocessed_data(main_path, n_participants, ...
            onsets_4_5, partition1, type_of_analysis);
        [p2_45, po_p2_45] = load_postprocessed_data(main_path, n_participants, ...
            onsets_4_5, partition2, type_of_analysis);
        [p3_45, po_p3_45] = load_postprocessed_data(main_path, n_participants, ...
            onsets_4_5, partition3, type_of_analysis);

        [p1_45_h, p1_45_l] = get_partitions_medium_split(p1_45, po_p1_45,...
            regression_type, 1, type_of_effect, weight_erps, weighting_factor);
        [p2_45_h, p2_45_l] = get_partitions_medium_split(p2_45, po_p2_45,...
            regression_type, 2, type_of_effect, weight_erps, weighting_factor);
        [p3_45_h, p3_45_l] = get_partitions_medium_split(p3_45, po_p3_45,...
            regression_type, 3, type_of_effect, weight_erps, weighting_factor);


        p1_45_h = bootstrap_erps(p1_45_h, e_idx);
        p1_45_l = bootstrap_erps(p1_45_l, e_idx);

        p2_45_h = bootstrap_erps(p2_45_h, e_idx);
        p2_45_l = bootstrap_erps(p2_45_l, e_idx);

        p3_45_h = bootstrap_erps(p3_45_h, e_idx);
        p3_45_l = bootstrap_erps(p3_45_l, e_idx);

        % p1,p2,p3 for onsets 6'7
        [p1_67, po_p1_67] = load_postprocessed_data(main_path, n_participants, ...
            onsets_6_7, partition1, type_of_analysis);
        [p2_67, po_p2_67] = load_postprocessed_data(main_path, n_participants, ...
            onsets_6_7, partition2, type_of_analysis);
        [p3_67, po_p3_67] = load_postprocessed_data(main_path, n_participants, ...
            onsets_6_7, partition3, type_of_analysis);

        [p1_67_h, p1_67_l] = get_partitions_medium_split(p1_67, po_p1_67,...
            regression_type, 1, type_of_effect, weight_erps, weighting_factor);
        [p2_67_h, p2_67_l] = get_partitions_medium_split(p2_67, po_p2_67,...
            regression_type, 2, type_of_effect, weight_erps, weighting_factor);
        [p3_67_h, p3_67_l] = get_partitions_medium_split(p3_67, po_p3_67,...
            regression_type, 3, type_of_effect, weight_erps, weighting_factor);

        p1_67_h = bootstrap_erps(p1_67_h, e_idx);
        p1_67_l = bootstrap_erps(p1_67_l, e_idx);

        p2_67_h = bootstrap_erps(p2_67_h, e_idx);
        p2_67_l = bootstrap_erps(p2_67_l, e_idx);

        p3_67_h = bootstrap_erps(p3_67_h, e_idx);
        p3_67_l = bootstrap_erps(p3_67_l, e_idx);

        data = [p1_45];

elseif strcmp(experiment_type, 'erps-23-45-67') || strcmp(experiment_type, 'erps-23-45-67-no-factor')
    if strcmp(type_of_analysis, 'time_domain')
        type_of_effect = 'sensitization';
        data_file23 = 'time_domain_mean_intercept_onsets_2_3_grand-average.mat';
        data_file45 = 'time_domain_mean_intercept_onsets_4_5_grand-average.mat';
        data_file67 = 'time_domain_mean_intercept_onsets_6_7_grand-average.mat';

        n_participants = 40;
        partition.is_partition = 0;
        partition.partition_number = 0;

        [data1, participant_order_1] = load_postprocessed_data(main_path, n_participants, ...
            data_file23, partition, type_of_analysis);
        e_idx = find(contains(data1{1}.label,peak_electrode));
        [data2, participant_order_2] = load_postprocessed_data(main_path, n_participants, ...
            data_file45, partition, type_of_analysis);
        [data3, participant_order_3] = load_postprocessed_data(main_path, n_participants, ...
            data_file67, partition, type_of_analysis);


        [data1_h, data1_l] = get_partitions_medium_split(data1, participant_order_1,...
            regression_type, 1, type_of_effect, weight_erps, weighting_factor);
        ci1_h = bootstrap_erps(data1_h,e_idx);
        ci1_l = bootstrap_erps(data1_l,e_idx);
        [data2_h, data2_l] = get_partitions_medium_split(data2, participant_order_2,...
            regression_type, 2, type_of_effect, weight_erps, weighting_factor);
        ci2_h = bootstrap_erps(data2_h,e_idx);
        ci2_l = bootstrap_erps(data2_l,e_idx);
        [data3_h, data3_l] = get_partitions_medium_split(data3, participant_order_3,...
            regression_type, 3,  type_of_effect, weight_erps, weighting_factor);
        ci3_h = bootstrap_erps(data3_h,e_idx);
        ci3_l = bootstrap_erps(data3_l,e_idx);

        data = [data1, data2, data3];


    elseif strcmp(type_of_analysis, 'frequency_domain')
        type_of_effect = 'sensitization';
        data_file23 = 'frequency_domain_mean_intercept_onsets_2_3_grand-average.mat';
        data_file45 = 'frequency_domain_mean_intercept_onsets_4_5_grand-average.mat';
        data_file67 = 'frequency_domain_mean_intercept_onsets_6_7_grand-average.mat';

        n_participants = 40;
        partition.is_partition = 0;
        partition.partition_number = 0;

        [data1, participant_order1] = load_postprocessed_data(main_path, n_participants, ...
            data_file23, partition, type_of_analysis);
        e_idx = find(contains(data1{1}.label,peak_electrode));
        [data2, participant_order2] = load_postprocessed_data(main_path, n_participants, ...
            data_file45, partition, type_of_analysis);
        [data3, participant_order3] = load_postprocessed_data(main_path, n_participants, ...
            data_file67, partition, type_of_analysis);

        p1_freq = to_frequency_data(main_path, partition, ...
            'load', 'trial-level', foi, ...
            'erp23_', data_file23, n_participants, toi);

        p2_freq = to_frequency_data(main_path, partition, ...
            'load', 'trial-level', foi, ...
            'erp45_', data_file45, n_participants, toi);

        p3_freq = to_frequency_data(main_path, partition, ...
            'load', 'trial-level', foi, ...
            'erp67_', data_file67, n_participants, toi);

        data1 = format_for_plotting_functions(p1_freq);
        data2 = format_for_plotting_functions(p2_freq);
        data3 = format_for_plotting_functions(p3_freq);

        [data1_h, data1_l] = get_partitions_medium_split(data1, participant_order1,...
            regression_type, 1, type_of_effect, weight_erps, weighting_factor);
        ci1_h = bootstrap_erps(data1_h, e_idx);
        ci1_l = bootstrap_erps(data1_l, e_idx);
        [data2_h, data2_l] = get_partitions_medium_split(data2, participant_order2,...
            regression_type, 2, type_of_effect, weight_erps, weighting_factor);
        ci2_h = bootstrap_erps(data2_h, e_idx);
        ci2_l = bootstrap_erps(data2_l, e_idx);
        [data3_h, data3_l] = get_partitions_medium_split(data3, participant_order3,...
            regression_type, 3, type_of_effect, weight_erps, weighting_factor);
        ci3_h = bootstrap_erps(data3_h, e_idx);
        ci3_l = bootstrap_erps(data3_l, e_idx);

        data = [data1, data2, data3];

    end
   
end
%% generate_supplementary information and indices used to plot
if strcmp(experiment_type, 'partitions-2-8') && ~contains(regression_type, 'p1')
    experiment_name = "Illustration of Onsets 2:8 {factor} x Habituation for Partitions";
elseif strcmp(experiment_type, 'partitions-2-8') && contains(regression_type, 'p1')
    experiment_name = "Illustration of Onsets 2:8 First Partition {factor}";
elseif strcmp(experiment_type, 'erps-23-45-67')
    experiment_name = 'Illustration of Onsets (2,3; 4,5; 6,7) {factor} x Sensitization';
elseif strcmp(experiment_type, 'erps-23-45-67-no-factor')
    experiment_name = 'ERPs 2,3; 4,5; 6,7 (No Factor) {factor}';
elseif strcmp(experiment_type, 'onsets-2-8-explicit')
    experiment_name = "Illustration of Onsets 2:8 Mean/Intercept";
elseif strcmp(experiment_type, 'pure-factor-effect')
    experiment_name = "Illustration of Onsets 2:8 Factor {factor}";
elseif strcmp(experiment_type, 'three-way-interaction')
    experiment_name = "Illustration of the Three-Way-Interaction: {factor} x Sensitization (over Onsets) x Habituation (over Partitions)";
else
    experiment_name = experiment_type;
end

first_partition_regresion = 0;
if contains(regression_type, 'p1') || contains(type_of_analysis, 'p1')
    first_partition_regresion = 1;
    label = '-';
else
    label = 'x';
end

if contains(regression_type, 'visual_stress')
    regression_type = "Visual Stress";
elseif contains(regression_type, 'discomfort')
    regression_type = "Discomfort";
elseif contains(regression_type, 'headache')
    regression_type = 'Headache';
end

regression_type = regexprep(regression_type,"(/<[a-z])","${upper($1)}");
effect_type = strcat(regexprep(effect_type,'(/<[a-z])','${upper($1)}'), ' Tail');
experiment_name = strrep(experiment_name,'{factor}',regression_type);
m_title = experiment_name + ", " + "Electrode " + peak_electrode;

start_peak = start_peak*1000;
end_peak = end_peak*1000;
cohens_d = round((2*t_value)/sqrt(df),2);
effect_size = round(sqrt((t_value*t_value)/((t_value*t_value)+df)),2);

time = data{1}.time * 1000;
peak_effect_plotting = peak_effect;
peak_effect = peak_effect*1000;
t_value = round(t_value, 2);
cluster_size = round(cluster_size, 0);

plot_line_width = 2;

labels_text_size = 14;

if strcmp(type_of_analysis, 'frequency_domain')
    ax_label = "dB";
else
    ax_label = "Microvolts (uV)";
end

if contains(experiment_type, 'onsets-2-8')
    t = tiledlayout(2,1, 'TileSpacing','Compact');
    time = data{1}.time * 1000;
    nexttile
    hold on;
    plot(time, ci.dist_pgi_avg, 'color', 'm', 'LineWidth', plot_line_width,'DisplayName','PGI')
    plot(time, ci.dist_pgi_high, 'LineWidth', 0.01, 'color', 'm','DisplayName','');
    plot(time, ci.dist_pgi_low, 'LineWidth', 0.001, 'color', 'm','DisplayName','');
    x2 = [time, fliplr(time)];
    inBetween = [ci.dist_pgi_high, fliplr(ci.dist_pgi_low)];
    h = fill(x2, inBetween, 'b' , 'LineStyle','none');
    set(h,'facealpha',.10)
    xlim(plotting_window);
    ylim([min(ci.dist_pgi_low)-0.5, max(ci.dist_pgi_high)+0.5])
    grid on
    xline(start_peak, '-', "LineWidth", 1.5);
    xline(end_peak, '-', "LineWidth", 1.5);
    xline(peak_effect, '--r', "LineWidth", 1.5);
    xline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)
    yline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)
    legend({'PGI'},'Location','bestoutside','FontSize', labels_text_size)
    %xlabel("Milliseconds", "FontSize",labels_text_size)
    %ylabel(ax_label, "FontSize",labels_text_size)

    %ylh = ylabel("PGI", "FontSize",16, "Rotation", 0, 'HorizontalAlignment','right', "FontWeight", "bold");
    %ylh.Position(2) = 2.75;
    %ylh.Position(1) = -220;

        % Shade the region before start_peak
    xregion([-inf, start_peak], 'FaceColor', [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');
    
    % Shade the region after end_peak
    xregion([end_peak, inf], 'FaceColor', [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');

    nexttile
    hold on;

    plot(NaN(1), 'Color', '#0072BD');
    plot(NaN(1), 'Color', '#D95319');
    plot(NaN(1), 'Color', 'g');
    legend({'Thin', 'Medium', 'Thick'},'Location','bestoutside', 'FontSize', labels_text_size)

    plot(time, ci.dist_thin_avg, 'Color', '#0072BD', 'LineWidth', plot_line_width, 'HandleVisibility','off')
    plot(time, ci.dist_thin_high, 'LineWidth', 0.01, 'Color', '#0072BD','HandleVisibility','off');
    plot(time, ci.dist_thin_low, 'LineWidth', 0.01, 'color', '#0072BD','HandleVisibility','off');
    x2 = [time, fliplr(time)];
    inBetween = [ci.dist_thin_high, fliplr(ci.dist_thin_low)];
    h = fill(x2, inBetween, [0, 0.447, 0.741], 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha',.10)

    plot(time, ci.dist_med_avg, 'color', '#D95319','LineWidth', plot_line_width, 'HandleVisibility','off');
    plot(time, ci.dist_med_high, 'LineWidth', 0.01, 'color', '#D95319','HandleVisibility','off');
    plot(time, ci.dist_med_low, 'LineWidth', 0.01, 'color', '#D95319','HandleVisibility','off');
    x2 = [time, fliplr(time)];
    inBetween = [ci.dist_med_high, fliplr(ci.dist_med_low)];
    h = fill(x2, inBetween, [0.851, 0.325, 0.098], 'HandleVisibility','off' , 'LineStyle','none');
    set(h,'facealpha',.10)


    plot(time, ci.dist_thick_avg, 'color', 'g','LineWidth', plot_line_width, 'HandleVisibility','off');
    plot(time, ci.dist_thick_high, 'LineWidth', 0.01, 'color', 'g','HandleVisibility','off');
    plot(time, ci.dist_thick_low, 'LineWidth', 0.01, 'color', 'g','HandleVisibility','off');
    x2 = [time, fliplr(time)];
    inBetween = [ci.dist_thick_high, fliplr(ci.dist_thick_low)];
    h = fill(x2, inBetween, [1,0.92,0], 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha',.10)

    xline(start_peak, '-', 'HandleVisibility','off', "LineWidth", 1.5);
    xline(end_peak, '-', 'HandleVisibility','off', "LineWidth", 1.5);
    xline(peak_effect, '--r', 'HandleVisibility','off', "LineWidth", 1.5);
    xline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)
    yline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)
    xlim(plotting_window);

    min_y = min([min(ci.dist_thick_low), min(ci.dist_thin_low), min(ci.dist_med_low)])-0.5;
    max_y = max([max(ci.dist_thick_high), max(ci.dist_thin_high), max(ci.dist_med_high)])+0.5;
    ylim([min_y, max_y])

            % Shade the region before start_peak
    xregion([-inf, start_peak], 'FaceColor', [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');
    
    % Shade the region after end_peak
    xregion([end_peak, inf], 'FaceColor', [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');


    %ylh = ylabel("Mean/Intercept", "FontSize",16, "Rotation", 0, 'HorizontalAlignment','right', "FontWeight", "bold");
    %ylh.Position(2) = 2.75;
    %ylh.Position(1) = -220;

    %xlabel("Milliseconds", "FontSize",labels_text_size)
    %ylabel(ax_label, "FontSize",labels_text_size)
    grid on


    hold off;

elseif strcmp(experiment_type,'pure-factor-effect') || strcmp(experiment_type,'factor_effect')
    figure;
    if plot_thin_med_thick        
        % Row 1
        pos1 = [0.05, 0.55, 0.35, 0.35];  % First plot (larger width)
        pos2 = [0.425, 0.55, 0.15, 0.35];  % Second plot (half width)
        pos3 = [0.6, 0.55, 0.35, 0.35];    % Third plot (larger width)
        
        % Row 2
        pos4 = [0.05, 0.5, 0.35, 0.35];    % First plot (larger width)
        pos5 = [0.425, 0.5, 0.15, 0.35];   % Second plot (half width)
        pos6 = [0.6, 0.5, 0.35, 0.35];     % Third plot (larger width)
    else
            % Row 1
        % Define the positions for the plots, keeping (2,1) and (2,3) empty
    % Define the positions for the plots, keeping (2,1) and (2,3) empty
    % Increasing width for pos1, pos3, pos4, and pos6
    pos1 = [0.05, 0.6, 0.325, 0.25];   % Row 1, Col 1 (wider)
    pos2 = [0.415, 0.6, 0.125, 0.25];  % Row 1, Col 2 (narrower)
    pos3 = [0.57, 0.6, 0.325, 0.25];   % Row 1, Col 3 (wider)
    
    posMiddle = [0.415, 0.39, 0.125, 0.15]; % Row 2, Col 2 (centered and narrower)
    
    pos4 = [0.05, 0.1, 0.325, 0.25];   % Row 3, Col 1 (wider)
    pos5 = [0.415, 0.1, 0.125, 0.25];  % Row 3, Col 2 (narrower)
    pos6 = [0.57, 0.1, 0.325, 0.25];   % Row 3, Col 3 (wider)
    end
    
    labels_text_size = 8;


  
    subplot(3,3,1);
    set(gca, 'Position', pos1); % Adjust the size and position
    time = data{1}.time * 1000;

        % Shade the region before start_peak
    xregion([-inf, start_peak], 'FaceColor', [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');
    
    % Shade the region after end_peak
    xregion([end_peak, inf], 'FaceColor', [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');

    line_plot_effect =140.00;

    hold on;
    %plot(NaN(1), 'r');
    %if contains(experiment_type, 'partitions-2-8')
    %    legend({'P1-PGI'},'Location','bestoutside','FontSize', labels_text_size)
    %elseif contains(experiment_type, 'pure-factor-effect')
    %    legend({'PGI'},'Location','bestoutside','FontSize', labels_text_size)
    %end

    % low
    plot(time, ci1_l.dist_pgi_avg, 'color', 'r', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, ci1_l.dist_pgi_high, 'LineWidth', 0.00001, 'color', 'r','HandleVisibility','off');
    plot(time, ci1_l.dist_pgi_low, 'LineWidth', 0.00001, 'color', 'r','HandleVisibility','off');
    x2 = [time, fliplr(time)];
    inBetween = [ci1_l.dist_pgi_high, fliplr(ci1_l.dist_pgi_low)];
    h = fill(x2, inBetween, 'r', 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha',.10)

    xlim(plotting_window);

    title("LOW",'FontSize', 18);
    ylim([-2, 6])
    grid on;
    hold off;

    if peak_effect ~= 0
        xline(start_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
        xline(end_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
        xline(peak_effect, '--r','HandleVisibility','off', "LineWidth", 1.5);
         xline(line_plot_effect, '--','HandleVisibility','off', "LineWidth", 1.5);
        xline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)
        yline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)
    end

    %xlabel("Milliseconds", "FontSize",labels_text_size)
    %ylabel(ax_label, "FontSize",labels_text_size)

    ylh = ylabel("PGI", "FontSize",16, "Rotation", 90, 'HorizontalAlignment','right', "FontWeight", "bold");
    ylh.Position(2) = 2;
    ylh.Position(1) = -220;

    subplot(3,3,2);
    set(gca, 'Position', pos2);
    hold on;
    plot(NaN(1), 'r');
    plot(NaN(1), 'k');
    if contains(experiment_type, 'partitions-2-8')
        legend({'P1-PGI'},'Location','northwest','FontSize', labels_text_size)
    elseif contains(experiment_type, 'pure-factor-effect')
        legend({'PGI'},'Location','northwest','FontSize', labels_text_size)
    end

    % interaction plot
    [~, peak_effect_idx] = min(abs(data{1}.time - line_plot_effect/1000));
    p1_high =  ci1_h.dist_pgi_avg(peak_effect_idx);
    p1_low =  ci1_l.dist_pgi_avg(peak_effect_idx);

    % thin thick average 
     %p1_thin_high = ci1_h.dist_thin_avg(peak_effect_idx);
     %p1_thick_high = ci1_h.dist_thick_avg(peak_effect_idx);
     %p1_thin_low = ci1_l.dist_thin_avg(peak_effect_idx);
     %p1_thick_low = ci1_l.dist_thick_avg(peak_effect_idx);

    %mean_thin_thick_low = mean([p1_thin_low, p1_thick_low]);
    %mean_thin_thick_high = mean([p1_thick_high, p1_thin_high]);

    partitions_1 = [p1_low, p1_high];
    %avg_thin_thick = [mean_thin_thick_low, mean_thin_thick_high];

    plot(partitions_1, '-o','color', 'r' ,'LineWidth', plot_line_width,'HandleVisibility','off');
    %plot(avg_thin_thick, '-o','color', 'k' ,'LineWidth', plot_line_width,'HandleVisibility','off');

    title("LOW vs. HIGH",'FontSize', 18);

    % Customize x-axis labels
    xticks([1 2]);                 % Set x-tick positions
    xticklabels({'LOW', 'HIGH'});   % Replace numeric x-ticks with custom labels
    
    % Remove x-axis numeric labels
    set(gca, 'XTickLabelMode', 'manual', 'XTickMode', 'manual');
      grid on;
    % interaction plot end

    subplot(3,3,3);
    set(gca, 'Position', pos3);
    hold on;


       % Shade the region before start_peak
    xregion([-inf, start_peak], 'FaceColor', [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');
    
    % Shade the region after end_peak
    xregion([end_peak, inf], 'FaceColor', [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');

    hold on;
    plot(time, ci1_h.dist_pgi_avg, 'color', 'r', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, ci1_h.dist_pgi_high, 'LineWidth', 0.00001, 'color', 'r','HandleVisibility','off');
    plot(time, ci1_h.dist_pgi_low, 'LineWidth', 0.00001, 'color', 'r','HandleVisibility','off');
    x2 = [time, fliplr(time)];
    inBetween = [ci1_h.dist_pgi_high, fliplr(ci1_h.dist_pgi_low)];
    h = fill(x2, inBetween, 'r', 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha',.10)

    xlim(plotting_window);

    title("HIGH",'FontSize', 18);
    ylim([-2, 6])
    grid on;
    hold off;

    if peak_effect ~= 0
        xline(start_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
        xline(end_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
        xline(peak_effect, '--r','HandleVisibility','off', "LineWidth", 1.5);
        xline(line_plot_effect, '--','HandleVisibility','off', "LineWidth", 1.5);
        xline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)
        yline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)
    end

    %xlabel("Milliseconds", "FontSize",labels_text_size)
   % ylabel(ax_label, "FontSize",labels_text_size)

    %% subplot middle
    subplot('position', posMiddle);
     hold on;

    plot(NaN(1), 'Color', '#D95319');
    plot(NaN(1), 'Color', 'g');
    if contains(experiment_type, 'partitions-2-8')
        legend({'P1-PGI'},'Location','northwest','FontSize', labels_text_size)
    elseif contains(experiment_type, 'pure-factor-effect')
        legend({'Medium', 'Avg. (Thin, Thick)'},'Location','northwest','FontSize', labels_text_size)
    end

    [~, peak_effect_idx] = min(abs(data{1}.time - line_plot_effect/1000));
    low_med =  ci1_l.dist_med_avg(peak_effect_idx);
    
    low_thin =  ci1_h.dist_thin_avg(peak_effect_idx);
    low_thick =  ci1_l.dist_thick_avg(peak_effect_idx);
    low_thin_thick_avg = mean([low_thin, low_thick]);

    high_med =  ci1_h.dist_med_avg(peak_effect_idx);

    high_thick =  ci1_h.dist_thick_avg(peak_effect_idx);
    high_thin =  ci1_h.dist_thin_avg(peak_effect_idx);
    high_thin_thick_avg = mean([high_thin, high_thick]);
    
    med = [low_med, high_med];
    avg_thin_thick = [low_thin_thick_avg, high_thin_thick_avg];
    

    plot(med, '-o','color', '#D95319' ,'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(avg_thin_thick,  '-o','color', 'g' ,'LineWidth', plot_line_width,'HandleVisibility','off');


    % Customize x-axis labels
    xticks([1 2]);                 % Set x-tick positions
    %xticklabels({'LOW', 'HIGH'});   % Replace numeric x-ticks with custom labels
    
    % Remove x-axis numeric labels
    set(gca, 'XTickLabelMode', 'manual', 'XTickMode', 'manual');
    grid on;


    %% subplot middle

    % low med etc
    subplot(3, 3, 7);
    set(gca, 'Position', pos4);
    hold on;
    %plot(NaN(1), 'Color', '#0072BD');
    %plot(NaN(1), 'Color', '#D95319');
    %plot(NaN(1), 'Color', '#FFFF00');
    %legend({'Thin', 'Medium', 'Thick'},'Location','bestoutside','FontSize', labels_text_size)

    plot(time, ci1_l.dist_thin_avg, 'color', '#0072BD', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, ci1_l.dist_thin_high, 'LineWidth', 0.01, 'color', '#0072BD','HandleVisibility','off');
    plot(time, ci1_l.dist_thin_low, 'LineWidth', 0.01, 'color', '#0072BD','HandleVisibility','off');
    x2 = [time, fliplr(time)];
    inBetween = [ci1_l.dist_thin_high, fliplr(ci1_l.dist_thin_low)];
    h = fill(x2, inBetween, [0, 0.447, 0.741], 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha',.10)

    plot(time, ci1_l.dist_med_avg, 'color', '#D95319', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, ci1_l.dist_med_high, 'LineWidth', 0.01, 'color', '#D95319','HandleVisibility','off');
    plot(time, ci1_l.dist_med_low, 'LineWidth', 0.01, 'color', '#D95319','HandleVisibility','off');
    x2 = [time, fliplr(time)];
    inBetween = [ci1_l.dist_med_high, fliplr(ci1_l.dist_med_low)];
    h = fill(x2, inBetween, [0.851, 0.325, 0.098], 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha',.10)

    plot(time, ci1_l.dist_thick_avg, 'color', '#FCD200', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, ci1_l.dist_thick_high, 'LineWidth', 0.01, 'color', '#FCD200','HandleVisibility','off');
    plot(time, ci1_l.dist_thick_low, 'LineWidth', 0.01, 'color', '#FCD200','HandleVisibility','off');
   x2 = [time, fliplr(time)];
    inBetween = [ci1_l.dist_thick_high, fliplr(ci1_l.dist_thick_low)];
    h = fill(x2, inBetween, [1,0.92,0], 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha',.175)

    xlim(plotting_window);

    ylim([-2, 8.5])
    grid on;
    hold off;

    if peak_effect ~= 0
        xline(start_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
        xline(end_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
        xline(peak_effect, '--r','HandleVisibility','off', "LineWidth", 1.5);
        xline(line_plot_effect, '--','HandleVisibility','off', "LineWidth", 1.5);
        xline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)
        yline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)
    end

    %xlabel("Milliseconds", "FontSize",labels_text_size)
    %ylabel(ax_label, "FontSize",labels_text_size)
    ylh = ylabel("Amplitude", "FontSize",16, "Rotation", 90, 'HorizontalAlignment','right', "FontWeight", "bold");
    ylh.Position(2) = 3.5;
    ylh.Position(1) = -220;

    % Shade the region before start_peak
    xregion([-inf, start_peak], 'FaceColor', [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');
    
    % Shade the region after end_peak
    xregion([end_peak, inf], 'FaceColor', [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');


    % high med etc
    subplot(3,3, 8);
set(gca, 'Position', pos5); % Adjust the size and position
     hold on;
    [~, peak_effect_idx] = min(abs(data{1}.time - line_plot_effect/1000));
    low_thick =  ci1_l.dist_thick_avg(peak_effect_idx);
    low_med =  ci1_l.dist_med_avg(peak_effect_idx);
    low_thin =  ci1_h.dist_thin_avg(peak_effect_idx);
    high_thick =  ci1_h.dist_thick_avg(peak_effect_idx);
    high_med =  ci1_h.dist_med_avg(peak_effect_idx);
    high_thin =  ci1_h.dist_thin_avg(peak_effect_idx);
    
    thick = [low_thick, high_thick];
    med = [low_med, high_med];
    thin = [low_thin, high_thin];

    plot(thick, '-o','color', '#FCD200' ,'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(thin,  '-o','color', '#0072BD' ,'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(med, '-o','color',  '#D95319','LineWidth', plot_line_width,'HandleVisibility','off');
    
    plot(NaN(1), 'Color', '#0072BD');
    plot(NaN(1), 'Color', '#D95319');
    plot(NaN(1), 'Color', '#FFFF00');
    %plot(NaN(1), 'Color', 'k');
    legend({'Thin', 'Medium', 'Thick'},'Location','northwest','FontSize', labels_text_size)
    

    % Customize x-axis labels
    xticks([1 2]);                 % Set x-tick positions
    xticklabels({'LOW', 'HIGH'});   % Replace numeric x-ticks with custom labels
    
    % Remove x-axis numeric labels
    set(gca, 'XTickLabelMode', 'manual', 'XTickMode', 'manual');
    grid on;

    subplot(3,3,9);
set(gca, 'Position', pos6); % Adjust the size and position
        hold on;
    
    plot(time, ci1_h.dist_thin_avg, 'color', '#0072BD', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, ci1_h.dist_thin_high, 'LineWidth', 0.01, 'color', '#0072BD','HandleVisibility','off');
    plot(time, ci1_h.dist_thin_low, 'LineWidth', 0.01, 'color', '#0072BD','HandleVisibility','off');
    x2 = [time, fliplr(time)];
    inBetween = [ci1_h.dist_thin_high, fliplr(ci1_h.dist_thin_low)];
    h = fill(x2, inBetween, [0, 0.447, 0.741], 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha',.10)

    plot(time, ci1_h.dist_med_avg, 'color', '#D95319', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, ci1_h.dist_med_high, 'LineWidth', 0.01, 'color', '#D95319','HandleVisibility','off');
    plot(time, ci1_h.dist_med_low, 'LineWidth', 0.01, 'color', '#D95319','HandleVisibility','off');
    x2 = [time, fliplr(time)];
    inBetween = [ci1_h.dist_med_high, fliplr(ci1_h.dist_med_low)];
    h = fill(x2, inBetween, [0.851, 0.325, 0.098], 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha',.10)

    plot(time, ci1_h.dist_thick_avg, 'color', '#FCD200', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, ci1_h.dist_thick_high, 'LineWidth', 0.01, 'color', '#FCD200','HandleVisibility','off');
    plot(time, ci1_h.dist_thick_low, 'LineWidth', 0.01, 'color', '#FCD200','HandleVisibility','off');
    x2 = [time, fliplr(time)];
    inBetween = [ci1_h.dist_thick_high, fliplr(ci1_h.dist_thick_low)];
    h = fill(x2, inBetween, [1,0.92,0], 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha',.175)

    xlim(plotting_window);

    ylim([-2, 8.5])
    grid on;
    hold off;

    if peak_effect ~= 0
        xline(start_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
        xline(end_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
        xline(peak_effect, '--r','HandleVisibility','off', "LineWidth", 1.5);
        xline(line_plot_effect, '--','HandleVisibility','off', "LineWidth", 1.5);
        xline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)
        yline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)
    end

    %ylabel(ax_label, "FontSize",labels_text_size)

    % Shade the region before start_peak
    xregion([-inf, start_peak], 'FaceColor', [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');
    
    % Shade the region after end_peak
    xregion([end_peak, inf], 'FaceColor', [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');


    %xlabel("Milliseconds", "FontSize",labels_text_size)
    %ylabel(ax_label, "FontSize",labels_text_size)




elseif strcmp(experiment_type, 'three-way-interaction')
    disp("....")
    figure;

    if strcmp(regression_type, "Visual Stress") 
        line_plot_effect = 226;
    elseif strcmp(regression_type, "Headache")
         line_plot_effect = 135;
    elseif strcmp(regression_type, "Discomfort")
        line_plot_effect = 200;
        end


    
    plot_thin_med_thick = 1;
    if plot_thin_med_thick
% Row 1 (with center plot)
pos1 = [0.1, 0.78, 0.3, 0.12];          % 0.1, 0.8, 0.3, 0.12
posMiddle1 = [0.4, 0.65, 0.2, 0.25];  % Center plot, aligned with Rows 1 & 2
posMiddle1 = [0.42, 0.65, 0.16, 0.25];
pos3 = [0.6, 0.78, 0.3, 0.12];          % Row 1, Col 3 (wider to the left)

% Row 2 (no center plot)
pos4 = [0.1, 0.65, 0.3, 0.12];         % Row 2, Col 1 (wider to the right)
pos5 = [0.6, 0.65, 0.3, 0.12];         % Row 2, Col 3 (wider to the left)

% Row 3 (with center plot)
pos6 = [0.1, 0.48, 0.3, 0.12];          % Row 3, Col 1 (wider to the right)
posMiddle3 = [0.4, 0.35, 0.2, 0.25];  % Center plot, aligned with Rows 3 & 4
posMiddle3 = [0.42, 0.35, 0.16, 0.25];  % Center plot, aligned with Rows 3 & 4
pos7 = [0.6, 0.48, 0.3, 0.12];          % Row 3, Col 3 (wider to the left)

% Row 4 (no center plot)
pos8 = [0.1, 0.35, 0.3, 0.12];         % Row 4, Col 1 (wider to the right)
pos9 = [0.6, 0.35, 0.3, 0.12];         % Row 4, Col 3 (wider to the left)

% Row 5 (with center plot)
pos10 = [0.1, 0.18, 0.3, 0.12];         % Row 5, Col 1 (wider to the right)
posMiddle5 = [0.4, 0.05, 0.2, 0.25];  % Center plot, aligned with Rows 5 & 6
posMiddle5 = [0.42, 0.05, 0.16, 0.25];  % Center plot, aligned with Rows 5 & 6
pos11 = [0.6, 0.18, 0.3, 0.12];         % Row 5, Col 3 (wider to the left)

% Row 6 (no center plot)
pos12 = [0.1, 0.05, 0.3, 0.12];        % Row 6, Col 1 (wider to the right)
pos13 = [0.6, 0.05, 0.3, 0.12];        % Row 6, Col 3 (wider to the left)

        % subplot("Position", pos1);
        % subplot("Position", posMiddle1);
        % subplot("Position", pos3);
        % subplot("Position", pos4);
        % subplot("Position", pos5);
        % subplot("Position", pos6);
        % subplot("Position", posMiddle3);
        % subplot("Position", pos7);
        % subplot("Position", pos8);
        % subplot("Position", pos9);
        % subplot("Position", pos10);
        % subplot("Position", posMiddle5);
        % subplot("Position", pos11);
        % subplot("Position", pos12);
        % subplot("Position", pos13);
    end

    
    labels_text_size = 8;
    t = tiledlayout(6,2, 'TileSpacing','Compact');
    time = data{1}.time * 1000;

    
    %% %%%%% PARTITION 1 LOW PANEL PGI
    % onsets 2;3, 4;5 6,7 for P1 low
    %subplot(6, 3,1);
    subplot("Position", pos1)
    set(gca, 'XTick', []);
    %set(gca, 'Position', pos1); % Adjust the size and position
    hold on;
    %plot(NaN(1), 'Color', 'r');
    %plot(NaN(1), 'Color', 'g');
    %plot(NaN(1), 'Color', 'b');
    %legend({'Onsets 2,3', 'Onsets 4,5', 'Onsets 6,7'},'Location','bestoutside','FontSize', labels_text_size)

        % onsets 2;3 p1
    plot(time, p1_23_l.dist_pgi_avg, 'color', 'r', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, p1_23_l.dist_pgi_high, 'color', 'r', 'LineWidth', 0.00001,'HandleVisibility','off');
    plot(time, p1_23_l.dist_pgi_low, 'color', 'r', 'LineWidth', 0.00001,'HandleVisibility','off');
     x2 = [time, fliplr(time)];
     inBetween = [p1_23_l.dist_pgi_high, fliplr(p1_23_l.dist_pgi_low)];
     h = fill(x2, inBetween, 'r', 'HandleVisibility','off', 'LineStyle','none');     set(h,'facealpha',.10)

        % onsets 4,5 p1 
    plot(time, p1_45_l.dist_pgi_avg, 'color', 'g', 'LineWidth', plot_line_width,'HandleVisibility','off');
     plot(time, p1_45_l.dist_pgi_high, 'color', 'g', 'LineWidth', 0.00001,'HandleVisibility','off');
     plot(time, p1_45_l.dist_pgi_low, 'color', 'g', 'LineWidth', 0.00001,'HandleVisibility','off');
     x2 = [time, fliplr(time)];
     inBetween = [p1_45_l.dist_pgi_high, fliplr(p1_45_l.dist_pgi_low)];
     h = fill(x2, inBetween, 'g', 'HandleVisibility','off', 'LineStyle','none');
     set(h,'facealpha',.10)  

        % onsets 6;7 p3
    plot(time, p1_67_l.dist_pgi_avg, 'color', 'b', 'LineWidth', plot_line_width,'HandleVisibility','off');
     plot(time, p1_67_l.dist_pgi_high, 'color', 'b', 'LineWidth', 0.00001,'HandleVisibility','off');
     plot(time, p1_67_l.dist_pgi_low, 'color', 'b', 'LineWidth', 0.00001,'HandleVisibility','off');
     x2 = [time, fliplr(time)];
     inBetween = [p1_67_l.dist_pgi_high, fliplr(p1_67_l.dist_pgi_low)];
     h = fill(x2, inBetween, 'b', 'HandleVisibility','off', 'LineStyle','none');
     set(h,'facealpha',.10) 

    xlim(plotting_window);
    ylim([-4, 12])
    grid on;
    title("LOW", "FontSize",18)
    hold off;

    xline(start_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
    xline(end_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
    xline(peak_effect, '--r','HandleVisibility','off', "LineWidth", 1.5);
    xline(line_plot_effect, '--','HandleVisibility','off', "LineWidth", 1.5);
    xline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)
    yline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)


    %xlabel("Milliseconds", "FontSize",labels_text_size)
    %ylabel(ax_label, "FontSize",labels_text_size)
    ylh = ylabel("P1 - PGI", "FontSize",12, "Rotation", 90, 'HorizontalAlignment','right', "FontWeight", "bold");
    ylh.Position(2) = 10;
    ylh.Position(1) = -220;

    %yyaxis right
    %yticks([])
    %ylabel(ax_label, "FontSize",labels_text_size, 'Color', 'black')

    % Shade the region before start_peak
    xregion([-inf, start_peak], 'FaceColor', [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');
    
    % Shade the region after end_peak
    xregion([end_peak, inf], 'FaceColor', [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');

    set(gca, 'XTick', []);
    
    %% %%%%% PARTITION 1 INTERACTION PANEL PGI
    subplot("Position", posMiddle1);
    hold on;
    
    % Find the index of the peak effect for plotting
    [~, peak_effect_idx] = min(abs(data{1}.time - line_plot_effect / 1000));
    
    % Extract data values
    p1_23_low = p1_23_l.dist_pgi_avg(peak_effect_idx);
    p1_23_high = p1_23_h.dist_pgi_avg(peak_effect_idx);
    p1_45_low = p1_45_l.dist_pgi_avg(peak_effect_idx);
    p1_45_high = p1_45_h.dist_pgi_avg(peak_effect_idx);
    p1_67_low = p1_67_l.dist_pgi_avg(peak_effect_idx);
    p1_67_high = p1_67_h.dist_pgi_avg(peak_effect_idx);

    % Data for partitions
    partitions_1 = [p1_23_low, p1_23_high];
    partitions_2 = [p1_45_low, p1_45_high];
    partitions_3 = [p1_67_low, p1_67_high];
    
    % Plot each partition with different colors
    plot(partitions_1, '-o', 'Color', 'r', 'LineWidth', plot_line_width, 'HandleVisibility', 'off');
    plot(partitions_2, '-o', 'Color', 'g', 'LineWidth', plot_line_width, 'HandleVisibility', 'off');
    plot(partitions_3, '-o', 'Color', 'b', 'LineWidth', plot_line_width, 'HandleVisibility', 'off');
    
    % Customize grid and axis settings
    grid on;                           % Enable gridlines
    set(gca, 'XTick', []); % Disable ticks
    
    % Set y-axis limits
    ylim([-1, 4]);
    
    % Title
    title("P1 - LOW vs. HIGH", 'FontSize', 18);
    %% %%%%% PARTITION 1 HIGH PANEL PGI
   % subplot(6, 3,3);
    %set(gca, 'Position', pos3); % Adjust the size and position
    subplot("Position", pos3)
    set(gca, 'YTick', []);
    hold on;
    plot(NaN(1), 'Color', 'r');
    plot(NaN(1), 'Color', 'g');
    plot(NaN(1), 'Color', 'b');
    legend({'Onsets 2,3', 'Onsets 4,5', 'Onsets 6,7'},'Location','northwest','FontSize', labels_text_size)

        % onsets 2;3 p1
    plot(time, p1_23_h.dist_pgi_avg, 'color', 'r', 'LineWidth', plot_line_width,'HandleVisibility','off');
     plot(time, p1_23_h.dist_pgi_high, 'color', 'r', 'LineWidth', 0.00001,'HandleVisibility','off');
     plot(time, p1_23_h.dist_pgi_low, 'color', 'r', 'LineWidth', 0.00001,'HandleVisibility','off');
     x2 = [time, fliplr(time)];
     inBetween = [p1_23_h.dist_pgi_high, fliplr(p1_23_h.dist_pgi_low)];
     h = fill(x2, inBetween, 'r', 'HandleVisibility','off', 'LineStyle','none');
     set(h,'facealpha',.10)

        % onsets 4,5 p1 
    plot(time, p1_45_h.dist_pgi_avg, 'color', 'g', 'LineWidth', plot_line_width,'HandleVisibility','off');
     plot(time, p1_45_h.dist_pgi_high, 'color', 'g', 'LineWidth', 0.00001,'HandleVisibility','off');
     plot(time, p1_45_h.dist_pgi_low, 'color', 'g', 'LineWidth', 0.00001,'HandleVisibility','off');
     x2 = [time, fliplr(time)];
     inBetween = [p1_45_h.dist_pgi_high, fliplr(p1_45_h.dist_pgi_low)];
     h = fill(x2, inBetween, 'g', 'HandleVisibility','off', 'LineStyle','none');
     set(h,'facealpha',.10)  

        % onsets 6;7 p3
    plot(time, p1_67_h.dist_pgi_avg, 'color', 'b', 'LineWidth', plot_line_width,'HandleVisibility','off');
     plot(time, p1_67_h.dist_pgi_high, 'color', 'b', 'LineWidth', 0.00001,'HandleVisibility','off');
     plot(time, p1_67_h.dist_pgi_low, 'color', 'b', 'LineWidth', 0.00001,'HandleVisibility','off');
     x2 = [time, fliplr(time)];
     inBetween = [p1_67_h.dist_pgi_high, fliplr(p1_67_h.dist_pgi_low)];
     h = fill(x2, inBetween, 'b', 'HandleVisibility','off', 'LineStyle','none');
     set(h,'facealpha',.10) 

    xlim(plotting_window);
    ylim([-4, 12])
    grid on;
    title("HIGH","FontSize", 18)
    hold off;

    xline(start_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
    xline(end_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
    xline(peak_effect, '--r','HandleVisibility','off', "LineWidth", 1.5);
    xline(line_plot_effect, '--','HandleVisibility','off', "LineWidth", 1.5);
    xline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)
    yline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)

    %xlabel("Milliseconds", "FontSize",labels_text_size)
    %ylabel(ax_label, "FontSize",labels_text_size)

    % Shade the region before start_peak
    xregion([-inf, start_peak], 'FaceColor', [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');
    
    % Shade the region after end_peak
    xregion([end_peak, inf], 'FaceColor', [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');

    set(gca, 'XTick', []);
    %% %%%%% PARTITION 1 LOW PANEL MEDIUM
    %subplot(6, 3,4);
    %set(gca, 'Position', pos4); % Adjust the size and position
    subplot("Position", pos4);
    hold on;
    %plot(NaN(1), 'Color', 'r');
    %plot(NaN(1), 'Color', 'g');
    %plot(NaN(1), 'Color', 'b');
    %legend({'Onsets 2,3', 'Onsets 4,5', 'Onsets 6,7'},'Location','bestoutside','FontSize', labels_text_size)

        % onsets 2;3 p1
    plot(time, p1_23_l.dist_med_avg, 'color', 'r', 'LineWidth', plot_line_width,'HandleVisibility','off');
     plot(time, p1_23_l.dist_med_high, 'color', 'r', 'LineWidth', 0.00001,'HandleVisibility','off');
     plot(time, p1_23_l.dist_med_low, 'color', 'r', 'LineWidth', 0.00001,'HandleVisibility','off');
     x2 = [time, fliplr(time)];
     inBetween = [p1_23_l.dist_med_high, fliplr(p1_23_l.dist_med_low)];
     h = fill(x2, inBetween, 'r', 'HandleVisibility','off', 'LineStyle','none');
     set(h,'facealpha',.10)

        % onsets 4,5 p1 
    plot(time, p1_45_l.dist_med_avg, 'color', 'g', 'LineWidth', plot_line_width,'HandleVisibility','off');
     plot(time, p1_45_l.dist_med_high, 'color', 'g', 'LineWidth', 0.00001,'HandleVisibility','off');
     plot(time, p1_45_l.dist_med_low, 'color', 'g', 'LineWidth', 0.00001,'HandleVisibility','off');
     x2 = [time, fliplr(time)];
    inBetween = [p1_45_l.dist_med_high, fliplr(p1_45_l.dist_med_low)];
    h = fill(x2, inBetween, 'g', 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha',.10)  

        % onsets 6;7 p3
    plot(time, p1_67_l.dist_med_avg, 'color', 'b', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, p1_67_l.dist_med_high, 'color', 'b', 'LineWidth', 0.00001,'HandleVisibility','off');
    plot(time, p1_67_l.dist_med_low, 'color', 'b', 'LineWidth', 0.00001,'HandleVisibility','off');
    x2 = [time, fliplr(time)];
    inBetween = [p1_67_l.dist_med_high, fliplr(p1_67_l.dist_med_low)];
    h = fill(x2, inBetween, 'b', 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha',.10) 

    xlim(plotting_window);
    ylim([-4, 12])
    grid on;
    %title("Medium","FontSize", 13)
    hold off;

    xline(start_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
    xline(end_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
    xline(peak_effect, '--r','HandleVisibility','off', "LineWidth", 1.5);
    xline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)
    yline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)

    %xlabel("Milliseconds", "FontSize",labels_text_size)
    %ylabel(ax_label, "FontSize",labels_text_size)
    ylh = ylabel("P1 - Medium", "FontSize",12, "Rotation", 90, 'HorizontalAlignment','right', "FontWeight", "bold");
    ylh.Position(2) =12;
    ylh.Position(1) = -220;

    %yyaxis right
    %yticks([])
    %ylabel(ax_label, "FontSize",labels_text_size, 'Color', 'black')

    %%%%%%%
    % onsets 2;3, 4;5 6,7 for P1 medium high
    %set(gca,'Color',[.85,.85,.85])
        % Shade the region before start_peak
    % Shade the region before start_peak
    xregion([-inf, start_peak], 'FaceColor', [0.45 0.45 0.45], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');
    
    % Shade the region after end_peak
    xregion([end_peak, inf], 'FaceColor', [0.45 0.45 0.45], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');

    %% %%%%% PARTITION 1 HIGH PANEL MEDIUM
    %subplot(6, 3,5);
    %set(gca, 'Position', pos5); % Adjust the size and position
    subplot("Position", pos5);
    set(gca, 'YTick', []);
    hold on;
    plot(NaN(1), 'Color', 'r');
    plot(NaN(1), 'Color', 'g');
    plot(NaN(1), 'Color', 'b');
    legend({'Onsets 2,3', 'Onsets 4,5', 'Onsets 6,7'},'Location','northwest','FontSize', labels_text_size)

        % onsets 2;3 p1
    plot(time, p1_23_h.dist_med_avg, 'color', 'r', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, p1_23_h.dist_med_high, 'color', 'r', 'LineWidth', 0.00001,'HandleVisibility','off');
    plot(time, p1_23_h.dist_med_low, 'color', 'r', 'LineWidth', 0.00001,'HandleVisibility','off');
    x2 = [time, fliplr(time)];
    inBetween = [p1_23_h.dist_med_high, fliplr(p1_23_h.dist_med_low)];
    h = fill(x2, inBetween, 'r', 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha',.10)

        % onsets 4,5 p1 
    plot(time, p1_45_h.dist_med_avg, 'color', 'g', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, p1_45_h.dist_med_high, 'color', 'g', 'LineWidth', 0.00001,'HandleVisibility','off');
    plot(time, p1_45_h.dist_med_low, 'color', 'g', 'LineWidth', 0.00001,'HandleVisibility','off');
    x2 = [time, fliplr(time)];
    inBetween = [p1_45_h.dist_med_high, fliplr(p1_45_h.dist_med_low)];
    h = fill(x2, inBetween, 'g', 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha',.10)  

        % onsets 6;7 p3
    plot(time, p1_67_h.dist_med_avg, 'color', 'b', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, p1_67_h.dist_med_high, 'color', 'b', 'LineWidth', 0.00001,'HandleVisibility','off');
    plot(time, p1_67_h.dist_med_low, 'color', 'b', 'LineWidth', 0.00001,'HandleVisibility','off');
    x2 = [time, fliplr(time)];
    inBetween = [p1_67_h.dist_med_high, fliplr(p1_67_h.dist_med_low)];
    h = fill(x2, inBetween, 'b', 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha',.10) 

    xlim(plotting_window);
    ylim([-2, 8.5])
    grid on;
    %title("Medium","FontSize", 13)
    hold off;

    xline(start_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
    xline(end_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
    xline(peak_effect, '--r','HandleVisibility','off', "LineWidth", 1.5);
    xline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)
    yline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)

    %xlabel("Milliseconds", "FontSize",labels_text_size)
    %ylabel(ax_label, "FontSize",labels_text_size)
    xregion([-inf, start_peak], 'FaceColor', [0.45 0.45 0.45], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');
    
    % Shade the region after end_peak
    xregion([end_peak, inf], 'FaceColor', [0.45 0.45 0.45], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');
    
    %% %%%%% PARTITION 2 LOW PANEL PGI
    %subplot(6, 3,6);
    %set(gca, 'Position', pos6); % Adjust the size and position
    subplot("Position", pos6);
    set(gca, 'XTick', []);
    hold on;

    %plot(NaN(1), 'Color', 'r');
    %plot(NaN(1), 'Color', 'g');
    %plot(NaN(1), 'Color', 'b');
    %legend({'Onsets 2,3', 'Onsets 4,5', 'Onsets 6,7'},'Location','bestoutside','FontSize', labels_text_size)

        % onsets 2;3 p1
    plot(time, p2_23_l.dist_pgi_avg, 'color', 'r', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, p2_23_l.dist_pgi_high, 'color', 'r', 'LineWidth', 0.00001,'HandleVisibility','off');
    plot(time, p2_23_l.dist_pgi_low, 'color', 'r', 'LineWidth', 0.00001,'HandleVisibility','off');
    x2 = [time, fliplr(time)];
    inBetween = [p2_23_l.dist_pgi_high, fliplr(p2_23_l.dist_pgi_low)];
    h = fill(x2, inBetween, 'r', 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha',.10)

        % onsets 4,5 p1 
    plot(time, p2_45_l.dist_pgi_avg, 'color', 'g', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, p2_45_l.dist_pgi_high, 'color', 'g', 'LineWidth', 0.00001,'HandleVisibility','off');
    plot(time, p2_45_l.dist_pgi_low, 'color', 'g', 'LineWidth', 0.00001,'HandleVisibility','off');
    x2 = [time, fliplr(time)];
    inBetween = [p2_45_l.dist_pgi_high, fliplr(p2_45_l.dist_pgi_low)];
    h = fill(x2, inBetween, 'g', 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha',.10)  

        % onsets 6;7 p3
    plot(time, p2_67_l.dist_pgi_avg, 'color', 'b', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, p2_67_l.dist_pgi_high, 'color', 'b', 'LineWidth', 0.00001,'HandleVisibility','off');
    plot(time, p2_67_l.dist_pgi_low, 'color', 'b', 'LineWidth', 0.00001,'HandleVisibility','off');
    x2 = [time, fliplr(time)];
    inBetween = [p2_67_l.dist_pgi_high, fliplr(p2_67_l.dist_pgi_low)];
    h = fill(x2, inBetween, 'b', 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha',.10) 

    xlim(plotting_window);
    ylim([-4, 12])
    grid on;
    %title("PGI","FontSize", 13)
    hold off;

    xline(start_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
    xline(end_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
    xline(peak_effect, '--r','HandleVisibility','off', "LineWidth", 1.5);
    xline(line_plot_effect, '--','HandleVisibility','off', "LineWidth", 1.5);
    xline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)
    yline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)

    %xlabel("Milliseconds", "FontSize",labels_text_size)
    %ylabel(ax_label, "FontSize",labels_text_size)
    ylh = ylabel("P2 - PGI", "FontSize",12, "Rotation", 90, 'HorizontalAlignment','right', "FontWeight", "bold");
    ylh.Position(2) = 10;
    ylh.Position(1) = -220;

    %yyaxis right
    %yticks([])
    %ylabel(ax_label, "FontSize",labels_text_size, 'Color', 'black')
    
    %%%%%%%
    xregion([-inf, start_peak], 'FaceColor', [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');
    
    % Shade the region after end_peak
    xregion([end_peak, inf], 'FaceColor', [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');

    set(gca, 'XTick', []);
        %% %%%%% PARTITION 2 INTERACTION PANEL PGI
     %subplot(6, 3,7);
    %set(gca, 'Position', posMiddle3); % Adjust the size and position
    subplot("Position", posMiddle3);
    grid on;
    [~, peak_effect_idx] = min(abs(data{1}.time - line_plot_effect/1000));

    p1_23_low = p2_23_l.dist_pgi_avg(peak_effect_idx);
    p1_23_high = p2_23_h.dist_pgi_avg(peak_effect_idx);
    
    p1_45_low = p2_45_l.dist_pgi_avg(peak_effect_idx);
    p1_45_high = p2_45_h.dist_pgi_avg(peak_effect_idx);

    p1_67_low = p2_67_l.dist_pgi_avg(peak_effect_idx);
    p1_67_high = p2_67_h.dist_pgi_avg(peak_effect_idx);

    hold on;

    partitions_1 = [p1_23_low, p1_23_high];
    partitions_2 = [p1_45_low, p1_45_high];
    partitions_3 = [p1_67_low, p1_67_high];

    plot(partitions_1, '-o','color', 'r' ,'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(partitions_2,  '-o','color', 'g' ,'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(partitions_3, '-o','color','b','LineWidth', plot_line_width,'HandleVisibility','off');
    
    % Customize grid and axis settings
    grid on;                           % Enable gridlines
    set(gca, 'XTick', []); % Disable ticks
    
    % Set y-axis limits
    ylim([-1, 4]);
    
    % Title
    title("P2 - LOW vs. HIGH",'FontSize', 18);

    %% %%%%% PARTITION 2 HIGH PANEL PGI
    % onsets 2;3, 4;5 6,7 for P2 high
    %subplot(6, 3,8);
    %set(gca, 'Position', pos7); % Adjust the size and position
    subplot("Position", pos7);
    set(gca, 'XTick', []);
    set(gca, 'YTick', []);
    hold on;
    plot(NaN(1), 'Color', 'r');
    plot(NaN(1), 'Color', 'g');
    plot(NaN(1), 'Color', 'b');
    legend({'Onsets 2,3', 'Onsets 4,5', 'Onsets 6,7'},'Location','northwest','FontSize', labels_text_size)
    set(legend,'color','white');

        % onsets 2;3 p1
    plot(time, p2_23_h.dist_pgi_avg, 'color', 'r', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, p2_23_h.dist_pgi_high, 'color', 'r', 'LineWidth', 0.00001,'HandleVisibility','off');
    plot(time, p2_23_h.dist_pgi_low, 'color', 'r', 'LineWidth', 0.00001,'HandleVisibility','off');
    x2 = [time, fliplr(time)];
    inBetween = [p2_23_h.dist_pgi_high, fliplr(p2_23_h.dist_pgi_low)];
    h = fill(x2, inBetween, 'r', 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha',.10)

        % onsets 4,5 p1 
    plot(time, p2_45_h.dist_pgi_avg, 'color', 'g', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, p2_45_h.dist_pgi_high, 'color', 'g', 'LineWidth', 0.00001,'HandleVisibility','off');
    plot(time, p2_45_h.dist_pgi_low, 'color', 'g', 'LineWidth', 0.00001,'HandleVisibility','off');
    x2 = [time, fliplr(time)];
    inBetween = [p2_45_h.dist_pgi_high, fliplr(p2_45_h.dist_pgi_low)];
    h = fill(x2, inBetween, 'g', 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha',.10)  

        % onsets 6;7 p3
    plot(time, p2_67_h.dist_pgi_avg, 'color', 'b', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, p2_67_h.dist_pgi_high, 'color', 'b', 'LineWidth', 0.00001,'HandleVisibility','off');
    plot(time, p2_67_h.dist_pgi_low, 'color', 'b', 'LineWidth', 0.00001,'HandleVisibility','off');
    x2 = [time, fliplr(time)];
    inBetween = [p2_67_h.dist_pgi_high, fliplr(p2_67_h.dist_pgi_low)];
    h = fill(x2, inBetween, 'b', 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha',.10) 

    xlim(plotting_window);
    ylim([-4, 12])
    grid on;
    %title("PGI","FontSize", 13)
    hold off;

    xline(start_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
    xline(end_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
    xline(peak_effect, '--r','HandleVisibility','off', "LineWidth", 1.5);
    xline(line_plot_effect, '--','HandleVisibility','off', "LineWidth", 1.5);
    xline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)
    yline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)

    %xlabel("Milliseconds", "FontSize",labels_text_size)
    %ylabel(ax_label, "FontSize",labels_text_size)
    xregion([-inf, start_peak], 'FaceColor', [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');
    
    % Shade the region after end_peak
    xregion([end_peak, inf], 'FaceColor', [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');

    %% %%%%% PARTITION 2 LOW PANEL MEDIUM
    % onsets 2;3, 4;5 6,7 for P2 high
    %subplot(6, 3,9);
    %set(gca, 'Position', pos8); % Adjust the size and position
    subplot("Position", pos8);
    %set(gca,'Color',[.85,.85,.85])
    hold on;
    %plot(NaN(1), 'Color', 'r');
    %plot(NaN(1), 'Color', 'g');
    %plot(NaN(1), 'Color', 'b');
    %legend({'Onsets 2,3', 'Onsets 4,5', 'Onsets 6,7'},'Location','bestoutside','FontSize', labels_text_size)

        % onsets 2;3 p1
    plot(time, p2_23_l.dist_med_avg, 'color', 'r', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, p2_23_l.dist_med_high, 'color', 'r', 'LineWidth', 0.00001,'HandleVisibility','off');
    plot(time, p2_23_l.dist_med_low, 'color', 'r', 'LineWidth', 0.00001,'HandleVisibility','off');
    x2 = [time, fliplr(time)];
    inBetween = [p2_23_l.dist_med_high, fliplr(p2_23_l.dist_med_low)];
    h = fill(x2, inBetween, 'r', 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha',.10)

        % onsets 4,5 p1 
    plot(time, p2_45_l.dist_med_avg, 'color', 'g', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, p2_45_l.dist_med_high, 'color', 'g', 'LineWidth', 0.00001,'HandleVisibility','off');
    plot(time, p2_45_l.dist_med_low, 'color', 'g', 'LineWidth', 0.00001,'HandleVisibility','off');
    x2 = [time, fliplr(time)];
    inBetween = [p2_45_l.dist_med_high, fliplr(p2_45_l.dist_med_low)];
    h = fill(x2, inBetween, 'g', 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha',.10)  

        % onsets 6;7 p3
    plot(time, p2_67_l.dist_med_avg, 'color', 'b', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, p2_67_l.dist_med_high, 'color', 'b', 'LineWidth', 0.00001,'HandleVisibility','off');
    plot(time, p2_67_l.dist_med_low, 'color', 'b', 'LineWidth', 0.00001,'HandleVisibility','off');
    x2 = [time, fliplr(time)];
    inBetween = [p2_67_l.dist_med_high, fliplr(p2_67_l.dist_med_low)];
    h = fill(x2, inBetween, 'b', 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha',.10) 

    xlim(plotting_window);
    ylim([-4, 12])
    grid on;
    %title("Medium","FontSize", 13)
    hold off;

    xline(start_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
    xline(end_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
    xline(peak_effect, '--r','HandleVisibility','off', "LineWidth", 1.5);
    xline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)
    yline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)

    %xlabel("Milliseconds", "FontSize",labels_text_size)
    %ylabel(ax_label, "FontSize",labels_text_size)
    ylh = ylabel("P2 - Medium", "FontSize",12, "Rotation", 90, 'HorizontalAlignment','right', "FontWeight", "bold");
    ylh.Position(2) =12;
    ylh.Position(1) = -220;

    %yyaxis right
    %yticks([])
    %ylabel(ax_label, "FontSize",labels_text_size, 'Color', 'black')
    xregion([-inf, start_peak], 'FaceColor', [0.45 0.45 0.45], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');
    
    % Shade the region after end_peak
    xregion([end_peak, inf], 'FaceColor', [0.45 0.45 0.45], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');

    %% %%%%% PARTITION 2 HIGH PANEL MEDIUM
    % onsets 2;3, 4;5 6,7 for P2 high
    %subplot(6, 3,10);
    %set(gca, 'Position', pos9); % Adjust the size and position
    subplot("Position", pos9);
    set(gca, 'YTick', []);
    hold on;
    plot(NaN(1), 'Color', 'r');
    plot(NaN(1), 'Color', 'g');
    plot(NaN(1), 'Color', 'b');
    legend({'Onsets 2,3', 'Onsets 4,5', 'Onsets 6,7'},'Location','northwest','FontSize', labels_text_size)

        % onsets 2;3 p1
    plot(time, p2_23_h.dist_med_avg, 'color', 'r', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, p2_23_h.dist_med_high, 'color', 'r', 'LineWidth', 0.00001,'HandleVisibility','off');
    plot(time, p2_23_h.dist_med_low, 'color', 'r', 'LineWidth', 0.00001,'HandleVisibility','off');
    x2 = [time, fliplr(time)];
    inBetween = [p2_23_h.dist_med_high, fliplr(p2_23_h.dist_med_low)];
    h = fill(x2, inBetween, 'r', 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha',.10)

        % onsets 4,5 p1 
    plot(time, p2_45_h.dist_med_avg, 'color', 'g', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, p2_45_h.dist_med_high, 'color', 'g', 'LineWidth', 0.00001,'HandleVisibility','off');
    plot(time, p2_45_h.dist_med_low, 'color', 'g', 'LineWidth', 0.00001,'HandleVisibility','off');
    x2 = [time, fliplr(time)];
    inBetween = [p2_45_h.dist_med_high, fliplr(p2_45_h.dist_med_low)];
    h = fill(x2, inBetween, 'g', 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha',.10)  

        % onsets 6;7 p3
    plot(time, p2_67_h.dist_med_avg, 'color', 'b', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, p2_67_h.dist_med_high, 'color', 'b', 'LineWidth', 0.00001,'HandleVisibility','off');
    plot(time, p2_67_h.dist_med_low, 'color', 'b', 'LineWidth', 0.00001,'HandleVisibility','off');
    x2 = [time, fliplr(time)];
    inBetween = [p2_67_h.dist_med_high, fliplr(p2_67_h.dist_med_low)];
    h = fill(x2, inBetween, 'b', 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha',.10) 

    xlim(plotting_window);
    ylim([-4, 12])
    grid on;
    %title("Medium","FontSize", 13)
    hold off;

    xline(start_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
    xline(end_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
    xline(peak_effect, '--r','HandleVisibility','off', "LineWidth", 1.5);
    xline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)
    yline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)

    %xlabel("Milliseconds", "FontSize",labels_text_size)
    %ylabel(ax_label, "FontSize",labels_text_size)
    xregion([-inf, start_peak], 'FaceColor', [0.45 0.45 0.45], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');
    
    % Shade the region after end_peak
    xregion([end_peak, inf], 'FaceColor', [0.45 0.45 0.45], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');

    %% %%%%% PARTITION 3 LOW PANEL PGI
    % onsets 2;3, 4;5 6,7 for P2 high
    %subplot(6, 3,11);
    %set(gca, 'Position', pos10); % Adjust the size and position
    subplot("Position", pos10);
    hold on;
    %plot(NaN(1), 'Color', 'r');
    %plot(NaN(1), 'Color', 'g');
    %plot(NaN(1), 'Color', 'b');
    %legend({'Onsets 2,3', 'Onsets 4,5', 'Onsets 6,7'},'Location','bestoutside','FontSize', labels_text_size)

        % onsets 2;3 p1
    plot(time, p3_23_l.dist_pgi_avg, 'color', 'r', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, p3_23_l.dist_pgi_high, 'color', 'r', 'LineWidth', 0.00001,'HandleVisibility','off');
    plot(time, p3_23_l.dist_pgi_low, 'color', 'r', 'LineWidth', 0.00001,'HandleVisibility','off');
    x2 = [time, fliplr(time)];
    inBetween = [p3_23_l.dist_pgi_high, fliplr(p3_23_l.dist_pgi_low)];
    h = fill(x2, inBetween, 'r', 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha',.10)

        % onsets 4,5 p1 
    plot(time, p3_45_l.dist_pgi_avg, 'color', 'g', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, p3_45_l.dist_pgi_high, 'color', 'g', 'LineWidth', 0.00001,'HandleVisibility','off');
    plot(time, p3_45_l.dist_pgi_low, 'color', 'g', 'LineWidth', 0.00001,'HandleVisibility','off');
    x2 = [time, fliplr(time)];
    inBetween = [p3_45_l.dist_pgi_high, fliplr(p3_45_l.dist_pgi_low)];
    h = fill(x2, inBetween, 'g', 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha',.10)  

        % onsets 6;7 p3
    plot(time, p3_67_l.dist_pgi_avg, 'color', 'b', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, p3_67_l.dist_pgi_high, 'color', 'b', 'LineWidth', 0.00001,'HandleVisibility','off');
    plot(time, p3_67_l.dist_pgi_low, 'color', 'b', 'LineWidth', 0.00001,'HandleVisibility','off');
    x2 = [time, fliplr(time)];
    inBetween = [p3_67_l.dist_pgi_high, fliplr(p3_67_l.dist_pgi_low)];
    h = fill(x2, inBetween, 'b', 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha',.10) 

    xlim(plotting_window);
    ylim([-4, 12])
    grid on;
    %title("PGI","FontSize", 13)
    hold off;

    xline(start_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
    xline(end_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
    xline(peak_effect, '--r','HandleVisibility','off', "LineWidth", 1.5);
    xline(line_plot_effect, '--','HandleVisibility','off', "LineWidth", 1.5);
    xline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)
    yline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)

    %xlabel("Milliseconds", "FontSize",labels_text_size)
    %ylabel(ax_label, "FontSize",labels_text_size)
    ylh = ylabel("P3 - PGI", "FontSize",12, "Rotation", 90, 'HorizontalAlignment','right', "FontWeight", "bold");
    ylh.Position(2) = 9;
    ylh.Position(1) = -220;

    %yyaxis right
    %yticks([])
    %ylabel(ax_label, "FontSize",labels_text_size, 'Color', 'black')
    
    %%%%%%%
    xregion([-inf, start_peak], 'FaceColor', [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');
    
    % Shade the region after end_peak
    xregion([end_peak, inf], 'FaceColor', [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');

    set(gca, 'XTick', []);

    %% %%%%% PARTITION 3 INTERACTION PANEL PGI
    % onsets 2;3, 4;5 6,7 for P2 high
    %subplot(6, 3,12);
    %set(gca, 'Position', posMiddle5); % Adjust the size and position
    subplot("Position", posMiddle5);
    hold on ;

    [~, peak_effect_idx] = min(abs(data{1}.time - line_plot_effect/1000));

    p1_23_low = p3_23_l.dist_pgi_avg(peak_effect_idx);
    p1_23_high = p3_23_h.dist_pgi_avg(peak_effect_idx);
    
    p1_45_low = p3_45_l.dist_pgi_avg(peak_effect_idx);
    p1_45_high = p3_45_h.dist_pgi_avg(peak_effect_idx);

    p1_67_low = p3_67_l.dist_pgi_avg(peak_effect_idx);
    p1_67_high = p3_67_h.dist_pgi_avg(peak_effect_idx);

    partitions_1 = [p1_23_low, p1_23_high];
    partitions_2 = [p1_45_low, p1_45_high];
    partitions_3 = [p1_67_low, p1_67_high];

    plot(partitions_1, '-o','color', 'r' ,'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(partitions_2,  '-o','color', 'g' ,'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(partitions_3, '-o','color','b','LineWidth', plot_line_width,'HandleVisibility','off');
        
    % Customize grid and axis settings
    grid on;                           % Enable gridlines
    set(gca, 'XTick', []); % Disable ticks
    
    % Set y-axis limits
    ylim([-1, 4]);
    
    % Title
    title("P3 - LOW vs. HIGH",'FontSize', 18);

    %% %%%%% PARTITION 3 HIGH PANEL PGI
    % onsets 2;3, 4;5 6,7 for P2 high
    %subplot(6, 3,13);
    %set(gca, 'Position', pos11); % Adjust the size and position
    subplot("Position", pos11);
    set(gca, 'YTick', []);
    set(gca, 'XTick', []);
    hold on;
    plot(NaN(1), 'Color', 'r');
    plot(NaN(1), 'Color', 'g');
    plot(NaN(1), 'Color', 'b');
    legend({'Onsets 2,3', 'Onsets 4,5', 'Onsets 6,7'},'Location','northwest','FontSize', labels_text_size)

        % onsets 2;3 p1
    plot(time, p3_23_h.dist_pgi_avg, 'color', 'r', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, p3_23_h.dist_pgi_high, 'color', 'r', 'LineWidth', 0.00001,'HandleVisibility','off');
    plot(time, p3_23_h.dist_pgi_low, 'color', 'r', 'LineWidth', 0.00001,'HandleVisibility','off');
    x2 = [time, fliplr(time)];
    inBetween = [p3_23_h.dist_pgi_high, fliplr(p3_23_h.dist_pgi_low)];
    h = fill(x2, inBetween, 'r', 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha',.10)

        % onsets 4,5 p1 
    plot(time, p3_45_h.dist_pgi_avg, 'color', 'g', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, p3_45_h.dist_pgi_high, 'color', 'g', 'LineWidth', 0.00001,'HandleVisibility','off');
    plot(time, p3_45_h.dist_pgi_low, 'color', 'g', 'LineWidth', 0.00001,'HandleVisibility','off');
    x2 = [time, fliplr(time)];
    inBetween = [p3_45_h.dist_pgi_high, fliplr(p3_45_h.dist_pgi_low)];
    h = fill(x2, inBetween, 'g', 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha',.10)  

        % onsets 6;7 p3
    plot(time, p3_67_h.dist_pgi_avg, 'color', 'b', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, p3_67_h.dist_pgi_high, 'color', 'b', 'LineWidth', 0.00001,'HandleVisibility','off');
    plot(time, p3_67_h.dist_pgi_low, 'color', 'b', 'LineWidth', 0.00001,'HandleVisibility','off');
    x2 = [time, fliplr(time)];
    inBetween = [p3_67_h.dist_pgi_high, fliplr(p3_67_h.dist_pgi_low)];
    h = fill(x2, inBetween, 'b', 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha',.10) 

    xlim(plotting_window);
    ylim([-4, 12])
    grid on;
    %title("PGI","FontSize", 13)
    hold off;

    xline(start_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
    xline(end_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
    xline(peak_effect, '--r','HandleVisibility','off', "LineWidth", 1.5);
    xline(line_plot_effect, '--','HandleVisibility','off', "LineWidth", 1.5);
    xline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)
    yline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)

    %xlabel("Milliseconds", "FontSize",labels_text_size)
    %ylabel(ax_label, "FontSize",labels_text_size)
    xregion([-inf, start_peak], 'FaceColor', [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');
    
    % Shade the region after end_peak
    xregion([end_peak, inf], 'FaceColor', [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');

    set(gca, 'XTick', []);

    %% %%%%% PARTITION 3 LOW PANEL MEDIUM
    % onsets 2;3, 4;5 6,7 for P2 high
    %subplot(6, 3,13);
    %set(gca, 'Position', pos12); % Adjust the size and position
    subplot("Position", pos12);
    hold on;
    %plot(NaN(1), 'Color', 'r');
    %plot(NaN(1), 'Color', 'g');
    %plot(NaN(1), 'Color', 'b');
    %legend({'Onsets 2,3', 'Onsets 4,5', 'Onsets 6,7'},'Location','bestoutside','FontSize', labels_text_size)

        % onsets 2;3 p1
    plot(time, p3_23_l.dist_med_avg, 'color', 'r', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, p3_23_l.dist_med_high, 'color', 'r', 'LineWidth', 0.00001,'HandleVisibility','off');
    plot(time, p3_23_l.dist_med_low, 'color', 'r', 'LineWidth', 0.00001,'HandleVisibility','off');
    x2 = [time, fliplr(time)];
    inBetween = [p3_23_l.dist_med_high, fliplr(p3_23_l.dist_med_low)];
    h = fill(x2, inBetween, 'r', 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha',.10)

        % onsets 4,5 p1 
    plot(time, p3_45_l.dist_med_avg, 'color', 'g', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, p3_45_l.dist_med_high, 'color', 'g', 'LineWidth', 0.00001,'HandleVisibility','off');
    plot(time, p3_45_l.dist_med_low, 'color', 'g', 'LineWidth', 0.00001,'HandleVisibility','off');
    x2 = [time, fliplr(time)];
    inBetween = [p3_45_l.dist_med_high, fliplr(p3_45_l.dist_med_low)];
    h = fill(x2, inBetween, 'g', 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha',.10)  

        % onsets 6;7 p3
    plot(time, p3_67_l.dist_med_avg, 'color', 'b', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, p3_67_l.dist_med_high, 'color', 'b', 'LineWidth', 0.00001,'HandleVisibility','off');
    plot(time, p3_67_l.dist_med_low, 'color', 'b', 'LineWidth', 0.00001,'HandleVisibility','off');
    x2 = [time, fliplr(time)];
    inBetween = [p3_67_l.dist_med_high, fliplr(p3_67_l.dist_med_low)];
    h = fill(x2, inBetween, 'b', 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha',.10) 

    xlim(plotting_window);
    ylim([-4, 12])
    grid on;
    %title("Medium","FontSize", 13)
    hold off;

    xline(start_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
    xline(end_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
    xline(peak_effect, '--r','HandleVisibility','off', "LineWidth", 1.5);
    xline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)
    yline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)

    %xlabel("Milliseconds", "FontSize",labels_text_size)
    %ylabel(ax_label, "FontSize",labels_text_size)
    ylh = ylabel("P3 - Medium", "FontSize",12, "Rotation", 90, 'HorizontalAlignment','right', "FontWeight", "bold");
    ylh.Position(2) =11;
    ylh.Position(1) = -220;

   % yyaxis right
    %yticks([])
    %ylabel(ax_label, "FontSize",labels_text_size, 'Color', 'black')

    %%%%%%%
    % onsets 2;3, 4;5 6,7 for P3 medium high
    %set(gca,'Color',[.85,.85,.85])
        xregion([-inf, start_peak], 'FaceColor', [0.45 0.45 0.45], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');
    
    % Shade the region after end_peak
    xregion([end_peak, inf], 'FaceColor', [0.45 0.45 0.45], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');

    %% %%%%% PARTITION 3 HIGH PANEL MEDIUM
    % onsets 2;3, 4;5 6,7 for P2 high
    %subplot(6, 3,14);
   % set(gca, 'Position', pos13); % Adjust the size and position
        subplot("Position", pos13);
    set(gca, 'YTick', []);
    hold on;
    plot(NaN(1), 'Color', 'r');
    plot(NaN(1), 'Color', 'g');
    plot(NaN(1), 'Color', 'b');
    legend({'Onsets 2,3', 'Onsets 4,5', 'Onsets 6,7'},'Location','northwest','FontSize', labels_text_size)

        % onsets 2;3 p1
    plot(time, p3_23_h.dist_med_avg, 'color', 'r', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, p3_23_h.dist_med_high, 'color', 'r', 'LineWidth', 0.00001,'HandleVisibility','off');
    plot(time, p3_23_h.dist_med_low, 'color', 'r', 'LineWidth', 0.00001,'HandleVisibility','off');
    x2 = [time, fliplr(time)];
    inBetween = [p3_23_h.dist_med_high, fliplr(p3_23_h.dist_med_low)];
    h = fill(x2, inBetween, 'r', 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha',.10)

        % onsets 4,5 p1 
    plot(time, p3_45_h.dist_med_avg, 'color', 'g', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, p3_45_h.dist_med_high, 'color', 'g', 'LineWidth', 0.00001,'HandleVisibility','off');
    plot(time, p3_45_h.dist_med_low, 'color', 'g', 'LineWidth', 0.00001,'HandleVisibility','off');
    x2 = [time, fliplr(time)];
    inBetween = [p3_45_h.dist_med_high, fliplr(p3_45_h.dist_med_low)];
    h = fill(x2, inBetween, 'g', 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha',.10)  

        % onsets 6;7 p3
    plot(time, p3_67_h.dist_med_avg, 'color', 'b', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, p3_67_h.dist_med_high, 'color', 'b', 'LineWidth', 0.00001,'HandleVisibility','off');
    plot(time, p3_67_h.dist_med_low, 'color', 'b', 'LineWidth', 0.00001,'HandleVisibility','off');
    x2 = [time, fliplr(time)];
    inBetween = [p3_67_h.dist_med_high, fliplr(p3_67_h.dist_med_low)];
    h = fill(x2, inBetween, 'b', 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha',.10) 

    xlim(plotting_window);
    ylim([-4, 12])
    grid on;
    %title("Medium","FontSize", 13)
    %set(gca,'Color',[.85,.85,.85])
    hold off;

    xline(start_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
    xline(end_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
    xline(peak_effect, '--r','HandleVisibility','off', "LineWidth", 1.5);
    xline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)
    yline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)

    %xlabel("Milliseconds", "FontSize",labels_text_size)
    %ylabel(ax_label, "FontSize",labels_text_size)
    
    %set(gca,'Color',[.85,.85,.85])
        xregion([-inf, start_peak], 'FaceColor', [0.45 0.45 0.45], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');
    
    % Shade the region after end_peak
    xregion([end_peak, inf], 'FaceColor', [0.45 0.45 0.45], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');


elseif strcmp(experiment_type, 'partitions-2-8') && first_partition_regresion == 1 || ...
    strcmp(experiment_type,'pure-factor-effect')
    figure;
    if plot_thin_med_thick        
        % Row 1
        pos1 = [0.05, 0.55, 0.35, 0.35];  % First plot (larger width)
        pos2 = [0.425, 0.55, 0.15, 0.35];  % Second plot (half width)
        pos3 = [0.6, 0.55, 0.35, 0.35];    % Third plot (larger width)
        
        % Row 2
        pos4 = [0.05, 0.5, 0.35, 0.35];    % First plot (larger width)
        pos5 = [0.425, 0.5, 0.15, 0.35];   % Second plot (half width)
        pos6 = [0.6, 0.5, 0.35, 0.35];     % Third plot (larger width)
    else
            % Row 1
        % Define the positions for the plots, keeping (2,1) and (2,3) empty
    % Define the positions for the plots, keeping (2,1) and (2,3) empty
    % Increasing width for pos1, pos3, pos4, and pos6
    pos1 = [0.05, 0.6, 0.325, 0.25];   % Row 1, Col 1 (wider)
    pos2 = [0.415, 0.6, 0.125, 0.25];  % Row 1, Col 2 (narrower)
    pos3 = [0.57, 0.6, 0.325, 0.25];   % Row 1, Col 3 (wider)
    
    posMiddle = [0.415, 0.39, 0.125, 0.15]; % Row 2, Col 2 (centered and narrower)
    
    pos4 = [0.05, 0.1, 0.325, 0.25];   % Row 3, Col 1 (wider)
    pos5 = [0.415, 0.1, 0.125, 0.25];  % Row 3, Col 2 (narrower)
    pos6 = [0.57, 0.1, 0.325, 0.25];   % Row 3, Col 3 (wider)
    end
    
    labels_text_size = 8;


  
    subplot(3,3,1);
    set(gca, 'Position', pos1); % Adjust the size and position
    time = data{1}.time * 1000;

        % Shade the region before start_peak
    xregion([-inf, start_peak], 'FaceColor', [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');
    
    % Shade the region after end_peak
    xregion([end_peak, inf], 'FaceColor', [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');

    line_plot_effect =174.00;

    hold on;
    %plot(NaN(1), 'r');
    %if contains(experiment_type, 'partitions-2-8')
    %    legend({'P1-PGI'},'Location','bestoutside','FontSize', labels_text_size)
    %elseif contains(experiment_type, 'pure-factor-effect')
    %    legend({'PGI'},'Location','bestoutside','FontSize', labels_text_size)
    %end

    % low
    plot(time, ci1_l.dist_pgi_avg, 'color', 'r', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, ci1_l.dist_pgi_high, 'LineWidth', 0.00001, 'color', 'r','HandleVisibility','off');
    plot(time, ci1_l.dist_pgi_low, 'LineWidth', 0.00001, 'color', 'r','HandleVisibility','off');
    x2 = [time, fliplr(time)];
    inBetween = [ci1_l.dist_pgi_high, fliplr(ci1_l.dist_pgi_low)];
    h = fill(x2, inBetween, 'r', 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha',.10)

    xlim(plotting_window);

    title("LOW",'FontSize', 18);
    ylim([-2, 6])
    grid on;
    hold off;

    if peak_effect ~= 0
        xline(start_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
        xline(end_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
        xline(peak_effect, '--r','HandleVisibility','off', "LineWidth", 1.5);
         xline(line_plot_effect, '--','HandleVisibility','off', "LineWidth", 1.5);
        xline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)
        yline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)
    end

    %xlabel("Milliseconds", "FontSize",labels_text_size)
    %ylabel(ax_label, "FontSize",labels_text_size)

    ylh = ylabel("PGI", "FontSize",16, "Rotation", 90, 'HorizontalAlignment','right', "FontWeight", "bold");
    ylh.Position(2) = 2;
    ylh.Position(1) = -220;

    subplot(3,3,2);
    set(gca, 'Position', pos2);
    hold on;
    plot(NaN(1), 'r');
    plot(NaN(1), 'k');
    if contains(experiment_type, 'partitions-2-8')
        legend({'P1-PGI'},'Location','northwest','FontSize', labels_text_size)
    elseif contains(experiment_type, 'pure-factor-effect')
        legend({'PGI'},'Location','northwest','FontSize', labels_text_size)
    end

    % interaction plot
    [~, peak_effect_idx] = min(abs(data{1}.time - line_plot_effect/1000));
    p1_high =  ci1_h.dist_pgi_avg(peak_effect_idx);
    p1_low =  ci1_l.dist_pgi_avg(peak_effect_idx);

    % thin thick average 
     %p1_thin_high = ci1_h.dist_thin_avg(peak_effect_idx);
     %p1_thick_high = ci1_h.dist_thick_avg(peak_effect_idx);
     %p1_thin_low = ci1_l.dist_thin_avg(peak_effect_idx);
     %p1_thick_low = ci1_l.dist_thick_avg(peak_effect_idx);

    %mean_thin_thick_low = mean([p1_thin_low, p1_thick_low]);
    %mean_thin_thick_high = mean([p1_thick_high, p1_thin_high]);

    partitions_1 = [p1_low, p1_high];
    %avg_thin_thick = [mean_thin_thick_low, mean_thin_thick_high];

    plot(partitions_1, '-o','color', 'r' ,'LineWidth', plot_line_width,'HandleVisibility','off');
    %plot(avg_thin_thick, '-o','color', 'k' ,'LineWidth', plot_line_width,'HandleVisibility','off');

    title("LOW vs. HIGH",'FontSize', 18);

    % Customize x-axis labels
    xticks([1 2]);                 % Set x-tick positions
    xticklabels({'LOW', 'HIGH'});   % Replace numeric x-ticks with custom labels
    
    % Remove x-axis numeric labels
    set(gca, 'XTickLabelMode', 'manual', 'XTickMode', 'manual');
      grid on;
    % interaction plot end

    subplot(3,3,3);
    set(gca, 'Position', pos3);
    hold on;


       % Shade the region before start_peak
    xregion([-inf, start_peak], 'FaceColor', [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');
    
    % Shade the region after end_peak
    xregion([end_peak, inf], 'FaceColor', [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');

    hold on;
    plot(time, ci1_h.dist_pgi_avg, 'color', 'r', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, ci1_h.dist_pgi_high, 'LineWidth', 0.00001, 'color', 'r','HandleVisibility','off');
    plot(time, ci1_h.dist_pgi_low, 'LineWidth', 0.00001, 'color', 'r','HandleVisibility','off');
    x2 = [time, fliplr(time)];
    inBetween = [ci1_h.dist_pgi_high, fliplr(ci1_h.dist_pgi_low)];
    h = fill(x2, inBetween, 'r', 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha',.10)

    xlim(plotting_window);

    title("HIGH",'FontSize', 18);
    ylim([-2, 6])
    grid on;
    hold off;

    if peak_effect ~= 0
        xline(start_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
        xline(end_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
        xline(peak_effect, '--r','HandleVisibility','off', "LineWidth", 1.5);
        xline(line_plot_effect, '--','HandleVisibility','off', "LineWidth", 1.5);
        xline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)
        yline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)
    end

    %xlabel("Milliseconds", "FontSize",labels_text_size)
   % ylabel(ax_label, "FontSize",labels_text_size)

    %% subplot middle
    subplot('position', posMiddle);
     hold on;

    plot(NaN(1), 'Color', '#D95319');
    plot(NaN(1), 'Color', 'g');
    if contains(experiment_type, 'partitions-2-8')
        legend({'P1-PGI'},'Location','northwest','FontSize', labels_text_size)
    elseif contains(experiment_type, 'pure-factor-effect')
        legend({'Medium', 'Avg. (Thin, Thick)'},'Location','northwest','FontSize', labels_text_size)
    end

    [~, peak_effect_idx] = min(abs(data{1}.time - line_plot_effect/1000));
    low_med =  ci1_l.dist_med_avg(peak_effect_idx);
    
    low_thin =  ci1_h.dist_thin_avg(peak_effect_idx);
    low_thick =  ci1_l.dist_thick_avg(peak_effect_idx);
    low_thin_thick_avg = mean([low_thin, low_thick]);

    high_med =  ci1_h.dist_med_avg(peak_effect_idx);

    high_thick =  ci1_h.dist_thick_avg(peak_effect_idx);
    high_thin =  ci1_h.dist_thin_avg(peak_effect_idx);
    high_thin_thick_avg = mean([high_thin, high_thick]);
    
    med = [low_med, high_med];
    avg_thin_thick = [low_thin_thick_avg, high_thin_thick_avg];
    

    plot(med, '-o','color', '#D95319' ,'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(avg_thin_thick,  '-o','color', 'g' ,'LineWidth', plot_line_width,'HandleVisibility','off');


    % Customize x-axis labels
    xticks([1 2]);                 % Set x-tick positions
    %xticklabels({'LOW', 'HIGH'});   % Replace numeric x-ticks with custom labels
    
    % Remove x-axis numeric labels
    set(gca, 'XTickLabelMode', 'manual', 'XTickMode', 'manual');
    grid on;


    %% subplot middle

    % low med etc
    subplot(3, 3, 7);
    set(gca, 'Position', pos4);
    hold on;
    %plot(NaN(1), 'Color', '#0072BD');
    %plot(NaN(1), 'Color', '#D95319');
    %plot(NaN(1), 'Color', '#FFFF00');
    %legend({'Thin', 'Medium', 'Thick'},'Location','bestoutside','FontSize', labels_text_size)

    plot(time, ci1_l.dist_thin_avg, 'color', '#0072BD', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, ci1_l.dist_thin_high, 'LineWidth', 0.01, 'color', '#0072BD','HandleVisibility','off');
    plot(time, ci1_l.dist_thin_low, 'LineWidth', 0.01, 'color', '#0072BD','HandleVisibility','off');
    x2 = [time, fliplr(time)];
    inBetween = [ci1_l.dist_thin_high, fliplr(ci1_l.dist_thin_low)];
    h = fill(x2, inBetween, [0, 0.447, 0.741], 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha',.10)

    plot(time, ci1_l.dist_med_avg, 'color', '#D95319', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, ci1_l.dist_med_high, 'LineWidth', 0.01, 'color', '#D95319','HandleVisibility','off');
    plot(time, ci1_l.dist_med_low, 'LineWidth', 0.01, 'color', '#D95319','HandleVisibility','off');
    x2 = [time, fliplr(time)];
    inBetween = [ci1_l.dist_med_high, fliplr(ci1_l.dist_med_low)];
    h = fill(x2, inBetween, [0.851, 0.325, 0.098], 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha',.10)

    plot(time, ci1_l.dist_thick_avg, 'color', '#FCD200', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, ci1_l.dist_thick_high, 'LineWidth', 0.01, 'color', '#FCD200','HandleVisibility','off');
    plot(time, ci1_l.dist_thick_low, 'LineWidth', 0.01, 'color', '#FCD200','HandleVisibility','off');
   x2 = [time, fliplr(time)];
    inBetween = [ci1_l.dist_thick_high, fliplr(ci1_l.dist_thick_low)];
    h = fill(x2, inBetween, [1,0.92,0], 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha',.175)

    xlim(plotting_window);

    ylim([-5, 7])
    grid on;
    hold off;

    if peak_effect ~= 0
        xline(start_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
        xline(end_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
        xline(peak_effect, '--r','HandleVisibility','off', "LineWidth", 1.5);
        xline(line_plot_effect, '--','HandleVisibility','off', "LineWidth", 1.5);
        xline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)
        yline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)
    end

    %xlabel("Milliseconds", "FontSize",labels_text_size)
    %ylabel(ax_label, "FontSize",labels_text_size)
    ylh = ylabel("Amplitude", "FontSize",16, "Rotation", 90, 'HorizontalAlignment','right', "FontWeight", "bold");
    ylh.Position(2) = 3.5;
    ylh.Position(1) = -220;

    % Shade the region before start_peak
    xregion([-inf, start_peak], 'FaceColor', [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');
    
    % Shade the region after end_peak
    xregion([end_peak, inf], 'FaceColor', [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');


    % high med etc
    subplot(3,3, 8);
set(gca, 'Position', pos5); % Adjust the size and position
     hold on;
    [~, peak_effect_idx] = min(abs(data{1}.time - line_plot_effect/1000));
    low_thick =  ci1_l.dist_thick_avg(peak_effect_idx);
    low_med =  ci1_l.dist_med_avg(peak_effect_idx);
    low_thin =  ci1_h.dist_thin_avg(peak_effect_idx);
    high_thick =  ci1_h.dist_thick_avg(peak_effect_idx);
    high_med =  ci1_h.dist_med_avg(peak_effect_idx);
    high_thin =  ci1_h.dist_thin_avg(peak_effect_idx);
    
    thick = [low_thick, high_thick];
    med = [low_med, high_med];
    thin = [low_thin, high_thin];

    plot(thick, '-o','color', '#FCD200' ,'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(thin,  '-o','color', '#0072BD' ,'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(med, '-o','color',  '#D95319','LineWidth', plot_line_width,'HandleVisibility','off');
    
    plot(NaN(1), 'Color', '#0072BD');
    plot(NaN(1), 'Color', '#D95319');
    plot(NaN(1), 'Color', '#FFFF00');
    %plot(NaN(1), 'Color', 'k');
    legend({'Thin', 'Medium', 'Thick'},'Location','northwest','FontSize', labels_text_size)
    

    % Customize x-axis labels
    xticks([1 2]);                 % Set x-tick positions
    xticklabels({'LOW', 'HIGH'});   % Replace numeric x-ticks with custom labels
    
    % Remove x-axis numeric labels
    set(gca, 'XTickLabelMode', 'manual', 'XTickMode', 'manual');
    grid on;

    subplot(3,3,9);
set(gca, 'Position', pos6); % Adjust the size and position
        hold on;
    
    plot(time, ci1_h.dist_thin_avg, 'color', '#0072BD', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, ci1_h.dist_thin_high, 'LineWidth', 0.01, 'color', '#0072BD','HandleVisibility','off');
    plot(time, ci1_h.dist_thin_low, 'LineWidth', 0.01, 'color', '#0072BD','HandleVisibility','off');
    x2 = [time, fliplr(time)];
    inBetween = [ci1_h.dist_thin_high, fliplr(ci1_h.dist_thin_low)];
    h = fill(x2, inBetween, [0, 0.447, 0.741], 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha',.10)

    plot(time, ci1_h.dist_med_avg, 'color', '#D95319', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, ci1_h.dist_med_high, 'LineWidth', 0.01, 'color', '#D95319','HandleVisibility','off');
    plot(time, ci1_h.dist_med_low, 'LineWidth', 0.01, 'color', '#D95319','HandleVisibility','off');
    x2 = [time, fliplr(time)];
    inBetween = [ci1_h.dist_med_high, fliplr(ci1_h.dist_med_low)];
    h = fill(x2, inBetween, [0.851, 0.325, 0.098], 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha',.10)

    plot(time, ci1_h.dist_thick_avg, 'color', '#FCD200', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, ci1_h.dist_thick_high, 'LineWidth', 0.01, 'color', '#FCD200','HandleVisibility','off');
    plot(time, ci1_h.dist_thick_low, 'LineWidth', 0.01, 'color', '#FCD200','HandleVisibility','off');
    x2 = [time, fliplr(time)];
    inBetween = [ci1_h.dist_thick_high, fliplr(ci1_h.dist_thick_low)];
    h = fill(x2, inBetween, [1,0.92,0], 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha',.175)

    xlim(plotting_window);

    ylim([-5, 7])
    grid on;
    hold off;

    if peak_effect ~= 0
        xline(start_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
        xline(end_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
        xline(peak_effect, '--r','HandleVisibility','off', "LineWidth", 1.5);
        xline(line_plot_effect, '--','HandleVisibility','off', "LineWidth", 1.5);
        xline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)
        yline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)
    end

    %ylabel(ax_label, "FontSize",labels_text_size)

    % Shade the region before start_peak
    xregion([-inf, start_peak], 'FaceColor', [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');
    
    % Shade the region after end_peak
    xregion([end_peak, inf], 'FaceColor', [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');


    %xlabel("Milliseconds", "FontSize",labels_text_size)
    %ylabel(ax_label, "FontSize",labels_text_size)



elseif strcmp(experiment_type, 'partitions-2-8') || strcmp(experiment_type, 'erps-23-45-67')
    figure;
    tick_mark_size = 12;
    line_plot_type = 'peak';
    if plot_thin_med_thick        
        % Row 1
        pos1 = [0.05, 0.55, 0.35, 0.35];  % First plot (larger width)
        pos2 = [0.425, 0.55, 0.15, 0.35];  % Second plot (half width)
        pos3 = [0.6, 0.55, 0.35, 0.35];    % Third plot (larger width)
        
        % Row 2
        pos4 = [0.05, 0.5, 0.35, 0.35];    % First plot (larger width)
        pos5 = [0.425, 0.5, 0.15, 0.35];   % Second plot (half width)
        pos6 = [0.6, 0.5, 0.35, 0.35];     % Third plot (larger width)
    else
% Row 1
        pos1 = [0.05, 0.55, 0.35, 0.35];  % First plot (larger width)
        pos2 = [0.425, 0.55, 0.15, 0.35];  % Second plot (half width)
        pos3 = [0.6, 0.55, 0.35, 0.35];    % Third plot (larger width)
        
        % Row 2
        pos4 = [0.05, 0.15, 0.35, 0.35];    % First plot (larger width)
        pos5 = [0.425, 0.15, 0.15, 0.35];   % Second plot (half width)
        pos6 = [0.6, 0.15, 0.35, 0.35];     % Third plot (larger width)
    end
    
    time = data{1}.time * 1000;
 

    % PGI LOW
    subplot(2,3,1);
    set(gca, 'Position', pos1); % Adjust the size and position

    % Shade the region before start_peak
    xregion([-inf, start_peak], 'FaceColor', [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');
    
    % Shade the region after end_peak
    xregion([end_peak, inf], 'FaceColor', [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');


    hold on;
    %plot(NaN(1), 'r');
    %plot(NaN(1), 'g');
    %plot(NaN(1),  'color','#4DBEEE');
    if contains(experiment_type, 'partitions-2-8')
        if strcmp(regression_type, "Visual Stress") 
            line_plot_effect = 226;
        elseif strcmp(regression_type, "Headache")
             line_plot_effect =171.094;
        elseif strcmp(regression_type, "Discomfort")
            line_plot_effect = 173.5;
        end

        %legend({'P1-PGI', 'P2-PGI', 'P3-PGI'},'Location','bestoutside','FontSize', labels_text_size)
        pgi_low = "Low Group: Partitions PGI";
        pgi_high = "High Group: Partitions: PGI";
        med_low = "Low Group: Medium Through the Partitions";
        med_high = "High Group: Medium Through the Partitions";
        low_p1 = "Low Group P1";
        high_p1 = "High Group P1";
        low_p2 = "Low Group P2";
        high_p2 = "High Group P2";
        low_p3 = "Low Group P3";
        high_p3 = "High Group P3";

    elseif strcmp(experiment_type, 'erps-23-45-67')
        if strcmp(regression_type, "Visual Stress") 
            line_plot_effect = 244;
        elseif strcmp(regression_type, "Headache")
            line_plot_effect =148;
        elseif strcmp(regression_type, "Discomfort")
            line_plot_effect = 205;
        end
        %legend({'Onsets 2:3', 'Onsets 4:5', 'Onsets 6:7'},'Location','bestoutside','FontSize', labels_text_size)
        pgi_low = "Low Group: Onsets PGI";
        pgi_high = "High Group: Onsets: PGI";
        med_low = "Low Group: Medium Through the Onsets";
        med_high = "High Group: Medium Through the Onsets";
        low_p1 = "Low Group Onsets 2,3";
        high_p1 = "High Group Onsets 2,3";
        low_p2 = "Low Group Onsets 4,5";
        high_p2 = "High Group Onsets 4,5";
        low_p3 = "Low Group Onsets 6,7";
        high_p3 = "High Group Onsets 6,7";

    end

    plot(time, ci1_l.dist_pgi_avg, 'color', 'r', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, ci1_l.dist_pgi_high, 'LineWidth', 0.01, 'color', 'r','HandleVisibility','off');
    plot(time, ci1_l.dist_pgi_low, 'LineWidth', 0.01, 'color', 'r','HandleVisibility','off');
    x2 = [time, fliplr(time)];
    inBetween = [ci1_l.dist_pgi_high, fliplr(ci1_l.dist_pgi_low)];
    h = fill(x2, inBetween, 'r', 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha',.10)

    plot(time, ci2_l.dist_pgi_avg, 'color', 'g', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, ci2_l.dist_pgi_high, 'LineWidth', 0.01, 'color', 'g','HandleVisibility','off');
    plot(time, ci2_l.dist_pgi_low, 'LineWidth', 0.01, 'color', 'g','HandleVisibility','off');
    x2 = [time, fliplr(time)];
    inBetween = [ci2_l.dist_pgi_high, fliplr(ci2_l.dist_pgi_low)];
    h = fill(x2, inBetween, 'g', 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha', .10)

    plot(time, ci3_l.dist_pgi_avg, 'color', [0.3010 0.7450 0.9330], 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, ci3_l.dist_pgi_high, 'LineWidth', 0.01, 'color', [0.3010 0.7450 0.9330],'HandleVisibility','off');
    plot(time, ci3_l.dist_pgi_low, 'LineWidth', 0.01, 'color', [0.3010 0.7450 0.9330],'HandleVisibility','off');
    x2 = [time, fliplr(time)];
    inBetween = [ci3_l.dist_pgi_high, fliplr(ci3_l.dist_pgi_low)];
    h = fill(x2, inBetween, [0.3010 0.7450 0.9330], 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha',.10)

    xlim(plotting_window);
    title("LOW",'FontSize', 18);



    min_yl = min([min(ci1_l.dist_pgi_low), min(ci2_l.dist_pgi_low), min(ci3_l.dist_pgi_low)])-0.5;
    max_yl = max([max(ci3_l.dist_pgi_high), max(ci2_l.dist_pgi_high), max(ci1_l.dist_pgi_high)])+0.5;

    min_yh = min([min(ci1_h.dist_pgi_low), min(ci2_h.dist_pgi_low), min(ci3_h.dist_pgi_low)])-0.5;
    max_yh = max([max(ci3_h.dist_pgi_high), max(ci2_h.dist_pgi_high), max(ci1_h.dist_pgi_high)])+0.5;

    min_y = min([min_yl, min_yh]);
    max_y = max([max_yl, max_yh]);

    ylim([min_y, max_y])

    grid on;
    hold off;
    
    % Gray out the region before the start_peak

    % Shade the region before start_peak
    xregion([-inf, start_peak], 'FaceColor', [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');
    
    % Shade the region after end_peak
    xregion([end_peak, inf], 'FaceColor', [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');


    xline(start_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
    xline(end_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
    xline(peak_effect, '--r','HandleVisibility','off', "LineWidth", 1.5);
    xline(line_plot_effect, '--','HandleVisibility','off', "LineWidth", 1.5);
    xline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)
    yline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)

    %xlabel("Milliseconds", "FontSize",labels_text_size)
    %ylabel(ax_label, "FontSize",labels_text_size)
    ylh = ylabel("PGI", "FontSize",23, "Rotation",90, 'HorizontalAlignment','right', "FontWeight", "bold");
    ylh.Position(2) = 0.5;
    ylh.Position(1) = -220;

    % interaction plot
    subplot(2,3,2);
    set(gca, 'Position', pos2);
    hold on;

    %plot(NaN(1), 'r');
    %plot(NaN(1), 'g'); 
    %plot(NaN(1),  'color','#4DBEEE');
    %if contains(experiment_type, 'partitions-2-8')
    %    legend({'P1: Low vs. High', 'P2: Low vs. High', 'P3: Low vs. High'},'Location','bestoutside','FontSize', labels_text_size)
    %elseif strcmp(experiment_type, 'erps-23-45-67')
    %    legend({'2:3 Low vs. High', '4:5 Low vs. High', '6:7 Loe vs. High'},'Location','bestoutside','FontSize', labels_text_size)
    %end

    [~, peak_effect_idx] = min(abs(data{1}.time - line_plot_effect/1000));

    if strcmp(line_plot_type, 'peak')
        p1_high =  ci1_h.dist_pgi_avg(peak_effect_idx);
        p2_high =  ci2_h.dist_pgi_avg(peak_effect_idx);
        p3_high =  ci3_h.dist_pgi_avg(peak_effect_idx);
        p1_low =  ci1_l.dist_pgi_avg(peak_effect_idx);
        p2_low =  ci2_l.dist_pgi_avg(peak_effect_idx);
        p3_low =  ci3_l.dist_pgi_avg(peak_effect_idx);
    elseif strcmp(line_plot_type, 'avg')
        p1_high= mean(ci1_h.dist_pgi_avg(start_peak:end_peak));
        p2_high= mean(ci2_h.dist_pgi_avg(start_peak:end_peak));
        p3_high= mean(ci3_h.dist_pgi_avg(start_peak:end_peak));
        p1_low= mean(ci1_l.dist_pgi_avg(start_peak:end_peak));
        p2_low= mean(ci2_l.dist_pgi_avg(start_peak:end_peak));
        p3_low= mean(ci3_l.dist_pgi_avg(start_peak:end_peak));
    end


    
    partitions_1 = [p1_low, p1_high];
    partitions_2 = [p2_low, p2_high];
    partitions_3 = [p3_low, p3_high];

    plot(partitions_1, '-o','color', 'r' ,'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(partitions_2,  '-o','color', 'g' ,'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(partitions_3, '-o','color','#4DBEEE','LineWidth', plot_line_width,'HandleVisibility','off');
    
    title("LOW vs. HIGH",'FontSize', 18);

    % Customize x-axis labels
    xticks([1 2]);                 % Set x-tick positions
    xticklabels({'LOW', 'HIGH'});   % Replace numeric x-ticks with custom labels
    
    % Remove x-axis numeric labels
    set(gca, 'XTickLabelMode', 'manual', 'XTickMode', 'manual');
      grid on;

    % PGI HIGH
subplot(2,3,3);
set(gca, 'Position', pos3); % Adjust the size and position
     hold on;


        % Shade the region before start_peak
    xregion([-inf, start_peak], 'FaceColor', [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');
    
    % Shade the region after end_peak
    xregion([end_peak, inf], 'FaceColor', [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');
    % plot(NaN(1), 'r');
    % plot(NaN(1), 'g'); 
    % plot(NaN(1),  'color','#4DBEEE');
    % if contains(experiment_type, 'partitions-2-8')
    %     legend({'P1-PGI', 'P2-PGI', 'P3-PGI'},'Location','bestoutside','FontSize', labels_text_size)
    % elseif strcmp(experiment_type, 'erps-23-45-67')
    %     legend({'Onsets 2:3', 'Onsets 4:5', 'Onsets 6:7'},'Location','bestoutside','FontSize', labels_text_size)
    % end

    plot(time, ci1_h.dist_pgi_avg, 'color', 'r', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, ci1_h.dist_pgi_high, 'LineWidth', 0.00001, 'color', 'r','HandleVisibility','off');
    plot(time, ci1_h.dist_pgi_low, 'LineWidth', 0.00001, 'color', 'r','HandleVisibility','off');
    x2 = [time, fliplr(time)];
    inBetween = [ci1_h.dist_pgi_high, fliplr(ci1_h.dist_pgi_low)];
    h = fill(x2, inBetween, 'r', 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha',.10)

    plot(time, ci2_h.dist_pgi_avg, 'color', 'g', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, ci2_h.dist_pgi_high, 'LineWidth', 0.00001, 'color', 'g','HandleVisibility','off');
    plot(time, ci2_h.dist_pgi_low, 'LineWidth', 0.00001, 'color', 'g','HandleVisibility','off');
    x2 = [time, fliplr(time)];
    inBetween = [ci2_h.dist_pgi_high, fliplr(ci2_h.dist_pgi_low)];
    h = fill(x2, inBetween, 'g', 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha',.10)

    plot(time, ci3_h.dist_pgi_avg, 'color', [0.3010 0.7450 0.9330], 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, ci3_h.dist_pgi_high, 'LineWidth', 0.00001, 'color', [0.3010 0.7450 0.9330],'HandleVisibility','off');
    plot(time, ci3_h.dist_pgi_low, 'LineWidth', 0.00001, 'color', [0.3010 0.7450 0.9330],'HandleVisibility','off');
    x2 = [time, fliplr(time)];
    inBetween = [ci3_h.dist_pgi_high, fliplr(ci3_h.dist_pgi_low)];
    h = fill(x2, inBetween, [0.3010 0.7450 0.9330], 'HandleVisibility','off', 'LineStyle','none');
    set(h,'facealpha',.10)

    xlim(plotting_window);
    title("HIGH",'FontSize', 18);


    ylim([min_y, max_y])
    grid on;
    hold off;

    xline(start_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
    xline(end_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
    xline(peak_effect, '--r','HandleVisibility','off', "LineWidth", 1.5);
    xline(line_plot_effect, '--','HandleVisibility','off', "LineWidth", 1.5);
    xline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)
    yline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)

    %xlabel("Milliseconds", "FontSize",labels_text_size)
    %ylabel(ax_label, "FontSize",labels_text_size)

    hold on;
    plot(NaN(1), 'r');
    plot(NaN(1), 'g'); 
    plot(NaN(1),  'color','#4DBEEE');
    if contains(experiment_type, 'partitions-2-8')
        legend({'P1-PGI', 'P2-PGI', 'P3-PGI'},'Location','northwest','FontSize', labels_text_size)
    elseif strcmp(experiment_type, 'erps-23-45-67')
        legend({'2:3 PGI', '4:5 PGI', '6:7 PGI'},'Location','northwest','FontSize', labels_text_size)
    end


    % MED LOW
subplot(2,3,4);
set(gca, 'Position', pos4); % Adjust the size and position
    hold on;

            % Shade the region before start_peak
    xregion([-inf, start_peak], 'FaceColor', [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');
    
    % Shade the region after end_peak
    xregion([end_peak, inf], 'FaceColor', [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');


    %plot(NaN(1), 'r');
    %plot(NaN(1), 'g');
    %plot(NaN(1), 'b');
    %if contains(experiment_type, 'partitions-2-8')
    %    legend({'Med-P1', 'Med-P2', 'Med-P3'},'Location','bestoutside','FontSize', labels_text_size)
    %elseif strcmp(experiment_type, 'erps-23-45-67')
    %    legend({'Med-2:3', 'Med-4:5', 'Med-6:7'},'Location','bestoutside','FontSize', labels_text_size)
    %end

    plot(time, ci1_l.dist_med_avg, 'color', 'r', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, ci2_l.dist_med_avg, 'color', 'g', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, ci3_l.dist_med_avg, 'color', 'b', 'LineWidth', plot_line_width,'HandleVisibility','off');

    xlim(plotting_window);
    %title("Medium",'FontSize', labels_text_size);

    min_y_low = min([min(ci1_l.dist_med_avg), min(ci2_l.dist_med_avg), min(ci3_l.dist_med_avg)])-0.5;
    max_y_low = max([max(ci1_l.dist_med_avg), max(ci2_l.dist_med_avg), max(ci3_l.dist_med_avg)])+0.5;

    min_y_high = min([min(ci1_h.dist_med_avg), min(ci2_h.dist_med_avg), min(ci3_h.dist_med_avg)])-0.5;
    max_y_high = max([max(ci1_h.dist_med_avg), max(ci2_h.dist_med_avg), max(ci3_h.dist_med_avg)])+0.5;

    min_y = min([min_y_low, min_y_high]);
    max_y = max([max_y_low, max_y_high]);

    ylim([min_y, max_y])

    grid on;
    hold off;

    xline(start_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
    xline(end_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
    xline(peak_effect, '--r','HandleVisibility','off', "LineWidth", 1.5);
    xline(line_plot_effect, '--','HandleVisibility','off', "LineWidth", 1.5);
    xline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)
    yline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)

    %xlabel("Milliseconds", "FontSize",labels_text_size)
    %ylabel(ax_label, "FontSize",labels_text_size)
    ylh = ylabel("Medium", "FontSize",23, "Rotation", 90 ,'HorizontalAlignment','right', "FontWeight", "bold");
    ylh.Position(2) = 5;
    ylh.Position(1) = -220;


    %% scatter plot 
subplot(2,3,5);
set(gca, 'Position', pos5); % Adjust the size and position
     hold on;
    [~, peak_effect_idx] = min(abs(data{1}.time - line_plot_effect/1000));
    p1_high =  ci1_h.dist_med_avg(peak_effect_idx);
    p2_high =  ci2_h.dist_med_avg(peak_effect_idx);
    p3_high =  ci3_h.dist_med_avg(peak_effect_idx);
    p1_low =  ci1_l.dist_med_avg(peak_effect_idx);
    p2_low =  ci2_l.dist_med_avg(peak_effect_idx);
    p3_low =  ci3_l.dist_med_avg(peak_effect_idx);
   
   if strcmp(line_plot_type, 'peak')
        p1_high =  ci1_h.dist_med_avg(peak_effect_idx);
        p2_high =  ci2_h.dist_med_avg(peak_effect_idx);
        p3_high =  ci3_h.dist_med_avg(peak_effect_idx);
        p1_low =  ci1_l.dist_med_avg(peak_effect_idx);
        p2_low =  ci2_l.dist_med_avg(peak_effect_idx);
        p3_low =  ci3_l.dist_med_avg(peak_effect_idx);
    elseif strcmp(line_plot_type, 'avg')
        p1_high= mean(ci1_h.dist_med_avg(start_peak:end_peak));
        p2_high= mean(ci2_h.dist_med_avg(start_peak:end_peak));
        p3_high= mean(ci3_h.dist_med_avg(start_peak:end_peak));
        p1_low= mean(ci1_l.dist_med_avg(start_peak:end_peak));
        p2_low= mean(ci2_l.dist_med_avg(start_peak:end_peak));
        p3_low= mean(ci3_l.dist_med_avg(start_peak:end_peak));
    end

    partitions_1 = [p1_low, p1_high];
    partitions_2 = [p2_low, p2_high];
    partitions_3 = [p3_low, p3_high];

    plot(partitions_1, '-o','color', 'r' ,'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(partitions_2,  '-o','color', 'g' ,'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(partitions_3, '-o','color', 'b','LineWidth', plot_line_width,'HandleVisibility','off');
    

    % Customize x-axis labels
    xticks([1 2]);                 % Set x-tick positions
    xticklabels({'LOW', 'HIGH'});   % Replace numeric x-ticks with custom labels
    
    % Remove x-axis numeric labels
    set(gca, 'XTickLabelMode', 'manual', 'XTickMode', 'manual');
    grid on;

    % MED HIGH
subplot(2,3,6);
set(gca, 'Position', pos6); % Adjust the size and position
    hold on;
  


    plot(time, ci1_h.dist_med_avg, 'color', 'r', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, ci2_h.dist_med_avg, 'color', 'g', 'LineWidth', plot_line_width,'HandleVisibility','off');
    plot(time, ci3_h.dist_med_avg, 'color', 'b', 'LineWidth', plot_line_width,'HandleVisibility','off');

    xlim(plotting_window);
    %title(med_high,'FontSize', labels_text_size);

    ylim([min_y, max_y])

    grid on;
    hold off;

    xline(start_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
    xline(end_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
    xline(peak_effect, '--r','HandleVisibility','off', "LineWidth", 1.5);
  xline(line_plot_effect, '--','HandleVisibility','off', "LineWidth", 1.5);
    xline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)
    yline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)

    %xlabel("Milliseconds", "FontSize",labels_text_size)
    %ylabel(ax_label, "FontSize",labels_text_size)

    % Shade the region before start_peak
    xregion([-inf, start_peak], 'FaceColor', [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');
    
    % Shade the region after end_peak
    xregion([end_peak, inf], 'FaceColor', [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'HandleVisibility', 'off');

     hold on;
    plot(NaN(1), 'r');
    plot(NaN(1), 'g');
    plot(NaN(1), 'b');
    if contains(experiment_type, 'partitions-2-8')
        legend({'P1-Med', 'P2-Med', 'P3-Med'},'Location','northwest','FontSize', labels_text_size)
    elseif strcmp(experiment_type, 'erps-23-45-67')
        legend({'2:3 Med', '4:5 Med', '6:7 Med'},'Location','northwest','FontSize', labels_text_size)
    end
    if plot_thin_med_thick
    
        % P1 LOW
        nexttile
        hold on;
        %plot(NaN(1), 'Color', '#0072BD');
        %plot(NaN(1), 'Color', '#D95319');
        %plot(NaN(1), 'Color', '#FFFF00');
        %legend({'Thin', 'Medium', 'Thick'},'Location','bestoutside','FontSize', labels_text_size)
    
        plot(time, ci1_l.dist_thin_avg, 'color', '#0072BD', 'LineWidth', plot_line_width,'HandleVisibility','off');
        plot(time, ci1_l.dist_thin_high, 'LineWidth', 0.01, 'color', '#0072BD','HandleVisibility','off');
        plot(time, ci1_l.dist_thin_low, 'LineWidth', 0.01, 'color', '#0072BD','HandleVisibility','off');
        x2 = [time, fliplr(time)];
        inBetween = [ci1_l.dist_thin_high, fliplr(ci1_l.dist_thin_low)];
        h = fill(x2, inBetween, [0, 0.447, 0.741], 'HandleVisibility','off', 'LineStyle','none');
        set(h,'facealpha',.10)
    
        plot(time, ci1_l.dist_med_avg, 'color', '#D95319', 'LineWidth', plot_line_width,'HandleVisibility','off');
        plot(time, ci1_l.dist_med_high, 'LineWidth', 0.01, 'color', '#D95319','HandleVisibility','off');
        plot(time, ci1_l.dist_med_low, 'LineWidth', 0.01, 'color', '#D95319','HandleVisibility','off');
        x2 = [time, fliplr(time)];
        inBetween = [ci1_l.dist_med_high, fliplr(ci1_l.dist_med_low)];
        h = fill(x2, inBetween, [0.851, 0.325, 0.098], 'HandleVisibility','off', 'LineStyle','none');
        set(h,'facealpha',.10)
    
        plot(time, ci1_l.dist_thick_avg, 'color', '#FCD200', 'LineWidth', plot_line_width,'HandleVisibility','off');
        plot(time, ci1_l.dist_thick_high, 'LineWidth', 0.01, 'color', '#FCD200','HandleVisibility','off');
        plot(time, ci1_l.dist_thick_low, 'LineWidth', 0.01, 'color', '#FCD200','HandleVisibility','off');
        x2 = [time, fliplr(time)];
        inBetween = [ci1_l.dist_thick_high, fliplr(ci1_l.dist_thick_low)];
        h = fill(x2, inBetween, [1,0.92,0], 'HandleVisibility','off', 'LineStyle','none');
        set(h,'facealpha',.175)
    
        xlim(plotting_window);
        %title(low_p1,'FontSize', labels_text_size);
    
        % calculate global min/maxes
        % P1
    
        min_yl = min([min(ci1_h.dist_thick_low), min(ci1_h.dist_med_low), min(ci1_h.dist_thin_low)])-0.5;
        max_yl = max([max(ci1_h.dist_thin_high), max(ci1_h.dist_med_high), max(ci1_h.dist_thick_high)])+0.5;
    
        min_yh = min([min(ci1_l.dist_thin_low), min(ci1_l.dist_med_low), min(ci1_l.dist_thick_low)])-0.5;
        max_yh = max([max(ci1_l.dist_thick_high), max(ci1_l.dist_med_high), max(ci1_l.dist_thin_high)])+0.5;
    
        min_y_p1 = min([min_yl, min_yh]);
        max_y_p1 = max([max_yl, max_yh]);
    
    
        % P2 
    
        min_yl = min([min(ci2_h.dist_thick_low), min(ci2_h.dist_med_low), min(ci2_h.dist_thin_low)])-0.5;
        max_yl = max([max(ci2_h.dist_thin_high), max(ci2_h.dist_med_high), max(ci2_h.dist_thick_high)])+0.5;
    
        min_yh = min([min(ci2_l.dist_thin_low), min(ci2_l.dist_med_low), min(ci2_l.dist_thick_low)])-0.5;
        max_yh = max([max(ci2_l.dist_thick_high), max(ci2_l.dist_med_high), max(ci2_l.dist_thin_high)])+0.5;
    
        min_y_p2 = min([min_yl, min_yh]);
        max_y_p2 = max([max_yl, max_yh]);
    
    
        % P3
    
        min_yl = min([min(ci3_h.dist_thick_low), min(ci3_h.dist_med_low), min(ci3_h.dist_thin_low)])-0.5;
        max_yl = max([max(ci3_h.dist_thin_high), max(ci3_h.dist_med_high), max(ci3_h.dist_thick_high)])+0.5;
    
        min_yh = min([min(ci3_l.dist_thin_low), min(ci3_l.dist_med_low), min(ci3_l.dist_thick_low)])-0.5;
        max_yh = max([max(ci3_l.dist_thick_high), max(ci3_l.dist_med_high), max(ci3_l.dist_thin_high)])+0.5;
    
        min_y_p3 = min([min_yl, min_yh]);
        max_y_p3 = max([max_yl, max_yh]);
    
        % calculate global min/max for partitions
        global_min_y = min([min_y_p3, min_y_p2, min_y_p1]);
        global_max_y = max([max_y_p3, max_y_p2, max_y_p1]);
    
    
        ylim([global_min_y, global_max_y])
    
        grid on;
        hold off;
    
        xline(start_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
        xline(end_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
        xline(peak_effect, '--r','HandleVisibility','off', "LineWidth", 1.5);
        xline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)
        yline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)
        %xlabel("Milliseconds", "FontSize",labels_text_size)
        %ylabel(ax_label, "FontSize",labels_text_size)
        if strcmp(experiment_type, 'erps-23-45-67')
            la = 'Onsets 2:3';
        else
            la = 'Partition 1';
        end
    
        ylh = ylabel(la, "FontSize",16, "Rotation", 0, 'HorizontalAlignment','right', "FontWeight", "bold");
        ylh.Position(2) = 2.75;
        ylh.Position(1) = -220;
    
        % P1 HIGH
        nexttile
        nexttile
        hold on;
        plot(NaN(1), 'Color', '#0072BD');
        plot(NaN(1), 'Color', '#D95319');
        plot(NaN(1), 'Color', '#FFFF00');
        legend({'Thin', 'Medium', 'Thick'},'Location','bestoutside','FontSize', labels_text_size)
    
        plot(time, ci1_h.dist_thin_avg, 'color','#0072BD', 'LineWidth', plot_line_width,'HandleVisibility','off');
        plot(time, ci1_h.dist_thin_high, 'LineWidth', 0.01, 'color', '#0072BD','HandleVisibility','off');
        plot(time, ci1_h.dist_thin_low, 'LineWidth', 0.01, 'color', '#0072BD','HandleVisibility','off');
        x2 = [time, fliplr(time)];
        inBetween = [ci1_h.dist_thin_high, fliplr(ci1_h.dist_thin_low)];
        h = fill(x2, inBetween, [0, 0.447, 0.741], 'HandleVisibility','off', 'LineStyle','none');
        set(h,'facealpha',.10)
    
        plot(time, ci1_h.dist_med_avg, 'color', '#D95319', 'LineWidth', plot_line_width,'HandleVisibility','off');
        plot(time, ci1_h.dist_med_high, 'LineWidth', 0.01, 'color', '#D95319','HandleVisibility','off');
        plot(time, ci1_h.dist_med_low, 'LineWidth', 0.01, 'color', '#D95319','HandleVisibility','off');
        x2 = [time, fliplr(time)];
        inBetween = [ci1_h.dist_med_high, fliplr(ci1_h.dist_med_low)];
        h = fill(x2, inBetween, [0.851, 0.325, 0.098], 'HandleVisibility','off', 'LineStyle','none');
        set(h,'facealpha',.10)
    
        plot(time, ci1_h.dist_thick_avg, 'color', '#FCD200', 'LineWidth', plot_line_width,'HandleVisibility','off');
        plot(time, ci1_h.dist_thick_high, 'LineWidth', 0.01, 'color', '#FCD200','HandleVisibility','off');
        plot(time, ci1_h.dist_thick_low, 'LineWidth', 0.01, 'color', '#FCD200','HandleVisibility','off');
        x2 = [time, fliplr(time)];
        inBetween = [ci1_h.dist_thick_high, fliplr(ci1_h.dist_thick_low)];
        h = fill(x2, inBetween, [1,0.92,0], 'HandleVisibility','off', 'LineStyle','none');
        set(h,'facealpha',.175)
    
        xlim(plotting_window);
        %title(high_p1,'FontSize', labels_text_size);
    
        ylim([global_min_y, global_max_y])
    
        grid on;
        hold off;
    
        xline(start_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
        xline(end_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
        xline(peak_effect, '--r','HandleVisibility','off', "LineWidth", 1.5);
        xline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)
        yline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)
        %xlabel("Milliseconds", "FontSize",labels_text_size)
        %ylabel(ax_label, "FontSize",labels_text_size)
    
    
        % P2 LOW
        nexttile
        hold on;
        %plot(NaN(1), 'Color', '#0072BD');
        %plot(NaN(1), 'Color', '#D95319');
        %plot(NaN(1), 'Color', '#FFFF00');
        %legend({'Thin', 'Medium', 'Thick'},'Location','bestoutside','FontSize', labels_text_size)
    
        plot(time, ci2_l.dist_thin_avg, 'Color', '#0072BD', 'LineWidth', plot_line_width,'HandleVisibility','off');
        plot(time, ci2_l.dist_thin_high, 'LineWidth', 0.01, 'Color', '#0072BD','HandleVisibility','off');
        plot(time, ci2_l.dist_thin_low, 'LineWidth', 0.01, 'Color', '#0072BD','HandleVisibility','off');
        x2 = [time, fliplr(time)];
        inBetween = [ci2_l.dist_thin_high, fliplr(ci2_l.dist_thin_low)];
        h = fill(x2, inBetween, [0, 0.447, 0.741], 'HandleVisibility','off', 'LineStyle','none');
        set(h,'facealpha',.10)
    
        plot(time, ci2_l.dist_med_avg, 'color', '#D95319', 'LineWidth', plot_line_width,'HandleVisibility','off');
        plot(time, ci2_l.dist_med_high, 'LineWidth', 0.01, 'color', '#D95319','HandleVisibility','off');
        plot(time, ci2_l.dist_med_low, 'LineWidth', 0.01, 'color', '#D95319','HandleVisibility','off');
        x2 = [time, fliplr(time)];
        inBetween = [ci2_l.dist_med_high, fliplr(ci2_l.dist_med_low)];
        h = fill(x2, inBetween, [0.851, 0.325, 0.098], 'HandleVisibility','off', 'LineStyle','none');
        set(h,'facealpha',.10)
    
        plot(time, ci2_l.dist_thick_avg, 'color', '#FCD200', 'LineWidth', plot_line_width,'HandleVisibility','off');
        plot(time, ci2_l.dist_thick_high, 'LineWidth', 0.01, 'color', '#FCD200','HandleVisibility','off');
        plot(time, ci2_l.dist_thick_low, 'LineWidth', 0.01, 'color', '#FCD200','HandleVisibility','off');
        x2 = [time, fliplr(time)];
        inBetween = [ci2_l.dist_thick_high, fliplr(ci2_l.dist_thick_low)];
        h = fill(x2, inBetween, [1,0.92,0], 'HandleVisibility','off', 'LineStyle','none');
        set(h,'facealpha',.175)
    
        xlim(plotting_window);
        %title(low_p2,'FontSize', labels_text_size);
    
        ylim([global_min_y, global_max_y])
    
        grid on;
        hold off;
    
        xline(start_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
        xline(end_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
        xline(peak_effect, '--r','HandleVisibility','off', "LineWidth", 1.5);
        xline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)
        yline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)
    
        %xlabel("Milliseconds", "FontSize",labels_text_size)
        %ylabel(ax_label, "FontSize",labels_text_size)
        if strcmp(experiment_type, 'erps-23-45-67')
            la = 'Onsets 4:5';
        else
            la = 'Partition 2';
        end
    
    
        ylh = ylabel(la, "FontSize",16, "Rotation", 0, 'HorizontalAlignment','right', "FontWeight", "bold");
        ylh.Position(2) = 2.75;
        ylh.Position(1) = -220;
    
        % P2 HIGH
        nexttile
        nexttile
        hold on;
        plot(NaN(1), 'Color', '#0072BD');
        plot(NaN(1), 'Color', '#D95319');
        plot(NaN(1), 'Color', '#FFFF00');
        legend({'Thin', 'Medium', 'Thick'},'Location','bestoutside','FontSize', labels_text_size)
    
        plot(time, ci2_h.dist_thin_avg, 'Color', '#0072BD', 'LineWidth', plot_line_width,'HandleVisibility','off');
        plot(time, ci2_h.dist_thin_high, 'LineWidth', 0.01, 'Color', '#0072BD','HandleVisibility','off');
        plot(time, ci2_h.dist_thin_low, 'LineWidth', 0.01, 'Color', '#0072BD','HandleVisibility','off');
        x2 = [time, fliplr(time)];
        inBetween = [ci2_h.dist_thin_high, fliplr(ci2_h.dist_thin_low)];
        h = fill(x2, inBetween, [0, 0.447, 0.741], 'HandleVisibility','off', 'LineStyle','none');
        set(h,'facealpha',.10)
    
        plot(time, ci2_h.dist_med_avg, 'color', '#D95319', 'LineWidth', plot_line_width,'HandleVisibility','off');
        plot(time, ci2_h.dist_med_high, 'LineWidth', 0.01, 'color', '#D95319','HandleVisibility','off');
        plot(time, ci2_h.dist_med_low, 'LineWidth', 0.01, 'color', '#D95319','HandleVisibility','off');
        x2 = [time, fliplr(time)];
        inBetween = [ci2_h.dist_med_high, fliplr(ci2_h.dist_med_low)];
        h = fill(x2, inBetween, [0.851, 0.325, 0.098], 'HandleVisibility','off', 'LineStyle','none');
        set(h,'facealpha',.10)
    
        plot(time, ci2_h.dist_thick_avg, 'color', '#FCD200', 'LineWidth', plot_line_width,'HandleVisibility','off');
        plot(time, ci2_h.dist_thick_high, 'LineWidth', 0.01, 'color', '#FCD200','HandleVisibility','off');
        plot(time, ci2_h.dist_thick_low, 'LineWidth', 0.01, 'color', '#FCD200','HandleVisibility','off');
        x2 = [time, fliplr(time)];
        inBetween = [ci2_h.dist_thick_high, fliplr(ci2_h.dist_thick_low)];
        h = fill(x2, inBetween, [1,0.92,0], 'HandleVisibility','off', 'LineStyle','none');
        set(h,'facealpha',.175)
    
        xlim(plotting_window);
        %title(high_p2,'FontSize', labels_text_size);
    
    
        ylim([global_min_y, global_max_y])
    
        grid on;
        hold off;
    
        xline(start_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
        xline(end_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
        xline(peak_effect, '--r','HandleVisibility','off', "LineWidth", 1.5);
        xline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)
        yline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)
    
        %xlabel("Milliseconds", "FontSize",labels_text_size)
        %ylabel(ax_label, "FontSize",labels_text_size)
    
    
        % P3 LOW
        nexttile
        hold on;
        %plot(NaN(1), 'Color', '#0072BD');
        %plot(NaN(1), 'Color', '#D95319');
        %plot(NaN(1), 'Color', '#FFFF00');
        %legend({'Thin', 'Medium', 'Thick'},'Location','bestoutside','FontSize', labels_text_size)
    
        plot(time, ci3_l.dist_thin_avg, 'Color', '#0072BD', 'LineWidth', plot_line_width,'HandleVisibility','off');
        plot(time, ci3_l.dist_thin_high, 'LineWidth', 0.01, 'Color', '#0072BD','HandleVisibility','off');
        plot(time, ci3_l.dist_thin_low, 'LineWidth', 0.01, 'Color', '#0072BD','HandleVisibility','off');
        x2 = [time, fliplr(time)];
        inBetween = [ci3_l.dist_thin_high, fliplr(ci3_l.dist_thin_low)];
        h = fill(x2, inBetween, [0, 0.447, 0.741], 'HandleVisibility','off', 'LineStyle','none');
        set(h,'facealpha',.10)
    
        plot(time, ci3_l.dist_med_avg, 'color', '#D95319', 'LineWidth', plot_line_width,'HandleVisibility','off');
        plot(time, ci3_l.dist_med_high, 'LineWidth', 0.01, 'color', '#D95319','HandleVisibility','off');
        plot(time, ci3_l.dist_med_low, 'LineWidth', 0.01, 'color', '#D95319','HandleVisibility','off');
        x2 = [time, fliplr(time)];
        inBetween = [ci3_l.dist_med_high, fliplr(ci3_l.dist_med_low)];
        h = fill(x2, inBetween, [0.851, 0.325, 0.098], 'HandleVisibility','off', 'LineStyle','none');
        set(h,'facealpha',.10)
    
        plot(time, ci3_l.dist_thick_avg, 'color', '#FCD200', 'LineWidth', plot_line_width,'HandleVisibility','off');
        plot(time, ci3_l.dist_thick_high, 'LineWidth', 0.01, 'color', '#FCD200','HandleVisibility','off');
        plot(time, ci3_l.dist_thick_low, 'LineWidth', 0.01, 'color', '#FCD200','HandleVisibility','off');
        x2 = [time, fliplr(time)];
        inBetween = [ci3_l.dist_thick_high, fliplr(ci3_l.dist_thick_low)];
        h = fill(x2, inBetween, [1,0.92,0], 'HandleVisibility','off', 'LineStyle','none');
        set(h,'facealpha',.175)
    
        xlim(plotting_window);
        %title(low_p3,'FontSize', labels_text_size);
    
        ylim([global_min_y, global_max_y])
    
        grid on;
        hold off;
    
        xline(start_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
        xline(end_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
        xline(peak_effect, '--r','HandleVisibility','off', "LineWidth", 1.5);
        xline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)
        yline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)
    
        %xlabel("Milliseconds", "FontSize",labels_text_size)
        %ylabel(ax_label, "FontSize",labels_text_size)
    
        if strcmp(experiment_type, 'erps-23-45-67')
            la = 'Onsets 6:7';
        else
            la = 'Partition 3';
        end
    
        ylh = ylabel(la, "FontSize",16, "Rotation", 0, 'HorizontalAlignment','right', "FontWeight", "bold");
        ylh.Position(2) = 2.75;
        ylh.Position(1) = -220;
    
        % P3 HIGH
        nexttile
        nexttile
        hold on;
        plot(NaN(1), 'Color', '#0072BD');
        plot(NaN(1), 'Color', '#D95319');
        plot(NaN(1), 'Color', '#FFFF00');
        legend({'Thin', 'Medium', 'Thick'},'Location','bestoutside','FontSize', labels_text_size)
    
        plot(time, ci3_h.dist_thin_avg, 'Color', '#0072BD', 'LineWidth', plot_line_width,'HandleVisibility','off');
        plot(time, ci3_h.dist_thin_high, 'LineWidth', 0.01, 'Color', '#0072BD','HandleVisibility','off');
        plot(time, ci3_h.dist_thin_low, 'LineWidth', 0.01, 'Color', '#0072BD','HandleVisibility','off');
        x2 = [time, fliplr(time)];
        inBetween = [ci3_h.dist_thin_high, fliplr(ci3_h.dist_thin_low)];
        h =  fill(x2, inBetween, [0, 0.447, 0.741], 'HandleVisibility','off', 'LineStyle','none');
        set(h,'facealpha',.10)
    
        plot(time, ci3_h.dist_med_avg, 'color', '#D95319', 'LineWidth', plot_line_width,'HandleVisibility','off');
        plot(time, ci3_h.dist_med_high, 'LineWidth', 0.01, 'color', '#D95319','HandleVisibility','off');
        plot(time, ci3_h.dist_med_low, 'LineWidth', 0.01, 'color', '#D95319','HandleVisibility','off');
        x2 = [time, fliplr(time)];
        inBetween = [ci3_h.dist_med_high, fliplr(ci3_h.dist_med_low)];
        h = fill(x2, inBetween, [0.851, 0.325, 0.098], 'HandleVisibility','off', 'LineStyle','none');
        set(h,'facealpha',.10)
    
        plot(time, ci3_h.dist_thick_avg, 'color', '#FCD200', 'LineWidth',3.5,'HandleVisibility','off');
        plot(time, ci3_h.dist_thick_high, 'LineWidth', 0.01, 'color', '#FCD200','HandleVisibility','off');
        plot(time, ci3_h.dist_thick_low, 'LineWidth', 0.01, 'color', '#FCD200','HandleVisibility','off');
        x2 = [time, fliplr(time)];
        inBetween = [ci3_h.dist_thick_high, fliplr(ci3_h.dist_thick_low)];
        h = fill(x2, inBetween, [1,0.92,0], 'HandleVisibility','off', 'LineStyle','none');
        set(h,'facealpha',.175)
    
        xlim(plotting_window);
        %title(high_p3,'FontSize', labels_text_size);
    
    
        ylim([global_min_y, global_max_y])
    
        grid on;
        hold off;
    
        xline(start_peak, '-','HandleVisibility','off', "LineWidth", 1.5);
        xline(end_peak, '-','HandleVisibility','off',"LineWidth", 1.5);
        xline(peak_effect, '--r','HandleVisibility','off', "LineWidth", 1.5);
        xline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)
        yline(0, '--b','HandleVisibility','off', "LineWidth", 1.5)
    
        %xlabel("Milliseconds", "FontSize",labels_text_size)
        %ylabel(ax_label, "FontSize",labels_text_size)
    end 

end

cluster_stats = "Cluster: P-value: " + num2str(round(pvalue, 3)) + ", Mass: " + num2str(cluster_size);
peak_stats = "Max sample: T" + "(" + num2str(df) + ")=" + num2str(t_value) ...
    + " Cohen's d: " + num2str(cohens_d) + ", Correlation: " + num2str(effect_size);

annotation('textbox', [0, 0.96, 1, 0.05], 'String', m_title, ...
           'FontSize', 20, 'HorizontalAlignment', 'center', ...
           'EdgeColor', 'none'); % Position the title at the top center

annotation('textbox', [0, 0.93, 1, 0.05], 'String', cluster_stats, ...
           'FontSize', 12, 'HorizontalAlignment', 'center', ...
           'EdgeColor', 'none'); % Position the first subtitle just below the main title

annotation('textbox', [0, 0.91, 1, 0.05], 'String', peak_stats, ...
           'FontSize', 12, 'HorizontalAlignment', 'center', ...
           'EdgeColor', 'none'); % Position the second subtitle just below the first


if contains(experiment_type, 'partitions-2-8') && first_partition_regresion ~= 1 || contains(experiment_type, 'erps-23-45-67') 
    if plot_thin_med_thick == 1
        set(gcf,'Position',[100 100 1250 1250])
        exportgraphics(gcf,save_dir,'Resolution',750);
    else
        set(gcf,'Position',[1.8 85.8 1750 1250])
        exportgraphics(gcf,save_dir,'Resolution',500);
    end
elseif contains(experiment_type, 'three-way-interaction')
   set(gcf,'Position',[1   49    1536    836.8])
    %p1 = text(-1800, 95, 'Partition 1', 'Fontsize', 12, 'Rotation', 90, "FontWeight","bold");
    %p2 = text(-1800, 65, 'Partition 2', 'Fontsize', 12, 'Rotation', 90, "FontWeight","bold");
   % p3 = text(-1800, 55, 'Partition 3', 'Fontsize', 12, 'Rotation', 90, "FontWeight","bold");
    exportgraphics(gcf,save_dir,'Resolution',750);
else
    set(gcf,'Position',[1 49.8 1536 836])
    exportgraphics(gcf,save_dir,'Resolution',500);
end
 close;
end
%% calculate partitions splits
function [data_high, data_low, high_ids, low_ids] ...
    = get_partitions_medium_split(data, participant_order, regression_type, ...
    partition, type_of_effect, weight_erps, weighting_factor)

    function [split_data, p_order] = get_participants(data, all_ids, current_ids)
        cnt = 1;
        split_data = {};
        p_order = {};
        for i=1:numel(data)
            participant = data{i};
            id = all_ids(i);

            if ismember(id{1}, current_ids)
                split_data{cnt} = participant;
                p_order{cnt} = id{1};
                cnt = cnt+1;
            end
        end
    end

    function data = weight_erps_based_on_score(data, ranks, type, order, weighting_factor)
        upper_weighting = 1 + weighting_factor;
        lower_weighting = 1 - weighting_factor;

        n = size(ranks, 1);
        quartile = floor(n/2);
        if strcmp(type, 'high')
            ids_in_quartile = ranks(1:quartile,2);
        else
            ids_in_quartile = ranks(quartile:end,2);
        end

        in_quartile = size(order,2);
        for i=1:in_quartile
            participant_order = order{i};
            participant = data{i};

            if ismember(participant_order,ids_in_quartile)
                participant.avg = participant.avg * upper_weighting;
                participant.thin = participant.thin * upper_weighting;
                participant.med = participant.med * upper_weighting;
                participant.thick = participant.thick * upper_weighting;
                participant.weighting = upper_weighting;
                data{i} = participant;
            else
                participant.avg = participant.avg * lower_weighting;
                participant.thin = participant.thin * lower_weighting;
                participant.med = participant.med * lower_weighting;
                participant.thick = participant.thick * lower_weighting;
                participant.weighting = lower_weighting;
                data{i} = participant;
            end
        end
    end

dataset = return_scores(regression_type, type_of_effect);
[scores, new_participants] = tweak_design_matrix(dataset, participant_order, data, type_of_effect);

if contains(regression_type, 'orthog')
    VS = return_scores('visual_stress', type_of_effect);
    DS = return_scores('discomfort', type_of_effect);
    HD = return_scores('headache', type_of_effect);

    [VS, ~] = tweak_design_matrix(VS, participant_order, data, type_of_effect);
    [DS, ~] = tweak_design_matrix(DS, participant_order, data, type_of_effect);
    [HD, ~] = tweak_design_matrix(HD, participant_order, data, type_of_effect);

    scores = orthog_data(VS, DS, HD, regression_type);
end

if partition == 1
    ratings = scores.one;
elseif partition == 2
    ratings = scores.two;
elseif partition == 3
    ratings = scores.three;
elseif partition == 0
    ratings = scores.one;
end

design = ratings(:,2);

sorted(:,1) = ratings(:,2);
sorted(:,2) = ratings(:,1);
sorted = flipud(sortrows(sorted));
n = ceil(numel(sorted(:,1))/2);
high = sorted(1:n,:);
high_ids = high(:,2);
low = sorted(n+1:end,:);
low_ids = low(:,2);

[data_high, high_order] = get_participants(data, participant_order, high_ids);
[data_low, low_order] = get_participants(data, participant_order, low_ids);

if weight_erps == 1
    [data_high] = weight_erps_based_on_score(data_high, high, 'high', high_order, weighting_factor);
    [data_low] = weight_erps_based_on_score(data_low, low, 'low', low_order, weighting_factor);
end
end

%% related to bootstrapping the erps
function ci = bootstrap_erps(data, e_idx)
[~, n_participants] = size(data);
[all_med, all_thick, all_thin] = deal({}, {}, {});

% get all of the participant trials into one matrix of each type
for participant=1:n_participants
    p = data{participant};
    fields = fieldnames(p);
    for k=1:numel(fields)
        time_series_name = fields{k};
        participant_level.series = p.(fields{k});
        participant_level.weighting = 1;

        if contains(time_series_name, 'med')
            participant_level.series = participant_level.series(e_idx,:);
            all_med{end+1} = participant_level;
        elseif contains(time_series_name, 'thick')
            participant_level.series = participant_level.series(e_idx,:);
            all_thick{end+1} = participant_level;
        elseif contains(time_series_name, 'thin')
            participant_level.series = participant_level.series(e_idx,:);
            all_thin{end+1} = participant_level;
        end

    end
end

[~, n_participants] = size(data);


% start the bootstrapping process to create the plots with CIs
n_iterations = 3000;
[dist_med, dist_thin, dist_thick, dist_pgi] = deal([ ], [], [], []);
for n =1:n_iterations
    % sample trials with replacement
    sampled_med = datasample(all_med,n_participants);
    sampled_thick = datasample(all_thick,n_participants);
    sampled_thin = datasample(all_thin,n_participants);

    % weight the ERPs using the arithmetic mean amd create a
    % bootstrapped ERP
    avg_med = calculate_aritmetic_mean(sampled_med);
    avg_thick = calculate_aritmetic_mean(sampled_thick);
    avg_thin = calculate_aritmetic_mean(sampled_thin);
    avg_pgi = avg_med - (avg_thin + avg_thick)/2;

    % add to our distribution of ERPs
    dist_med(:,:,n) = avg_med;
    dist_thin(:,:,n) = avg_thin;
    dist_thick(:,:,n) = avg_thick;
    dist_pgi(:,:,n) = avg_pgi;
end

[dist_med_low, dist_med_high, dist_med_avg] = deal([], [], []);
[dist_thick_low, dist_thick_high, dist_thick_avg] = deal([], [], []);
[dist_thin_low, dist_thin_high, dist_thin_avg] = deal([], [], []);
[dist_pgi_low, dist_pgi_high, dist_pgi_avg] = deal([], [], []);


n_samples = size(dist_med, 2);
for i=1:n_samples

    % medium 2.5% and 95% CI
    med_at_time_t = dist_med(1,i,:);
    lower = prctile(med_at_time_t, 2.5);
    upper = prctile(med_at_time_t, 97.5);
    dist_med_low(i) = lower;
    dist_med_high(i) = upper;
    dist_med_avg(i) = mean(med_at_time_t);

    % thick 2.5% and 95% CI
    thick_at_time_t = dist_thick(1,i,:);
    lower = prctile(thick_at_time_t, 2.5);
    upper = prctile(thick_at_time_t, 95);
    dist_thick_low(i) = lower;
    dist_thick_high(i) = upper;
    dist_thick_avg(i) = mean(thick_at_time_t);

    % thin 2.5% and 95% CI
    thin_at_time_t = dist_thin(1,i,:);
    lower = prctile(thin_at_time_t, 2.5);
    upper = prctile(thin_at_time_t, 95);
    dist_thin_low(i) = lower;
    dist_thin_high(i) = upper;
    dist_thin_avg(i) = mean(thin_at_time_t);

    % pgi 2.5% and 95% CI
    pgi_at_time_t = dist_pgi(1,i,:);
    lower = prctile(pgi_at_time_t, 2.5);
    upper = prctile(pgi_at_time_t, 95);
    dist_pgi_low(i) = lower;
    dist_pgi_high(i) = upper;
    dist_pgi_avg(i) = mean(pgi_at_time_t);
end

ci.dist_pgi_low = dist_pgi_low;
ci.dist_pgi_high = dist_pgi_high;
ci.dist_pgi_avg = dist_pgi_avg;

ci.dist_thin_low = dist_thin_low;
ci.dist_thin_high = dist_thin_high;
ci.dist_thin_avg = dist_thin_avg;

ci.dist_med_high = dist_med_high;
ci.dist_med_low = dist_med_low;
ci.dist_med_avg = dist_med_avg;

ci.dist_thick_low = dist_thick_low;
ci.dist_thick_high = dist_thick_high;
ci.dist_thick_avg = dist_thick_avg;

end

% mainly to handle memory constraints
function ft = cut_data_using_analysis_window(ft, analysis_window)
time = ft.time*1000;
[~,start_t] = min(abs(time-analysis_window(1)));
[~,end_t] = min(abs(time-analysis_window(2)));

fields = fieldnames(ft);

for k=1:numel(fields)
    time_series = ft.(fields{k});
    if numel(size(time_series)) == 2 || numel(size(time_series)) == 1
        filtered_series = time_series(:,start_t:end_t);
        ft.(fields{k}) = filtered_series;
    else
        new_time_series = [];
        [~,~,z] = size(time_series);
        for i=1:z
            trial =  time_series(:,start_t:end_t,i);
            new_time_series(:,:,i) = trial;
        end
        ft.(fields{k}) = new_time_series;
    end
end
end

%% applies the wavelett decomposition to the data
function dataset = to_frequency_data(save_dir, partition, type, ...
    participant_level, foi, analysis_type, filename, n_participants, toi)

    low = foi(1);
    high = foi(2);
    
    time_start = toi(1);
    time_end = toi(2);

    if low == 10 && high == 15
        num_cycles = 5;
    elseif low == 20 && high == 30
        num_cycles = 13;
    elseif low == 30 && high == 45
        num_cycles = 17;
    elseif low == 45 && high == 60
        num_cycles = 24;
    elseif low == 60 && high == 80
        num_cycles = 32;
    elseif low == 10 && high == 80
        num_cycles = 5;
    end

cfg = [];
cfg.channel = 'all';
cfg.method = 'wavelet';
cfg.width = num_cycles;
cfg.output = 'pow';
cfg.pad = 'nextpow2';
cfg.foi = foi(1):1:foi(2);
cfg.toi = -0.5:0.002:3.05;
cfg.keeptrials = 'yes';


dataset = {};
for i=1:n_participants

    main_dir = strcat(save_dir, int2str(i), '\', 'frequency_domain');

    if not(exist(main_dir, 'dir'))
        mkdir(main_dir);
    end

    save_path = strcat(save_dir, int2str(i), '\', 'frequency_domain\', analysis_type, '_', int2str(partition.partition_number), '_');


    full_save_dir = save_path + "trial_level_" + int2str(foi(1)) + "_" + int2str(foi(2)) + "_numcycles_" + int2str(num_cycles) + "_" + num2str(time_start) + "_" + num2str(time_end) + ".mat";

    if strcmp(type, 'preprocess')
      
        % if we want to preprocess, load each participants data
        disp(strcat('LOADING PARTICIPANT...', int2str(i)));
        participant_main_path = strcat(save_dir, int2str(i));
        [participant, participant_number] = load_one_postprocessed_data( ...
            main_dir, filename, partition, i);

        if participant_number == -9999
            continue;
        end

        % end of loading

        med.label = participant.label;
        med.elec = participant.elec;
        med.trial = participant.med;
        med.dimord = 'chan_time';

        thick.label = participant.label;
        thick.elec = participant.elec;
        thick.trial = participant.thick;
        thick.dimord = 'chan_time';

        thin.label = participant.label;
        thin.elec = participant.elec;
        thin.trial = participant.thin;
        thin.dimord = 'chan_time';

        if strcmp(participant_level, 'trial-level')
            med.time = update_with_time_info(med.trial, participant.time);
            thin.time = update_with_time_info(thin.trial, participant.time);
            thick.time = update_with_time_info(thick.trial, participant.time);
        else
            med.time = participant.time;
            thick.time = participant.time;
            thin.time = participant.time;
        end


        % to freq domain
        TFRwave_med = ft_freqanalysis(cfg, med);
        TFRwave_thin = ft_freqanalysis(cfg, thin);
        TFRwave_thick = ft_freqanalysis(cfg, thick);

        TFRwave_med.info = 'medium';
        TFRwave_thick.info = 'thick';
        TFRwave_thin.info = 'thin';

        %crop the epoch
        TFRwave_med.time = TFRwave_med.time(200:end); %200 before, 46 after
        TFRwave_med.powspctrm = TFRwave_med.powspctrm(:,:,:,200:end);
        TFRwave_thin.time = TFRwave_thin.time(200:end);
        TFRwave_thin.powspctrm = TFRwave_thin.powspctrm(:,:,:,200:end);
        TFRwave_thick.time = TFRwave_thick.time(200:end);
        TFRwave_thick.powspctrm = TFRwave_thick.powspctrm(:,:,:,200:end);

        %average across trials
        newcfg = [];
        avg_TFRwave_med = ft_freqdescriptives(newcfg, TFRwave_med);
        avg_TFRwave_thin = ft_freqdescriptives(newcfg, TFRwave_thin);
        avg_TFRwave_thick = ft_freqdescriptives(newcfg, TFRwave_thick);

        % baseline rescale
        newcfg = [];
        newcfg.baselinetype = 'db';
        newcfg.baseline = [-0.1 0];
        avg_TFRwave_med = ft_freqbaseline(newcfg,avg_TFRwave_med);
        avg_TFRwave_thin = ft_freqbaseline(newcfg,avg_TFRwave_thin);
        avg_TFRwave_thick = ft_freqbaseline(newcfg,avg_TFRwave_thick);


        disp(sum(sum(sum(avg_TFRwave_med.powspctrm)), 'omitnan'))

        TFRwave_pgi = avg_TFRwave_med;
        TFRwave_pgi.info = 'PGI';
        TFRwave_pgi.powspctrm = avg_TFRwave_med.powspctrm-(avg_TFRwave_thin.powspctrm+avg_TFRwave_thick.powspctrm)/2; %%!!!!%%%
        TFRwave_pgi.elec = med.elec;

        frequency_data.med = avg_TFRwave_med;
        frequency_data.thick = avg_TFRwave_thick;
        frequency_data.thin = avg_TFRwave_thin;
        frequency_data.pgi = TFRwave_pgi;
        frequency_data.participant_number = participant_number;



        save(full_save_dir, 'frequency_data', '-v7.3')
        dataset{end+1} = frequency_data;
        clear frequency_data;
    elseif strcmp(type, 'load')
        if isfile(full_save_dir)
            load(full_save_dir);
            dataset{end+1} = frequency_data;
        else
            continue;
        end
    end
end
end

%% update with time information
function trial_info = update_with_time_info(trial, time)
trial_info = {};
n = numel(trial);
for i=1:n
    trial_info{i} = time;
end
end

%% load the frequency data
function itc_data = calculate_itc(data, required_channel)

itc           = [];
itc.label     = data.label;
itc.freq      = data.freq;
itc.time      = data.time;
itc.dimord    = 'chan_freq_time';

channel_idx = find(contains(itc.label,required_channel));

F = data.fourierspctrm;
N = size(F,1);           % number of trials
F = F(2:end,:,:,:);

% compute inter-trial phase coherence (itpc)
itc.itpc      = F./abs(F);         % divide by amplitude
itc.itpc      = sum(itc.itpc,1);   % sum angles
itc.itpc      = abs(itc.itpc)/N;   % take the absolute value and normalize
itc.itpc      = squeeze(itc.itpc); % remove the first singleton dimension
inter_trial_coherence = itc.itpc(channel_idx, :, :); % get the nth channel
inter_trial_coherence = squeeze(inter_trial_coherence); % remove 1st dim

itc_data.inter_trial_coherence = inter_trial_coherence;
itc_data.time = itc.time;
itc_data.freq = data.freq;
end

%% calculate the arithmetic mean based on weightings
function avg_mtx = calculate_aritmetic_mean(data)

n = size(data,2);
total_weight = 0;
matricies = [];
for i=1:n
    matricies(:,:,i) = data{i}.series;
    total_weight = total_weight + data{i}.weighting;
end

sum_mtx = sum(matricies,3);
avg_mtx = sum_mtx/total_weight;

end

%% compute spectrograms of the participant data
function f_data = prepare_data(data, analysis_type)
if strcmp(analysis_type, 'trial-level')

    [thin, med, thick] = deal({}, {}, {});
    n_participants = size(data,2);

    for i=1:n_participants
        participant = data{i};
        thin{end+1} = participant.thin;
        thick{end+1} = participant.thick;
        med{end+1} = participant.med;
    end

    % save the freq plots
    cfg = [];
    thin_avg = ft_freqgrandaverage(cfg, thin{:});
    thick_avg = ft_freqgrandaverage(cfg, thick{:});
    med_avg = ft_freqgrandaverage(cfg, med{:});

    freq_data.thin = thin_avg;
    freq_data.thick = thick_avg;
    freq_data.med = med_avg;
    freq_data.time = data{1}.med.time;
    freq_data.freq = data{1}.med.freq;
    f_data{1} = freq_data;
elseif strcmp(analysis_type, 'participant-level')
    f_data = data;
end
end

%% compute a barchat of frequencies through time
function compute_frequency_bar_chart(mtx, time, freq, save_path)
interval = 0.020; % 20ms
max_time = 0.300; % 300ms

% theta, alpha, beta, gamma
bands = [[5,7]; [8,15];[16,31];[32, 100]];
N = size(bands,1);


for b=1:N
    f = bands(b,:);

    save_path = strcat(save_path, {'_'}, int2str(f(1)), {'-'}, int2str(f(2)), '.png');
    save_path = save_path{1};

    time_of_interest = 0; % start a 0ms
    time_axes = [];
    power_axes = [];
    while time_of_interest <= max_time

        % get indices that fall between the desired time intervals
        [~, time_idx] = find(time>=time_of_interest & time<=time_of_interest+interval);
        [~, freq_idx] = find(freq>=f(1) & freq<=f(2));

        % get the data for plotting
        subset = mtx(freq_idx,time_idx);
        avg_at_t = mean(subset,'all','omitnan');

        time_axes(end+1) = time_of_interest;
        power_axes(end+1) = avg_at_t;
        time_of_interest = time_of_interest + interval;
    end

    figure;
    plot(time_axes, power_axes);
    xlabel('Time in 20ms intervals');
    ylabel('ITC value at time T (0-1)')
    t = strcat('ITC values at frequencies:', {' '}, int2str(f(1)), {'-'}, int2str(f(2)));
    t = t{1};
    set(get(gca, 'title'), 'string', t)
    set(gca, 'xtick', time_axes);
    ylim([0,1]);
    grid on
    set(gcf,'Position',[100 100 750 750]);
    exportgraphics(gcf,save_path,'Resolution',500);
    close;
end
end

%% hardcoded values from the MUA analysis which returns apriori electrode
function electrode = return_mua_electrode(factor)
if strcmp(factor, 'no-factor')
    electrode = 'A22';
elseif strcmp(factor, 'headache')
    electrode = 'A22';
elseif strcmp(factor, 'discomfort')
    electrode = 'A15';
elseif strcmp(factor, 'visual_stress')
    electrode = 'A22';
end
end

%% Woody Algorithm Implementation
% x is a matrix where each row is a participants ERP at electrode X over
% time
function out = woody_filtering(x, e_idx)
%Default parameter values

N = numel(x);
erps = [];
for i=1:N
    erps(i,:) = x{i}.avg(e_idx,:);
end

x = erps';
x = x(1:257,:);

tol= 0.1;
max_it=100;
xcorr_mthd='unbiased';

[N,M]=size(x);
mx=mean(x,2);
p=zeros(N,1);
conv=1;
run=0;
sig_x=diag(sqrt(x'*x));
X=xcorr(mx);
ref=length(X)/2;

if(mod(ref,2))
    ref=ceil(ref);
else
    ref=floor(ref);
end

if(nargout>1)
    %In this case we output the la3g of the trials as well
    lag_data=zeros(1,M);
end

while(conv*(run<max_it))
    disp(run);
    z=zeros(N,1);
    w=ones(N,1);
    for i=1:M

        y=x(:,i);
        xy=xcorr(mx,y,xcorr_mthd);

        [~,ind]=max(xy);
        if(ind>ref)
            lag=ref-ind-1;
        else
            lag=ref-ind;
        end
        if(lag>0)
            num=w(lag:end)-1;
            z(1:N-lag+1)=( z(1:N-lag+1).*num + y(lag:end))./w(lag:end);
            w(lag:end)=w(lag:end)+1;
        elseif(lag<0)
            num=w(lag*(-1)+1:end)-1;
            z(lag*(-1)+1:end)=( z(lag*(-1)+1:end).*num + y(1:N+lag) )./w(lag*(-1)+1:end);
            w(lag*(-1)+1:end)=w(lag*(-1)+1:end)+1;
        else
            z=z.*(w-1)./w + y./w;
            w=w+1;
        end
        if(exist('lag_data','var'))
            lag_data(i)=lag;
        end
    end

    old_mx=mx;
    mx=z;
    p_old=p;
    p=mx'*x./(sqrt(mx'*mx).*sig_x');
    p=sum(p)./M;
    err=abs(p-p_old);
    if(err<tol)
        conv=0;
    end
    run=run+1;
end
out=mx;
end

%% plot sepctrogram
function plot_spectrogram(data, save_path, partition, ...
    channel, cat_type)

participants = size(data,2);

cfg = [];
cfg.baseline = 'yes';
cfg.baseline     = [-0.5 0];
cfg.baselinetype = 'db';
cfg.maskstyle    = 'saturation';
cfg.xlim = [-0.200,0.500];
cfg.ylim = [0, 15];
%cfg.zlim = [-5, 5];
cfg.channel = channel;


for k = 1:numel(data)
    thin = data{k}.thin;
    thick = data{k}.thick;
    med = data{k}.med;
    %pgi = data{k}.avg;

    title = strcat(cat_type, {' '}, 'Partition:', {' '}, int2str(partition), ',', {' '}, 'Grating:', {' '},...
        'Medium,', {' '}, 'Channel:' ,{' '}, channel);
    title = title{1};
    cfg.title = title;
    figure
    ft_singleplotTFR(cfg, med);
    %rectangle('Position', [0.09, 5, 0.1, 2.6],'EdgeColor','r', 'LineWidth', 1)
    save_dir = strcat(save_path, '/spectrograms/',  cat_type, '_p', int2str(partition), 'part', int2str(k), '_medium_freq.png');
    exportgraphics(gcf,save_dir,'Resolution',500);
    close;

    % if aggr avg
    if isfield(data, 'aggr_avg')
        cfg.zlim = 'maxmin';
        title = strcat(cat_type, {' '}, 'Partition:', {' '}, int2str(partition), ',', {' '}, 'Grating:', {' '},...
            'Aggrgated Avg,', {' '}, 'Channel:' ,{' '}, channel);
        cfg.title = title;
        figure
        aggr_avg = data.aggr_avg;
        ft_singleplotTFR(cfg, aggr_avg);
        rectangle('Position', [0.09, 5, 0.1, 2.6],'EdgeColor','r', 'LineWidth', 1)
        save_dir = strcat(save_path, '/spectrograms/',  cat_type, '_p', int2str(partition),'_medium_freq.png');
        exportgraphics(gcf,save_dir,'Resolution',500);
        close;
    end
end
end

%% aggregate the power data across participants
function new_data = aggregate_freq_data(data, type)

fields = fieldnames(data);
N = numel(fields);

[thin_pwrspec, thick_pwrspec, med_pwrspec] = deal([],[],[]);
for i= 1:N
    data_i = data.(fields{i});
    if i == 1
        example_thin = data_i.thin;
        example_thick = data_i.thick;
        example_med = data_i.med;
        example_aggr = data_i.med;
    end

    thin_pwr = data_i.thin.powspctrm;
    thick_pwr = data_i.thick.powspctrm;
    med_pwr = data_i.med.powspctrm;

    thin_pwrspec(:,:,:,end+1) = thin_pwr;
    thick_pwrspec(:,:,:,end+1) = thick_pwr;
    med_pwrspec(:,:,:,end+1) = med_pwr;

end

avg_thick = mean(thick_pwrspec,4);
avg_thin = mean(thin_pwrspec,4);
avg_med = mean(med_pwrspec,4);
aggr_avg = (avg_thick + avg_thin + avg_med)/3;

example_thin.powspctrm = avg_thin;
example_thick.powspctrm = avg_thick;
example_med.powspctrm = avg_med;
example_aggr.powspctrm = aggr_avg;

new_data.thin = example_thin;
new_data.thick = example_thick;
new_data.med = example_med;
new_data.aggr_avg = example_aggr;

end

%% create power values from each participant
function average_power_values(data, freq, time, electrode, type)
N = numel(data);
electrode = find(contains(data{1}.label,electrode));

participant_data = [];
for i=1:N
    participant = data{i};
    participant_number = participant.participant_number;

    data_time = participant.med.time;
    data_freq = participant.med.freq;

    [~,start_time] = min(abs(data_time-time(1)));
    [~,end_time] = min(abs(data_time-time(2)));

    [~,start_freq] = min(abs(data_freq-freq(1)));
    [~,end_freq] = min(abs(data_freq-freq(2)));

    if strcmp(type, 'pow')
        thin = squeeze(participant.thin.powspctrm(electrode, ...
            start_freq:end_freq, ...
            start_time:end_time));
    else
        thin = participant.thin.inter_trial_coherence(...
            start_freq:end_freq, ...
            start_time:end_time);
    end

    avg_thin = nanmean(thin(:));

    if strcmp(type, 'pow')
        thick = squeeze(participant.thick.powspctrm(electrode, ...
            start_freq:end_freq, ...
            start_time:end_time));
    else
        thick = participant.thick.inter_trial_coherence(...
            start_freq:end_freq, ...
            start_time:end_time);
    end
    avg_thick = nanmean(thick(:));


    if strcmp(type, 'pow')
        med = squeeze(participant.med.powspctrm(electrode, ...
            start_freq:end_freq, ...
            start_time:end_time));
    else
        med = participant.med.inter_trial_coherence(...
            start_freq:end_freq, ...
            start_time:end_time);
    end
    avg_med = nanmean(med(:));


    avg_pgi = avg_med - (avg_thin + avg_thick)/2;

    participant_data(i, 1) = avg_pgi;
    participant_data(i,2) = participant_number;
end
end

%% used to format the frequency data for plotting functions
function freq = format_for_plotting_functions(freq)
num_participants = numel(freq);
for i=1:num_participants
    participant = freq{i};

    thin = squeeze(mean(participant.thin.powspctrm,2));
    thick = squeeze(mean(participant.thick.powspctrm,2));
    medium = squeeze(mean(participant.med.powspctrm,2));
    pgi = squeeze(mean(participant.pgi.powspctrm,2));
    time = participant.pgi.time;

    new_struct.thin = thin;
    new_struct.thick = thick;
    new_struct.medium = medium;
    new_struct.pgi = pgi;
    new_struct.avg = pgi;
    new_struct.time = time;

    freq{i} = new_struct;
end

end

function extract_timeseries_through_time(data)



    time = 0.134;
    location = 'A12';

    number_of_trials = size(data, 2);
    EEG_timeseries = [];
    for k=1:number_of_trials
        participants = data{k};
        number_of_participants = size(participants, 2);
        participant_data = [];
        for p=1:number_of_participants
            participant = participants{p};
            PGI_data = participant.avg;
            
            location_electrode = find(contains(participant.label,location));
            [~, location_time] = min( abs( participant.time-time ) );
            EEG_value = PGI_data(location_electrode, location_time);
            participant_data(p) = EEG_value;
        end
        avg_participant_eeg = mean(participant_data);
        EEG_timeseries(k) = avg_participant_eeg;
    end

    disp(EEG_timeseries);
end

%% combine images together for visual purposes
function combine_images(save_dir)

    clusters_through_time = imread(save_dir + "/positive_cluster.png");
    topographic_maps = imread(save_dir + "/positive_topographic.png");
    erps = imread(save_dir + "/positive_peak_erp_1.png");
    highlighted_electrode = imread(save_dir + "/highlighted_electrode.png");

    % remove whitespace from topographic maps
    top = topographic_maps(1:1200, :, :);
    bottom = topographic_maps(1850:end, :, :);
    topographic_maps = [top; bottom];

    
    clusters_through_time = imresize(clusters_through_time, [size(clusters_through_time, 1)/2, size(clusters_through_time,2)/2])
    erps = imresize(erps, [size(erps, 1)*4, size(erps,2)*4]);

    x = imtile({erps, clusters_through_time, topographic_maps}, 'GridSize', [3, 1], ...
        'BackgroundColor', 'white');

    path = save_dir + "/combined.png";
    imwrite(x,path,'JPEG');


    disp('1')
end

%% load post-processed fildtrip data
function [ft, particpant_order] = ...
    load_one_postprocessed_data(participant_main_path, filename, ...
    partition, i)
    
    disp(strcat('LOADING PARTICIPANT...', int2str(i)));
    
    if exist(participant_main_path, 'dir')
        cd(participant_main_path);
    
        if isfile(filename)
            load(filename);
    
            ft.label = data.label;
            ft.time = data.time{1};
            ft.trialinfo = [1];
            ft.elec = data.elec;
            ft.dimord = 'chan_time';
        
            % find the condition labels used to match up the data
        
            if partition.is_partition
                if partition.partition_number == 1
                    if isfield(data, 'p1_pgi')
                        pgi = data.p1_pgi;
                    end
                    thin = data.p1_thin;
                    med = data.p1_med;
                    thick = data.p1_thick;
                elseif partition.partition_number == 2
                    if isfield(data, 'p2_pgi')
                        pgi = data.p2_pgi;
                    end
                    thin = data.p2_thin;
                    med = data.p2_med;
                    thick = data.p2_thick;
                elseif partition.partition_number == 3
                    if isfield(data, 'p3_pgi')
                        pgi = data.p3_pgi;
                    end
                    thin = data.p3_thin;
                    med = data.p3_med;
                    thick = data.p3_thick;
                else
                    thin = data.thin;
                    med = data.med;
                    thick = data.thick;
                end
            elseif ~partition.is_partition
                try
                    pgi = data.med - (data.thin + data.thick)/2;
                    ft.avg = pgi;
                catch
                    disp('freq')
                end
                thin = data.thin;
                med = data.med;
                thick = data.thick;
        
            end
        
            if isfield(data, 'p1_pgi') || isfield(data, 'p2_pgi') || isfield(data, 'p3_pgi')
                ft.avg = pgi;
            end
        
            ft.thin = thin;
            ft.med = med;
            ft.thick = thick;
        
            if isfield(data, 'thick_order') || isfield(data, 'thin_order') ...
                    || isfield(data, 'med_order')
                ft.thick_order = data.thick_order;
                ft.thin_order = data.thin_order;
                ft.med_order = data.med_order;
            end
        
            ft = rmfield(ft, "trialinfo");
            particpant_order = i;
        else
            particpant_order = -9999;
            ft = - 9999;
        end
    else
            particpant_order = -9999;
            ft = - 9999;
    end
end
%% aggregated avg for selecting frequencies
function create_agg_average(freq)

    x = size(participant.thin.powspctrm, 1);
    y = size(participant.thin.powspctrm, 2);
    z = size(participant.thin.powspctrm, 3);

    agg_avgs = zeros(1, x, y, z);

    for i =1:size(freq, 2)
        participant = freq{i};
        
        tf_hann = participant.thin;
        thin = participant.thin.powspctrm;
        thick = participant.thick.powspctrm;
        med = participant.med.powspctrm;    

        agg_avg = (thin + med + thick)/3;
        agg_avgs(i, :, :, :) = agg_avg;
    end

    aggregated_average_across_participants = squeeze(mean(agg_avgs));

    dummy_participant = freq{1}.thin;
    dummy_participant.powspctrm = aggregated_average_across_participants;

    cfg = [];
    cfg.channel = 'A23';
    cfg.baseline     = [-0.5 0];
    cfg.baselinetype = 'absolute';
    cfg.zlim = [-2, 2];
    ft_singleplotTFR(cfg, dummy_participant);

end
%end