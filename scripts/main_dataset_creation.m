function main_dataset_creation()
    % MAIN_DATASET_CREATION - Complete pipeline for creating the final maritime dataset
    %
    % This is the main script that orchestrates the entire process of creating
    % the final dataset for pilot optimization analysis. It combines:
    % 1. Dynamic ship tracking data (AIS)
    % 2. Static ship specifications
    % 3. Weather and atmospheric data
    % 4. Oceanographic and bathymetric data
    % 5. Derived safety and navigation parameters
    %
    % Usage:
    %   Run this script to create the complete final dataset
    %   The script will automatically call the required functions in sequence
    %
    % Output:
    %   - required final data set.csv (complete dataset ready for analysis)
    %
    % Requirements:
    %   - MATLAB with required toolboxes
    %   - Input data files in ../dataSet/ directory
    %   - Sufficient memory for processing large datasets
    %
    % For 84-day analysis:
    %   - Modify file paths to include multiple days of data
    %   - Ensure adequate computational resources
    %   - Consider processing in batches for very large datasets
    
    clc;
    clear;
    close all;
    
    fprintf('========================================\n');
    fprintf('MARITIME DATASET CREATION PIPELINE\n');
    fprintf('========================================\n');
    fprintf('Creating comprehensive dataset for pilot optimization\n');
    fprintf('Date: %s\n', datestr(now));
    fprintf('Location: Busan Port, South Korea\n');
    fprintf('========================================\n\n');
    
    % Record start time
    start_time = tic;
    
    try
        % Step 1: Create merged dataset (dynamic + static)
        fprintf('PHASE 1: MERGING DYNAMIC AND STATIC DATA\n');
        fprintf('----------------------------------------\n');
        create_final_dataset();
        fprintf('Phase 1 completed successfully!\n\n');
        
        % Step 2: Add environmental data
        fprintf('PHASE 2: ADDING ENVIRONMENTAL DATA\n');
        fprintf('----------------------------------\n');
        add_environmental_data();
        fprintf('Phase 2 completed successfully!\n\n');
        
        % Step 3: Validate and summarize the final dataset
        fprintf('PHASE 3: VALIDATION AND SUMMARY\n');
        fprintf('-------------------------------\n');
        validate_final_dataset();
        fprintf('Phase 3 completed successfully!\n\n');
        
        % Calculate total processing time
        total_time = toc(start_time);
        
        fprintf('========================================\n');
        fprintf('DATASET CREATION COMPLETED SUCCESSFULLY!\n');
        fprintf('========================================\n');
        fprintf('Total processing time: %.2f seconds (%.2f minutes)\n', total_time, total_time/60);
        fprintf('Output file: required final data set.csv\n');
        fprintf('Location: ../dataSet/\n');
        fprintf('========================================\n\n');
        
        % Display next steps
        fprintf('NEXT STEPS:\n');
        fprintf('----------\n');
        fprintf('1. Review the dataset quality and completeness\n');
        fprintf('2. Perform exploratory data analysis\n');
        fprintf('3. Run pilot optimization algorithms\n');
        fprintf('4. Conduct cross-traffic analysis\n');
        fprintf('5. Generate safety assessment reports\n\n');
        
        fprintf('For 84-day analysis:\n');
        fprintf('- Prepare additional daily datasets\n');
        fprintf('- Modify file paths in the scripts\n');
        fprintf('- Consider batch processing for large datasets\n');
        fprintf('- Ensure adequate computational resources\n\n');
        
    catch ME
        fprintf('\n========================================\n');
        fprintf('ERROR OCCURRED DURING PROCESSING\n');
        fprintf('========================================\n');
        fprintf('Error message: %s\n', ME.message);
        fprintf('Error location: %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
        fprintf('========================================\n\n');
        
        fprintf('TROUBLESHOOTING STEPS:\n');
        fprintf('1. Check that all input files exist in ../dataSet/\n');
        fprintf('2. Verify MATLAB toolbox requirements\n');
        fprintf('3. Ensure sufficient memory and disk space\n');
        fprintf('4. Check file permissions\n');
        fprintf('5. Review error details above\n\n');
        
        rethrow(ME);
    end
end

function validate_final_dataset()
    % VALIDATE_FINAL_DATASET - Validates the created dataset and provides summary
    
    fprintf('Loading final dataset for validation...\n');
    
    % Load the final dataset
    dataset_file = '../dataSet/required final data set.csv';
    
    if ~isfile(dataset_file)
        error('Final dataset file not found: %s', dataset_file);
    end
    
    data = readtable(dataset_file, 'VariableNamingRule', 'preserve');
    
    fprintf('Dataset loaded successfully\n');
    fprintf('Dimensions: %d rows × %d columns\n\n', height(data), width(data));
    
    % Validation checks
    fprintf('VALIDATION CHECKS:\n');
    fprintf('-----------------\n');
    
    % Check 1: Data completeness
    fprintf('1. Data Completeness:\n');
    total_cells = height(data) * width(data);
    missing_cells = sum(sum(ismissing(data)));
    completeness = (total_cells - missing_cells) / total_cells * 100;
    fprintf('   Overall completeness: %.1f%%\n', completeness);
    
    % Check critical columns
    critical_columns = {'MMSI', 'Latitude', 'Longitude', 'DateTime', 'SOG'};
    for i = 1:length(critical_columns)
        col_name = critical_columns{i};
        if any(strcmp(data.Properties.VariableNames, col_name))
            missing_count = sum(ismissing(data.(col_name)));
            fprintf('   %s: %.1f%% complete\n', col_name, (height(data)-missing_count)/height(data)*100);
        end
    end
    
    % Check 2: Coordinate validity
    fprintf('\n2. Coordinate Validity:\n');
    lat_valid = sum(data.Latitude >= 34.5 & data.Latitude <= 35.5);
    lon_valid = sum(data.Longitude >= 128.5 & data.Longitude <= 129.5);
    fprintf('   Valid latitudes (34.5-35.5°N): %d/%d (%.1f%%)\n', ...
        lat_valid, height(data), lat_valid/height(data)*100);
    fprintf('   Valid longitudes (128.5-129.5°E): %d/%d (%.1f%%)\n', ...
        lon_valid, height(data), lon_valid/height(data)*100);
    
    % Check 3: Speed validity
    fprintf('\n3. Speed Validity:\n');
    valid_sog = sum(data.SOG >= 0 & data.SOG <= 30); % Reasonable speed range
    fprintf('   Valid SOG (0-30 kts): %d/%d (%.1f%%)\n', ...
        valid_sog, height(data), valid_sog/height(data)*100);
    
    % Check 4: Environmental data ranges
    fprintf('\n4. Environmental Data Ranges:\n');
    if any(strcmp(data.Properties.VariableNames, 'WindSpeed_kts'))
        wind_valid = sum(data.WindSpeed_kts >= 0 & data.WindSpeed_kts <= 50);
        fprintf('   Valid wind speed (0-50 kts): %d/%d (%.1f%%)\n', ...
            wind_valid, height(data), wind_valid/height(data)*100);
    end
    
    if any(strcmp(data.Properties.VariableNames, 'SeawaterDepth_m'))
        depth_valid = sum(data.SeawaterDepth_m > 0 & data.SeawaterDepth_m <= 1000);
        fprintf('   Valid water depth (0-1000 m): %d/%d (%.1f%%)\n', ...
            depth_valid, height(data), depth_valid/height(data)*100);
    end
    
    % Check 5: UKC safety analysis
    fprintf('\n5. Safety Analysis (UKC):\n');
    if any(strcmp(data.Properties.VariableNames, 'UKC_m'))
        ukc_data = data.UKC_m(~isnan(data.UKC_m));
        if ~isempty(ukc_data)
            safe_ukc = sum(ukc_data >= 2); % Safe UKC threshold
            critical_ukc = sum(ukc_data < 1); % Critical UKC threshold
            fprintf('   Records with safe UKC (≥2m): %d/%d (%.1f%%)\n', ...
                safe_ukc, length(ukc_data), safe_ukc/length(ukc_data)*100);
            fprintf('   Records with critical UKC (<1m): %d/%d (%.1f%%)\n', ...
                critical_ukc, length(ukc_data), critical_ukc/length(ukc_data)*100);
        end
    end
    
    % Summary statistics
    fprintf('\nDATASET SUMMARY STATISTICS:\n');
    fprintf('--------------------------\n');
    
    % Vessel statistics
    unique_vessels = length(unique(data.MMSI));
    fprintf('Unique vessels: %d\n', unique_vessels);
    
    % Time range
    if any(strcmp(data.Properties.VariableNames, 'DateTime'))
        min_time = min(data.DateTime);
        max_time = max(data.DateTime);
        fprintf('Time range: %d - %d\n', min_time, max_time);
    end
    
    % Geographic coverage
    lat_range = [min(data.Latitude), max(data.Latitude)];
    lon_range = [min(data.Longitude), max(data.Longitude)];
    fprintf('Latitude range: %.4f° - %.4f°\n', lat_range(1), lat_range(2));
    fprintf('Longitude range: %.4f° - %.4f°\n', lon_range(1), lon_range(2));
    
    % Column summary
    fprintf('\nCOLUMN SUMMARY:\n');
    fprintf('--------------\n');
    column_names = data.Properties.VariableNames;
    
    % Group columns by category
    dynamic_cols = {};
    static_cols = {};
    weather_cols = {};
    ocean_cols = {};
    derived_cols = {};
    
    for i = 1:length(column_names)
        col_name = column_names{i};
        if any(contains(col_name, {'MMSI', 'DateTime', 'Latitude', 'Longitude', 'SOG', 'COG', 'Heading'}))
            dynamic_cols{end+1} = col_name;
        elseif any(contains(col_name, {'ShipType', 'LOA', 'Width', 'Draft', 'GrossTonnage'}))
            static_cols{end+1} = col_name;
        elseif any(contains(col_name, {'Wind', 'Wave', 'Air', 'Visibility'}))
            weather_cols{end+1} = col_name;
        elseif any(contains(col_name, {'Seawater', 'Tide', 'Current', 'Water', 'Salinity'}))
            ocean_cols{end+1} = col_name;
        else
            derived_cols{end+1} = col_name;
        end
    end
    
    fprintf('Dynamic data columns (%d): %s\n', length(dynamic_cols), strjoin(dynamic_cols, ', '));
    fprintf('Static data columns (%d): %s\n', length(static_cols), strjoin(static_cols, ', '));
    fprintf('Weather data columns (%d): %s\n', length(weather_cols), strjoin(weather_cols, ', '));
    fprintf('Oceanographic columns (%d): %s\n', length(ocean_cols), strjoin(ocean_cols, ', '));
    fprintf('Derived columns (%d): %s\n', length(derived_cols), strjoin(derived_cols, ', '));
    
    fprintf('\nValidation completed successfully!\n');
    
    % Generate validation report
    report_file = '../dataSet/dataset_validation_report.txt';
    generate_validation_report(data, report_file);
    fprintf('Validation report saved to: %s\n', report_file);
end

function generate_validation_report(data, report_file)
    % Generate a detailed validation report
    
    fid = fopen(report_file, 'w');
    if fid == -1
        warning('Could not create validation report file');
        return;
    end
    
    fprintf(fid, 'MARITIME DATASET VALIDATION REPORT\n');
    fprintf(fid, '==================================\n');
    fprintf(fid, 'Generated: %s\n', datestr(now));
    fprintf(fid, 'Dataset: required final data set.csv\n\n');
    
    fprintf(fid, 'DATASET OVERVIEW:\n');
    fprintf(fid, 'Rows: %d\n', height(data));
    fprintf(fid, 'Columns: %d\n', width(data));
    fprintf(fid, 'Unique vessels: %d\n', length(unique(data.MMSI)));
    
    fprintf(fid, '\nCOLUMN DETAILS:\n');
    for i = 1:width(data)
        col_name = data.Properties.VariableNames{i};
        missing_count = sum(ismissing(data.(col_name)));
        fprintf(fid, '%s: %d missing (%.1f%%)\n', col_name, missing_count, missing_count/height(data)*100);
    end
    
    fclose(fid);
end