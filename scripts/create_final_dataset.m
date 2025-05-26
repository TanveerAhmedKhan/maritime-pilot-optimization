function create_final_dataset()
    % CREATE_FINAL_DATASET - Merges dynamic and static ship data and prepares for environmental data
    %
    % This function:
    % 1. Reads dynamic ship data (AIS tracking data)
    % 2. Reads static ship data (vessel specifications)
    % 3. Merges static data into dynamic data for each row based on MMSI
    % 4. Prepares columns for weather and seawater depth data
    % 5. Saves the result as 'required final data set.csv'
    %
    % Input files required:
    % - Busan_Dynamic_20230607_sorted.csv (dynamic data)
    % - Static_Busan_Dynamic_20230607.csv (static data)
    %
    % Output:
    % - required final data set.csv (merged dataset ready for environmental data)
    
    clc;
    clear;
    close all;
    
    fprintf('=== Creating Final Dataset ===\n');
    fprintf('Step 1: Reading input files...\n');
    
    % Define file paths
    dynamic_file = '../dataSet/Busan_Dynamic_20230607_sorted.csv';
    static_file = '../dataSet/Static_Busan_Dynamic_20230607.csv';
    output_file = '../dataSet/required final data set.csv';
    
    % Check if input files exist
    if ~isfile(dynamic_file)
        error('Dynamic data file not found: %s', dynamic_file);
    end
    if ~isfile(static_file)
        error('Static data file not found: %s', static_file);
    end
    
    % Read dynamic data (AIS tracking data)
    fprintf('Reading dynamic data...\n');
    dynamic_data = readtable(dynamic_file, 'VariableNamingRule', 'preserve');
    fprintf('Dynamic data loaded: %d rows, %d columns\n', height(dynamic_data), width(dynamic_data));
    
    % Read static data (vessel specifications)
    fprintf('Reading static data...\n');
    static_data = readtable(static_file, 'VariableNamingRule', 'preserve');
    fprintf('Static data loaded: %d rows, %d columns\n', height(static_data), width(static_data));
    
    % Display column information
    fprintf('\nDynamic data columns: ');
    fprintf('%s ', dynamic_data.Properties.VariableNames{:});
    fprintf('\n');
    
    fprintf('Static data columns: ');
    fprintf('%s ', static_data.Properties.VariableNames{:});
    fprintf('\n\n');
    
    fprintf('Step 2: Merging static data into dynamic data...\n');
    
    % Get unique MMSI values from dynamic data
    unique_mmsi_dynamic = unique(dynamic_data.MMSI);
    unique_mmsi_static = unique(static_data.MMSI);
    
    fprintf('Unique vessels in dynamic data: %d\n', length(unique_mmsi_dynamic));
    fprintf('Unique vessels in static data: %d\n', length(unique_mmsi_static));
    
    % Find MMSI values that exist in dynamic but not in static
    missing_static = setdiff(unique_mmsi_dynamic, unique_mmsi_static);
    if ~isempty(missing_static)
        fprintf('Warning: %d vessels in dynamic data have no static data:\n', length(missing_static));
        for i = 1:min(10, length(missing_static))  % Show first 10
            fprintf('  MMSI: %d\n', missing_static(i));
        end
        if length(missing_static) > 10
            fprintf('  ... and %d more\n', length(missing_static) - 10);
        end
    end
    
    % Initialize new columns for static data in dynamic dataset
    num_rows = height(dynamic_data);
    
    % Pre-allocate arrays for static data columns
    ship_type = NaN(num_rows, 1);
    loa = NaN(num_rows, 1);
    width = NaN(num_rows, 1);
    draft = NaN(num_rows, 1);
    gross_tonnage = NaN(num_rows, 1);
    
    % Create lookup table for faster processing
    fprintf('Creating MMSI lookup table...\n');
    static_lookup = containers.Map('KeyType', 'double', 'ValueType', 'any');
    
    for i = 1:height(static_data)
        mmsi = static_data.MMSI(i);
        static_info = struct();
        static_info.ShipType = static_data.ShipType(i);
        static_info.LOA = static_data.LOA(i);
        static_info.Width = static_data.Width(i);
        static_info.Draft = static_data.Draft(i);
        static_info.grossTonnage = static_data.grossTonnage(i);
        
        static_lookup(num2str(mmsi)) = static_info;
    end
    
    fprintf('Step 3: Mapping static data to dynamic records...\n');
    
    % Progress tracking
    progress_interval = floor(num_rows / 20);  % Update every 5%
    matched_count = 0;
    
    % Map static data to each dynamic record
    for i = 1:num_rows
        if mod(i, progress_interval) == 0
            fprintf('Progress: %.1f%% (%d/%d rows)\n', (i/num_rows)*100, i, num_rows);
        end
        
        mmsi = dynamic_data.MMSI(i);
        mmsi_key = num2str(mmsi);
        
        if isKey(static_lookup, mmsi_key)
            static_info = static_lookup(mmsi_key);
            ship_type(i) = static_info.ShipType;
            loa(i) = static_info.LOA;
            width(i) = static_info.Width;
            draft(i) = static_info.Draft;
            gross_tonnage(i) = static_info.grossTonnage;
            matched_count = matched_count + 1;
        end
        % If no match found, values remain NaN (already initialized)
    end
    
    fprintf('Mapping complete: %d/%d records matched (%.1f%%)\n', ...
        matched_count, num_rows, (matched_count/num_rows)*100);
    
    fprintf('Step 4: Adding static data columns to dynamic dataset...\n');
    
    % Add static data columns to dynamic dataset
    dynamic_data.ShipType = ship_type;
    dynamic_data.LOA = loa;
    dynamic_data.Width = width;
    dynamic_data.Draft = draft;
    dynamic_data.GrossTonnage = gross_tonnage;
    
    fprintf('Step 5: Preparing columns for environmental data...\n');
    
    % Add placeholder columns for weather data (to be filled later)
    dynamic_data.WindSpeed_kts = NaN(num_rows, 1);          % Wind speed in knots
    dynamic_data.WindDirection_deg = NaN(num_rows, 1);      % Wind direction in degrees
    dynamic_data.WaveHeight_m = NaN(num_rows, 1);           % Significant wave height in meters
    dynamic_data.WavePeriod_s = NaN(num_rows, 1);           % Wave period in seconds
    dynamic_data.WaveDirection_deg = NaN(num_rows, 1);      % Wave direction in degrees
    dynamic_data.AirTemperature_C = NaN(num_rows, 1);       % Air temperature in Celsius
    dynamic_data.AirPressure_hPa = NaN(num_rows, 1);        % Air pressure in hPa
    dynamic_data.Visibility_km = NaN(num_rows, 1);          % Visibility in kilometers
    
    % Add placeholder columns for oceanographic data (to be filled later)
    dynamic_data.SeawaterDepth_m = NaN(num_rows, 1);        % Water depth in meters
    dynamic_data.TideHeight_m = NaN(num_rows, 1);           % Tide height in meters
    dynamic_data.CurrentSpeed_kts = NaN(num_rows, 1);       % Current speed in knots
    dynamic_data.CurrentDirection_deg = NaN(num_rows, 1);   % Current direction in degrees
    dynamic_data.WaterTemperature_C = NaN(num_rows, 1);     % Water temperature in Celsius
    dynamic_data.Salinity_psu = NaN(num_rows, 1);           % Salinity in practical salinity units
    
    % Add derived/calculated columns (to be filled later)
    dynamic_data.UKC_m = NaN(num_rows, 1);                  % Under Keel Clearance in meters
    dynamic_data.DistanceToPort_nm = NaN(num_rows, 1);      % Distance to port in nautical miles
    dynamic_data.TimeToPort_hours = NaN(num_rows, 1);       % Estimated time to port in hours
    
    fprintf('Step 6: Saving final dataset...\n');
    
    % Save the merged dataset
    writetable(dynamic_data, output_file);
    
    fprintf('Dataset saved successfully to: %s\n', output_file);
    fprintf('Final dataset dimensions: %d rows, %d columns\n', height(dynamic_data), width(dynamic_data));
    
    % Display summary statistics
    fprintf('\n=== Dataset Summary ===\n');
    fprintf('Total records: %d\n', height(dynamic_data));
    fprintf('Date range: %s\n', '2023-06-07 (single day)');
    fprintf('Unique vessels: %d\n', length(unique(dynamic_data.MMSI)));
    
    % Static data coverage
    fprintf('\nStatic Data Coverage:\n');
    fprintf('  Records with ship type: %d (%.1f%%)\n', ...
        sum(~isnan(dynamic_data.ShipType)), sum(~isnan(dynamic_data.ShipType))/num_rows*100);
    fprintf('  Records with LOA: %d (%.1f%%)\n', ...
        sum(~isnan(dynamic_data.LOA)), sum(~isnan(dynamic_data.LOA))/num_rows*100);
    fprintf('  Records with draft: %d (%.1f%%)\n', ...
        sum(~isnan(dynamic_data.Draft)), sum(~isnan(dynamic_data.Draft))/num_rows*100);
    
    % Display column names of final dataset
    fprintf('\nFinal Dataset Columns (%d total):\n', width(dynamic_data));
    column_names = dynamic_data.Properties.VariableNames;
    for i = 1:length(column_names)
        fprintf('  %2d. %s\n', i, column_names{i});
    end
    
    fprintf('\n=== Next Steps ===\n');
    fprintf('1. Add weather data using weather API or meteorological files\n');
    fprintf('2. Add seawater depth data using bathymetry files (NetCDF)\n');
    fprintf('3. Add tidal data using tide prediction models\n');
    fprintf('4. Calculate derived parameters (UKC, distances, etc.)\n');
    fprintf('5. Validate and clean the final dataset\n');
    
    fprintf('\n=== Process Complete ===\n');
    
end