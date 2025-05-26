# Usage Instructions

## Quick Start Guide

### Prerequisites
1. **MATLAB** with the following toolboxes:
   - Statistics and Machine Learning Toolbox
   - Mapping Toolbox
   - Signal Processing Toolbox

2. **Input Data Files** (place in `data/` directory):
   - `Busan_Dynamic_20230607_sorted.csv` - Dynamic ship tracking data
   - `Static_Busan_Dynamic_20230607.csv` - Static vessel specifications
   - `gebco_2024_n35.149_s34.691_w128.6523_e129.0327.nc` - Bathymetry data (optional)
   - `hourly_tide_heights.csv` - Tidal data (optional)

### Running the Complete Dataset Creation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/TanveerAhmedKhan/maritime-pilot-optimization.git
   cd maritime-pilot-optimization
   ```

2. **Open MATLAB and navigate to the scripts directory:**
   ```matlab
   cd scripts
   ```

3. **Run the main dataset creation pipeline:**
   ```matlab
   main_dataset_creation()
   ```

This will:
- Merge dynamic and static ship data
- Add weather and oceanographic data
- Calculate safety parameters
- Generate the final dataset: `required final data set.csv`

### Step-by-Step Execution

If you prefer to run individual steps:

```matlab
% Step 1: Merge dynamic and static data
create_final_dataset()

% Step 2: Add environmental data
add_environmental_data()

% Step 3: Validate results (optional)
validate_final_dataset()
```

### Testing and Validation

To test the process with Python (for validation):
```bash
cd scripts
python validate_output.py
```

## Expected Output

### Final Dataset Structure
The output file `required final data set.csv` will contain:

**Dynamic Data (8 columns):**
- MMSI, DateTime, Latitude, Longitude, SOG, SOG_ms, COG, Heading

**Static Data (5 columns):**
- ShipType, LOA, Width, Draft, GrossTonnage

**Weather Data (8 columns):**
- WindSpeed_kts, WindDirection_deg, WaveHeight_m, WavePeriod_s, WaveDirection_deg, AirTemperature_C, AirPressure_hPa, Visibility_km

**Oceanographic Data (6 columns):**
- SeawaterDepth_m, TideHeight_m, CurrentSpeed_kts, CurrentDirection_deg, WaterTemperature_C, Salinity_psu

**Derived Parameters (3 columns):**
- UKC_m (Under Keel Clearance), DistanceToPort_nm, TimeToPort_hours

### Dataset Statistics
- **Rows:** ~459,473 (all dynamic records)
- **Columns:** 30 total
- **Static Data Coverage:** ~96.9% (126/130 vessels)
- **Geographic Coverage:** Busan port area (34.5-35.5°N, 128.5-129.5°E)
- **Time Coverage:** June 7, 2023 (single day)

## Extending to Multiple Days

To process 84 days of data:

1. **Prepare data files** with naming convention:
   - `Busan_Dynamic_20230607_sorted.csv`
   - `Busan_Dynamic_20230608_sorted.csv`
   - ... (continue for all days)

2. **Modify the scripts** to loop through multiple files:
   ```matlab
   % In create_final_dataset.m, add loop:
   for day = 7:90  % June 7 to August 30 (84 days)
       dynamic_file = sprintf('../data/Busan_Dynamic_202306%02d_sorted.csv', day);
       % Process each day...
   end
   ```

## Troubleshooting

### Common Issues

1. **"File not found" errors:**
   - Ensure data files are in the correct directory
   - Check file names match exactly
   - Verify file paths in scripts

2. **Memory issues:**
   - Close other applications
   - Increase MATLAB memory allocation
   - Process data in smaller batches

3. **NetCDF reading errors:**
   - Install required MATLAB toolboxes
   - Check NetCDF file integrity
   - Scripts will fall back to estimated depths

4. **Coordinate validation failures:**
   - Verify GPS coordinates are in decimal degrees
   - Check for reasonable lat/lon ranges

### Performance Tips

- Use SSD storage for faster I/O
- Increase available RAM
- Use MATLAB Parallel Computing Toolbox for large datasets
- Pre-filter data to reduce processing overhead

## Output Validation

The scripts include comprehensive validation:
- Data structure verification
- Geographic coordinate validation
- Speed and environmental parameter ranges
- Safety parameter calculations
- Missing data analysis

Check the validation report: `dataset_validation_report.txt`

## Next Steps

After creating the dataset, you can:
1. Run pilot optimization analysis
2. Perform cross-traffic analysis
3. Conduct safety assessments
4. Generate route optimization studies
5. Extend to 84-day analysis

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the validation output
3. Examine the generated validation report
4. Refer to the main README.md for detailed documentation