clc; clear; close all;

% Read data from the CSV file
dynamicData = readtable('Busan_Dynamic_20230607_sorted.csv', 'VariableNamingRule', 'preserve');

% Extract relevant columns
Latitude = dynamicData.Latitude;
Longitude = dynamicData.Longitude;
MMSI = dynamicData.MMSI;
DateTime = dynamicData.DateTime;

% UTM Conversion Constants
a = 6378137; % Semi-major axis of the Earth (WGS84)
f = 1 / 298.257223563; % Flattening (WGS84)
k0 = 0.9996; % Scale factor
e2 = 2 * f - f^2; % Square of eccentricity
zone = 52; % UTM Zone for Busan
lon0 = (zone - 1) * 6 - 180 + 3; % Central meridian of the zone

% Convert Latitude and Longitude to UTM
xCart = zeros(size(Latitude));
yCart = zeros(size(Latitude));

for i = 1:length(Latitude)
    lat = deg2rad(Latitude(i));
    lon = deg2rad(Longitude(i));
    
    N = a / sqrt(1 - e2 * sin(lat)^2); % Radius of curvature
    T = tan(lat)^2;
    C = e2 / (1 - e2) * cos(lat)^2;
    A = cos(lat) * (lon - deg2rad(lon0));
    
    M = a * ((1 - e2 / 4 - 3 * e2^2 / 64 - 5 * e2^3 / 256) * lat ...
        - (3 * e2 / 8 + 3 * e2^2 / 32 + 45 * e2^3 / 1024) * sin(2 * lat) ...
        + (15 * e2^2 / 256 + 45 * e2^3 / 1024) * sin(4 * lat) ...
        - (35 * e2^3 / 3072) * sin(6 * lat));
    
    xCart(i) = k0 * N * (A + (1 - T + C) * A^3 / 6 + (5 - 18 * T + T^2 + 72 * C - 58 * e2) * A^5 / 120);
    yCart(i) = k0 * (M + N * tan(lat) * (A^2 / 2 + (5 - T + 9 * C + 4 * C^2) * A^4 / 24 ...
        + (61 - 58 * T + T^2 + 600 * C - 330 * e2) * A^6 / 720));
end

% Add Cartesian coordinates to the dynamic data table
dynamicData.X_Cartesian = xCart;
dynamicData.Y_Cartesian = yCart;

% Save the results to a new CSV file
writetable(dynamicData, 'Busan_Dynamic_XY.csv');

% Plot the ship positions
figure;
scatter(xCart, yCart, 10, 'filled');
title('Ship Positions in Cartesian Coordinates');
xlabel('X (meters)');
ylabel('Y (meters)');
grid on;

% Annotate the plot with MMSI if needed
for i = 1:10 % Limit annotations to the first 10 points to avoid clutter
    text(xCart(i), yCart(i), sprintf('%d', MMSI(i)), 'FontSize', 8, 'HorizontalAlignment', 'right');
end

disp('Conversion completed and results saved as Busan_Dynamic_XY.csv');