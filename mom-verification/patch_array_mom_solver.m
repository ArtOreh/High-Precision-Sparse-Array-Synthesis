clear; clc; close all;

%% 1. Physical Parameters
f_center = 2.4e9;              
c = 3e8;                       
lambda = c / f_center;         
Aperture_lambda = 42.0;         
Aperture_meters = Aperture_lambda * lambda; 

pts_normalized = [0, 0.14028429855149321, 0.20817901966840463, 0.25098483964450696, ...
                  0.28985099930867564, 0.3132479771891962, 0.3371132340284352, 0.3661198756807549, ...
                  0.3866899852730659, 0.40709809029235466, 0.4273165823812088, 0.4461040173032893, ...
                  0.46605796312747455, 0.4826586533293225, 0.4979911132854833, 0.5159991492648616, ...
                  0.5304735602759582, 0.5483120444347149, 0.5607728567338037, 0.577950359764457, ...
                  0.5925425703878104, 0.6087008569699883, 0.6227141048073231, 0.6401352141679311, ...
                  0.6527743610786504, 0.6702197027306397, 0.6844265024637199, 0.7025573910807097, ...
                  0.7169730670479489, 0.7328871719546651, 0.7487526178880747, 0.7665758349427597, ...
                  0.7820679725888011, 0.8014807300051912, 0.820965224462642, 0.8421380542482239, ...
                  0.8673268217058332, 0.8897741573092661, 0.914214640731509, 0.9435423852894256, ...
                  0.9732663687708607, 1];

%% 2. Conversion to Centered Physical Coordinates (in Meters) and Patch Generation
pts_centered = pts_normalized - 0.5; 
x_coordinates = pts_centered * Aperture_meters; 
spacings = diff(x_coordinates); 

% Create a single microstrip patch element
p = patchMicrostrip;
p = design(p, f_center); 

% Compute the uniform structural allocation length for each element's ground plane.
% Dividing the total expanded aperture by the number of elements ensures 
% continuous overlapping of the individual sub-elements along the x-axis,
% thereby forcing the solver to generate a monolithic, seamless ground plane.
p.GroundPlaneLength = (Aperture_meters + 0.2) / length(pts_normalized); 
p.Substrate.Length  = (Aperture_meters + 0.2) / length(pts_normalized);

% Set the board width proportional to the wavelength
p.GroundPlaneWidth  = 1.2 * lambda; 
p.Substrate.Width   = 1.2 * lambda;



%% Print Final Antenna Parameters for the Manuscript
fprintf('\n--- Final Patch Element Parameters for the Manuscript ---\n');
fprintf('  Operating Frequency: %.2f GHz\n', f_center / 1e9);
fprintf('  Wavelength (lambda): %.2f mm\n', lambda * 1000);
fprintf('  Patch Length: %.2f mm\n', p.Length * 1000);
fprintf('  Patch Width:  %.2f mm\n', p.Width * 1000);
fprintf('  Substrate Height (Thickness): %.2f mm\n', p.Height * 1000);
fprintf('  Substrate Material: %s (EpsilonR = %.1f, LossTangent = %.3f)\n', ...
    p.Substrate.Name, p.Substrate.EpsilonR, p.Substrate.LossTangent);

fprintf('  Conductor Material: %s\n', p.Conductor.Name);
fprintf('  Conductor Conductivity: %.2e S/m\n', p.Conductor.Conductivity);
if p.Conductor.Thickness == 0
    fprintf('  Conductor (Patch) Thickness: 0.00 mm (Infinitesimally thin PEC)\n');
else
    fprintf('  Conductor (Patch) Thickness: %.3f mm\n', p.Conductor.Thickness * 1000);
end

fprintf('  Feed Offset (X, Y): [%.2f, %.2f] mm\n', p.FeedOffset(1)*1000, p.FeedOffset(2)*1000);
fprintf('=========================================================\n');

%% 3. Generate the Linear Array
array = linearArray;
array.Element = p;
array.NumElements = length(pts_normalized);
array.ElementSpacing = spacings; 


meshData = mesh(array, 'MaxEdgeLength', lambda/18);

% Print the number of triangles directly to the command window (console)
fprintf('  Number of triangles: %d\n', meshData.NumTriangles);

%% 4. Geometry Visualization
figure;
show(array);
title('Physical Model of the Patch Array');

figure;
mesh(array); 
title('Computational Mesh of the Patch Array');
%pause;
%quit;

%% 5. Calculation and Plotting of 2D Radiation Pattern Cut in the XZ-Plane (Cartesian Coordinates)
% Define elevation angles from -90 to 270 degrees to fully cover the XZ-plane
el_angles = -90:0.1:270; 
az_angle = 0; % Fix azimuth at 0 degrees (defines the XZ-plane)

[dir_values, az, el] = pattern(array, f_center, az_angle, el_angles);

% Convert elevation angles so that 0 degrees corresponds to Zenith (Z-axis),
% while left and right angles extend to -90 and +90 degrees (off-vertical deviation)
theta_deg = 90 - el;

% Sort data by ascending angle from -90 to 90 degrees for correct plot generation
[theta_sorted, idx] = sort(theta_deg);
dir_sorted = dir_values(idx);

% Restrict the range strictly from -90 to +90 degrees relative to the Z-axis
valid_range = (theta_sorted >= -90) & (theta_sorted <= 90);
theta_plot = theta_sorted(valid_range);
dir_plot = dir_sorted(valid_range);

% NORMALIZED linear 2D plot
dir_normalized = dir_plot - max(dir_plot);

figure;
plot(theta_plot, dir_normalized, 'LineWidth', 2, 'Color', [0.8500 0.3250 0.0980]);
grid on;
xlim([-90 90]);
ylim([-40 3]);
xlabel('Deviation angle from zenith \theta (degrees)');
ylabel('Normalized Directivity (dB)');
title('Normalized Radiation Pattern Section in XZ-Plane');

%% 6. Data Export
export_data = [theta_plot(:), dir_normalized(:)];

% Save to an ASCII text file using space delimiters
save('patch_array_pattern_42_42_mb1.5.txt', 'export_data', '-ascii');

fprintf('=========================================\n');
fprintf('  DATA SAVED SUCCESSFULLY!\n');
fprintf('  File: patch_array_pattern_42_42_mb1.5.txt\n');
fprintf('=========================================\n');
