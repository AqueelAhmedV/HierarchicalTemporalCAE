import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
import matplotlib.animation as animation




def read_coordinates(filepath):
    """
    Read coordinates from OpenFOAM points file
    
    Args:
        filepath (str): Path to the points file
        
    Returns:
        numpy.ndarray: Array of coordinates with shape (n_points, 3)
    """
    
    
    coordinates = []
    
    with open(filepath, 'r') as f:
        # Skip header until we find the delimiter
        for line in f:
            if '// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //' in line:
                break
        
        # Skip any empty lines
        for line in f:
            if line.strip():
                # First non-empty line should be number of points
                n_points = int(line.strip())
                break
        
        # Skip the opening parenthesis line
        f.readline()
        
        # Read coordinates until we hit closing parenthesis or reach n_points
        point_count = 0
        for line in f:
            line = line.strip()
            
            # Stop if we hit closing parenthesis or read all points
            if line == ')' or point_count >= n_points:
                break
                
            # Parse coordinate line
            if line.startswith('(') and line.endswith(')'):
                # Remove parentheses and split into components
                coords = line[1:-1].split()
                # Convert to floats
                coords = [float(x) for x in coords]
                coordinates.append(coords)
                point_count += 1
    
    return np.array(coordinates)

def read_velocity_field(filepath):
    """
    Read velocity field from OpenFOAM U file
    
    Args:
        filepath (str): Path to the U (velocity) file
        
    Returns:
        numpy.ndarray: Array of velocities with shape (n_points, 3)
    """
    
    
    velocities = []
    
    with open(filepath, 'r') as f:
        # Skip header until we find 'internalField   nonuniform List<vector>'
        for line in f:
            if 'internalField   nonuniform List<vector>' in line:
                break
        
        # Read the number of velocity points
        n_points = int(f.readline().strip())
        
        # Skip the opening parenthesis
        f.readline()
        
        # Read velocities until we hit closing parenthesis or reach n_points
        point_count = 0
        for line in f:
            line = line.strip()
            
            # Stop if we hit closing parenthesis or read all points
            if line == ')' or point_count >= n_points:
                break
                
            # Parse velocity line
            if line.startswith('(') and line.endswith(')'):
                # Remove parentheses and split into components
                vels = line[1:-1].split()
                # Convert to floats
                vels = [float(x) for x in vels]
                velocities.append(vels)
                point_count += 1
    
    return np.array(velocities)

def process_data(coords, velocities):
    """
    Process coordinates and velocities into a single dataset:
    1. Filter coordinates to keep only z=1 points and extract (x,y)
    2. Remove w component from velocities, keeping only (u,v)
    3. Combine into a single array of shape (n_points, 4) containing (x,y,u,v)
    
    Args:
        coords: numpy.ndarray with shape (n_points, 3) for (x,y,z)
        velocities: numpy.ndarray with shape (n_points, 3) for (u,v,w)
        
    Returns:
        numpy.ndarray: Combined dataset with shape (n_points, 4) for (x,y,u,v)
    """
    
    
    # First find indices where z=1
    z_one_indices = np.where(coords[:, 2] == 0)[0]
    
    # Get corresponding velocities (using only indices that exist in velocities array)
    valid_indices = z_one_indices[z_one_indices < len(velocities)]
    
    # Extract x,y coordinates and u,v velocities
    coords_2d = coords[valid_indices][:, :2]    # Only x,y components
    vels_2d = velocities[valid_indices][:, :2]  # Only u,v components
    
    # Combine into single dataset
    dataset = np.hstack((coords_2d, vels_2d))
    
    print(f"Original shapes - Coords: {coords.shape}, Velocities: {velocities.shape}")
    print(f"Final dataset shape: {dataset.shape} (format: x,y,u,v)")
    
    return dataset

def prepare_for_autoencoder(dataset, grid_size=(256, 80)):  # Changed from 88 to 80
    """
    Prepare the (x,y,u,v) dataset for the autoencoder by:
    1. Reshaping into a grid
    2. Separating velocities for training
    
    Args:
        dataset: numpy array of shape (n_points, 4) containing (x,y,u,v)
        grid_size: tuple of (height, width) for desired grid
        
    Returns:
        numpy array of shape (1, height, width, 2) containing (u,v) components
    """
    
    
    coords = dataset[:, :2]  # (x,y)
    vels = dataset[:, 2:]    # (u,v)
    
    grid_data = np.zeros((1, grid_size[0], grid_size[1], 2))  # Shape: (1, height, width, 2)
    
    x_min, x_max = -10.0, 25.0  # Range: 35
    y_min, y_max = -10.0, 10.0  # Range: 20
    
    for i, (x, y) in enumerate(coords):
        # Convert x,y coordinates to grid indices using actual ranges
        x_idx = int((x - x_min) * (grid_size[0] - 1) / (x_max - x_min))
        y_idx = int((y - y_min) * (grid_size[1] - 1) / (y_max - y_min))
        
        if 0 <= x_idx < grid_size[0] and 0 <= y_idx < grid_size[1]:
            grid_data[0, x_idx, y_idx] = vels[i]
    
    empty_points = np.sum(np.all(grid_data[0] == 0, axis=-1))
    total_points = grid_size[0] * grid_size[1]
    print(f"Grid coverage: {(1 - empty_points/total_points)*100:.2f}% ({empty_points} empty points out of {total_points})")
    
    # Normalize the data
    non_zero_mask = ~np.all(grid_data == 0, axis=-1)
    if np.any(non_zero_mask):
        mean_vel = np.mean(grid_data[non_zero_mask])
        std_vel = np.std(grid_data[non_zero_mask])
        grid_data = (grid_data - mean_vel) / std_vel
    
    return grid_data

def analyze_coordinate_ranges(dataset):
    """
    Analyze the ranges of x and y coordinates in the dataset
    
    Args:
        dataset: numpy array of shape (n_points, 4) containing (x,y,u,v)
        
    Returns:
        dict: Dictionary containing min/max values and ranges for x and y
    """
    
    
    # Extract x and y coordinates
    x_coords = dataset[:, 0]
    y_coords = dataset[:, 1]
    
    # Calculate ranges
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    ranges = {
        'x_min': x_min,
        'x_max': x_max,
        'x_range': x_max - x_min,
        'y_min': y_min,
        'y_max': y_max,
        'y_range': y_max - y_min
    }
    
    print(f"Coordinate Ranges:")
    print(f"X: [{x_min:.4f}, {x_max:.4f}], range: {x_max-x_min:.4f}")
    print(f"Y: [{y_min:.4f}, {y_max:.4f}], range: {y_max-y_min:.4f}")
    
    return ranges

def plot_data(dataset):
    """
    Plot the processed data using coordinates and the magnitude of velocities.
    
    Args:
        dataset: numpy array of shape (n_points, 4) containing (x,y,u,v)
    """
    
    # Extract x, y coordinates and calculate the magnitude of velocities
    x = dataset[:, 0]
    y = dataset[:, 1]
    u = dataset[:, 2]
    v = dataset[:, 3]
    magnitude = np.sqrt(u**2 + v**2)  # Calculate magnitude of velocity
    
    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(x, y, c=magnitude, cmap='viridis', s=10)  # Color by magnitude
    plt.colorbar(scatter, label='Velocity Magnitude')  # Add color bar
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Scatter Plot of Velocity Magnitude')
    plt.grid()
    plt.show()

def read_csv_to_dataset(filepath):
    """
    Read a CSV file containing points and velocity data and return a dataset.
    
    Args:
        filepath (str): Path to the CSV file.
        
    Returns:
        numpy.ndarray: Array of shape (n_points, 4) containing (x, y, u, v).
    """
    
    # Read the CSV file
    df = pd.read_csv(filepath)
    
    # Filter out rows where Points_2 is 1
    df = df[df['Points_2'] != 1]
    
    # Extract relevant columns
    x = df['Points_0'].values
    y = df['Points_1'].values
    u = df['U_0'].values
    v = df['U_1'].values
    # Combine into a single dataset
    dataset = np.column_stack((x, y, u, v))
    
    return dataset

def read_timestep_data(base_dir):
    """
    Read coordinates and velocities for all timesteps in the given directory.
    Points file is read once from constant/polyMesh/points.
    
    Args:
        base_dir (str): Path to the directory containing timestep folders
        
    Returns:
        dict: Dictionary with timesteps as keys and (coordinates, velocities) as values
    """
    timestep_data = {}
    
    # Read points file once (it's the same for all timesteps)
    points_file = Path(base_dir) / 'constant/polyMesh/points'
    if not points_file.exists():
        raise FileNotFoundError(f"Points file not found at {points_file}")
    
    coords = read_coordinates(str(points_file))
    print(f"Read mesh coordinates: {coords.shape} points")
    
    # Get all subdirectories (timesteps)
    timestep_dirs = []
    for d in Path(base_dir).iterdir():
        if d.is_dir():
            if d.name == '0':
                continue
            try:
                float(d.name)  # Attempt to convert to float
                timestep_dirs.append(d)
            except ValueError:
                continue  # Skip if not a number (e.g., 'constant' directory)
    
    timestep_dirs = sorted(timestep_dirs, key=lambda x: float(x.name))
    
    for timestep_dir in timestep_dirs:
        timestep = float(timestep_dir.name)
        velocity_file = timestep_dir / 'U'
        print(velocity_file)
        if velocity_file.exists():
            vels = read_velocity_field(str(velocity_file))
            timestep_data[timestep] = (coords, vels)
            print(f"Read timestep {timestep}: {vels.shape} velocities")
        else:
            print(f"Warning: Missing velocity file for timestep {timestep}")
    
    return timestep_data

def prepare_timestep_data(base_dir, grid_size=(256, 80)):
    """
    Prepare data from all timesteps into a single sorted array.
    
    Args:
        base_dir (str): Path to the directory containing timestep folders
        grid_size (tuple): Grid size for prepare_for_autoencoder
        
    Returns:
        numpy.ndarray: Array of shape (n_timesteps, 1, height, width, 2) containing all timesteps
        list: Sorted list of timestep values
    """
    # Read all timestep data
    timestep_data = read_timestep_data(base_dir)
    
    # Sort timesteps
    sorted_timesteps = sorted(timestep_data.keys())
    
    # Initialize array to store all processed data
    all_data = np.zeros((len(sorted_timesteps), 1, grid_size[0], grid_size[1], 2))
    
    # Process each timestep
    for i, timestep in enumerate(sorted_timesteps):
        coords, vels = timestep_data[timestep]
        dataset = process_data(coords, vels)
        grid_data = prepare_for_autoencoder(dataset, grid_size)
        all_data[i] = grid_data
        
        print(f"Processed timestep {timestep} ({i+1}/{len(sorted_timesteps)})")
    
    print(f"\nFinal data shape: {all_data.shape}")
    print(f"Timesteps: {sorted_timesteps}")
    
    return all_data, sorted_timesteps



def timestep_data_generator(csv_filepath, chunk_size=10000):
    """
    Generator function that yields data for each timestep from a CSV file.
    
    Args:
        csv_filepath (str): Path to the CSV file
        chunk_size (int): Number of rows to read at a time
        
    Yields:
        tuple: (timestep, numpy.ndarray) where the array has shape (n_points, 4) containing (x,y,u,v)
    """
    # Define the columns we need
    required_cols = ['TimeStep', 'Points:0', 'Points:1', 'U:0', 'U:1']
    
    # Keep track of the current timestep being processed
    current_timestep = None
    current_data = []
    
    # Process the file in chunks
    for chunk in pd.read_csv(csv_filepath, chunksize=chunk_size, usecols=required_cols):
        # Remove quotes from column names if present
        chunk.columns = chunk.columns.str.strip('"')
        
        # Group the chunk by TimeStep
        for timestep, group in chunk.groupby('TimeStep'):
            # If we're starting a new timestep and have data from previous timestep
            if current_timestep is not None and current_timestep != timestep and current_data:
                # Convert accumulated data to numpy array and yield it
                data_array = np.array(current_data)
                yield current_timestep, data_array
                # Reset the current_data list
                current_data = []
            
            # Update current timestep
            current_timestep = timestep
            
            # Extract the required columns and append to current_data
            points_and_velocities = group[['Points:0', 'Points:1', 'U:0', 'U:1']].values
            current_data.extend(points_and_velocities)
    
    # Don't forget to yield the last timestep's data
    if current_data:
        data_array = np.array(current_data)
        yield current_timestep, data_array

# Example usage:
csv_file = './data/csv/timestamp_data.csv'

csv_full_file = './data/csv/timestamp_data_full.csv'

# Process CSV file in chunks to avoid memory issues
def filter_csv_by_z_coord(filepath: str):
    chunk_size = 10000  # Adjust based on available memory
    output_file = filepath.replace('.csv', '_filtered.csv')
    first_chunk = True

    try:
        # Process the file in chunks
        for chunk in pd.read_csv(filepath, chunksize=chunk_size):
            # Filter rows in this chunk
            filtered_chunk = chunk[chunk['Points:2'] != 1]
            
            if first_chunk:
                # Write header for first chunk
                filtered_chunk.to_csv(output_file, index=False, mode='w')
                first_chunk = False
            else:
                # Append without header for subsequent chunks
                filtered_chunk.to_csv(output_file, index=False, mode='a', header=False)
        
        # Replace original file with filtered version
        os.replace(output_file, filepath)
        print(f"Successfully filtered data and saved to {filepath}")
            
    except Exception as e:
        print(f"Error processing CSV: {str(e)}")
        if os.path.exists(output_file):
            os.remove(output_file)  # Clean up temporary file if there was an error




# filter_csv_by_z_coord(csv_full_file)

# Create figure and axis once
# fig, ax = plt.subplots(figsize=(10, 6))
# plt.ion()  # Turn on interactive mode

# # Initialize empty plot
# magnitude = np.zeros((256, 80))
# img = ax.imshow(magnitude, aspect=0.3125, cmap='viridis')
# colorbar = plt.colorbar(img, label='Velocity Magnitude')
# ax.set_xlabel('Y coordinate')
# ax.set_ylabel('X coordinate')

# # Process each timestep's data
# for timestep, data in timestep_data_generator(csv_file):
#     print(f"Processing timestep {timestep}, data shape: {data.shape}")
#     processed_data = prepare_for_autoencoder(data)
    
#     # Update the plot data
#     magnitude = np.sqrt(processed_data[0, :, :, 0]**2 + processed_data[0, :, :, 1]**2)
#     img.set_array(magnitude)
#     img.set_clim(vmin=magnitude.min(), vmax=magnitude.max())
#     ax.set_title(f'Velocity Field at Timestep {timestep}')
    
#     # Update the display
#     plt.draw()
#     plt.pause(0.1)  # Adjust this value to control animation speed

# plt.ioff()  # Turn off interactive mode
# plt.show()

