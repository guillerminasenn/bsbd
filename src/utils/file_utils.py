"""Functions for reading MCMC sampler objects and analyzing their size.
Functions:
- `read`: Reads a saved MCMC sampler object from a file and returns the sampler, chunk files, and estimations.
- `get_size_detailed`: Recursively calculates the size of an object and its components.
- `print_object_size_breakdown`: Prints a detailed breakdown of an object's size by component.
"""

# Standard library imports
import sys
import os
import pickle

# Third-party library imports
import numpy as np

def read_stats(path, folder, model_filename):
    """Read the stats file from the MCMC run.
    
    Params:
    -------
    path: str
        Root path to the file where the folder with the MCMC results is saved.
    folder: str
        Folder where the MCMC results are saved.
    model_filename: str
        Name of the model file to read.

    Returns:
    -------
    stats: dict
        Dictionary containing the stats from the MCMC run.
    """

    # Load the sampler object file
    absolute_filename_stats = os.path.join(path, folder, model_filename + '_stats.pkl')
    with open(absolute_filename_stats, 'rb') as file:
        stats = pickle.load(file)
    print(f'Stats dict has been read from {absolute_filename_stats}.')
    return stats

def read_sampler_object(path, folder, model_filename):
    """Read a saved MCMC sampler object from a file.
    
    Params:
    -------
    path: str
        Root path to the file where the folder with the MCMC results is saved.
    folder: str
        Folder where the MCMC results are saved.
    model_filename: str
        Name of the model file to read.

    Returns:
    -------
    sampler: dict
        A dictionary containing the MCMC sampler object.
    """

    # Load the sampler dict file
    absolute_filename_sampler = os.path.join(path, folder, model_filename + '.pkl')
    with open(absolute_filename_sampler, 'rb') as file:
        sampler = pickle.load(file)
    print(f'Sampler dict has been read from {absolute_filename_sampler}.')
    return sampler

def read_samples(path, folder, model_filename, chunk_size=5000, thinning=1, max_iterations=None):
    """Read a saved MCMC sampler object from a file.
    Params:
    -------
    path: str
        Root path to the file where the folder with the MCMC results is saved.
    folder: str
        Folder where the MCMC results are saved.
    model_filename: str
        Name of the model file to read.
    thinning: int
        Thinning factor for the samples. If > 1, will read every `thinning`-th sample.

    Returns:
    -------
    estimations: dict
        Dictionary with the estimations from all the chunk files.   
    chunk_files: list
        List of chunk files read from the samples folder.
    """

    # Load the estimations from all the chunk files
    absolute_folder_samples = os.path.join(path, folder, 'samples')
    
    # Old
    # chunk_files = sorted([f for f in os.listdir(absolute_folder_samples)])
    
    # New
    chunk_files = [f for f in os.listdir(absolute_folder_samples) if f.endswith('.pkl')]
    def extract_chunk_index(filename):
        """Sort files based on the numeric chunk index"""
        try:
            return int(filename.split('_chunk_')[1].split('.')[0])
        except (IndexError, ValueError):
            return -1  # Return -1 for files that don't match the pattern
    chunk_files = sorted(chunk_files, key=extract_chunk_index) 
    
    # Calculate total samples after thinning using chunk indices from filenames
    chunk_indices = []
    for chunk_file in chunk_files:
        # Extract chunk index from filename (format is typically "*_chunk_X.pkl")
        try:
            chunk_index = int(chunk_file.split('_chunk_')[1].split('.')[0])
            chunk_indices.append(chunk_index)
            # print(chunk_index)
        except (IndexError, ValueError):
            print(f"Warning: Could not extract chunk index from {chunk_file}")
            continue
    total_chunks = max(chunk_indices) + 1 # The total chunks is the max index + 1 (since indices are 0-based)

    # Calculate how many chunks to read
    if max_iterations is not None:
        total_chunks = int(np.ceil(max_iterations / chunk_size))
        chunk_files = chunk_files[:total_chunks]

    # Calculate samples before thinning
    if thinning > chunk_size:
        raise ValueError(f"Thinning factor {thinning} cannot be greater than chunk size {chunk_size}.")
    chunk_size_thinned = chunk_size // thinning + int(chunk_size % thinning > 0)
    # total_samples = total_chunks * chunk_size
    total_samples_thinned = total_chunks * chunk_size_thinned
    print(f"Chunk size: {chunk_size}, Total chunks: {total_chunks}, thinning: {thinning}, chunk size thinned: {chunk_size_thinned}, Total samples thinned: {total_samples_thinned}")

    # Determine dimensions of the estimation dictionary values
    with open(os.path.join(absolute_folder_samples, chunk_files[0]), 'rb') as file:
        first_chunk = pickle.load(file)
    
    # Pre-allocate arrays
    estimations = {}
    for key, value in first_chunk.items():
        row_dim = value.shape[0]
        estimations[key] = np.zeros((row_dim, total_samples_thinned))
    # print(f"Pre-allocated estimations dictionary with keys: {list(estimations.keys())} and {estimations[key].shape[1]} columns for each key.")
    
    # Fill pre-allocated arrays
    current_pos = 0
    for i, chunk_file in enumerate(chunk_files):
        absolute_filename_chunk = os.path.join(absolute_folder_samples, chunk_file)
        with open(absolute_filename_chunk, 'rb') as file:
            chunk = pickle.load(file)
        
        for key, value in chunk.items():
            if value.shape[1] > 1:
                thinned_value = value[:, ::thinning]
                end_pos = current_pos + thinned_value.shape[1]
                estimations[key][:, current_pos:end_pos] = thinned_value
                # print(f"Thinned {key} from shape {value.shape} to {thinned_value.shape}.")
                # print(f"end_pos: {end_pos}, current_pos: {current_pos}, total_samples_thinned: {total_samples_thinned}")
            else:
                estimations[key][:, current_pos] = value[:, 0]
        
        current_pos += chunk_size_thinned
        # print(f"Processed chunk {i+1}/{len(chunk_files)}: {chunk_file}")
    print(f"All chunks loaded.")
    return estimations, chunk_files

# def read_samples(path, folder, model_filename, chunk_size=5000, thinning=1):
#     """Read a saved MCMC sampler object from a file.
#     Params:
#     -------
#     path: str
#         Root path to the file where the folder with the MCMC results is saved.
#     folder: str
#         Folder where the MCMC results are saved.
#     model_filename: str
#         Name of the model file to read.
#     thinning: int
#         Thinning factor for the samples. If > 1, will read every `thinning`-th sample.

#     Returns:
#     -------
#     estimations: dict
#         Dictionary with the estimations from all the chunk files.   
#     chunk_files: list
#         List of chunk files read from the samples folder.
#     """

#     # Load the estimations from all the chunk files
#     absolute_folder_samples = os.path.join(path, folder, 'samples')
    
#     # Old
#     # chunk_files = sorted([f for f in os.listdir(absolute_folder_samples)])
    
#     # New
#     chunk_files = [f for f in os.listdir(absolute_folder_samples) if f.endswith('.pkl')]
#     def extract_chunk_index(filename):
#         """Sort files based on the numeric chunk index"""
#         try:
#             return int(filename.split('_chunk_')[1].split('.')[0])
#         except (IndexError, ValueError):
#             return -1  # Return -1 for files that don't match the pattern
#     chunk_files = sorted(chunk_files, key=extract_chunk_index) 
    
#     # Calculate total samples after thinning using chunk indices from filenames
#     chunk_indices = []
#     for chunk_file in chunk_files:
#         # Extract chunk index from filename (format is typically "*_chunk_X.pkl")
#         try:
#             chunk_index = int(chunk_file.split('_chunk_')[1].split('.')[0])
#             chunk_indices.append(chunk_index)
#             print(chunk_index)
#         except (IndexError, ValueError):
#             print(f"Warning: Could not extract chunk index from {chunk_file}")
#             continue
#     total_chunks = max(chunk_indices) + 1 # The total chunks is the max index + 1 (since indices are 0-based)

#     # Calculate samples before thinning
#     if thinning > chunk_size:
#         raise ValueError(f"Thinning factor {thinning} cannot be greater than chunk size {chunk_size}.")
#     chunk_size_thinned = chunk_size // thinning + int(chunk_size % thinning > 0)
#     # total_samples = total_chunks * chunk_size
#     total_samples_thinned = total_chunks * chunk_size_thinned
#     print(f"Chunk size: {chunk_size}, Total chunks: {total_chunks}, thinning: {thinning}, chunk size thinned: {chunk_size_thinned}, Total samples thinned: {total_samples_thinned}")

#     # Determine dimensions of the estimation dictionary values
#     with open(os.path.join(absolute_folder_samples, chunk_files[0]), 'rb') as file:
#         first_chunk = pickle.load(file)
    
#     # Pre-allocate arrays
#     estimations = {}
#     for key, value in first_chunk.items():
#         row_dim = value.shape[0]
#         estimations[key] = np.zeros((row_dim, total_samples_thinned))
#     print(f"Pre-allocated estimations dictionary with keys: {list(estimations.keys())} and {estimations[key].shape[1]} columns for each key.")
    
#     # Fill pre-allocated arrays
#     current_pos = 0
#     for i, chunk_file in enumerate(chunk_files):
#         absolute_filename_chunk = os.path.join(absolute_folder_samples, chunk_file)
#         with open(absolute_filename_chunk, 'rb') as file:
#             chunk = pickle.load(file)
        
#         for key, value in chunk.items():
#             if value.shape[1] > 1:
#                 thinned_value = value[:, ::thinning]
#                 end_pos = current_pos + thinned_value.shape[1]
#                 estimations[key][:, current_pos:end_pos] = thinned_value
#                 # print(f"Thinned {key} from shape {value.shape} to {thinned_value.shape}.")
#                 # print(f"end_pos: {end_pos}, current_pos: {current_pos}, total_samples_thinned: {total_samples_thinned}")
#             else:
#                 estimations[key][:, current_pos] = value[:, 0]
        
#         current_pos += chunk_size_thinned
#         print(f"Processed chunk {i+1}/{len(chunk_files)}: {chunk_file}")

#     return estimations, chunk_files

def get_size_detailed(obj, name="object", level=0, max_level=2, seen=None):
    """Recursively find the size of an object and its components in bytes"""
    if seen is None:
        seen = {}
    
    obj_id = id(obj)
    if obj_id in seen:
        return 0, []
    
    # Mark object as seen with its path
    seen[obj_id] = name
    
    size = sys.getsizeof(obj)
    components = []
    
    # Only dig deeper if we haven't reached max level
    if level < max_level:
        # Handle dictionaries
        if isinstance(obj, dict):
            for k, v in obj.items():
                k_name = f"{k}"
                v_name = f"{name}.{k_name}"
                v_size, v_components = get_size_detailed(v, v_name, level + 1, max_level, seen)
                k_size, k_components = get_size_detailed(k, f"{name}.key({k_name})", level + 1, max_level, seen)
                size += v_size + k_size
                components.extend(v_components)
                components.extend(k_components)
                components.append((v_name, v_size))
        
        # Handle objects with __dict__
        elif hasattr(obj, '__dict__'):
            for k, v in obj.__dict__.items():
                v_name = f"{name}.{k}"
                v_size, v_components = get_size_detailed(v, v_name, level + 1, max_level, seen)
                size += v_size
                components.extend(v_components)
                components.append((v_name, v_size))
        
        # Handle lists, tuples, sets
        elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray, np.ndarray)):
            for i, v in enumerate(obj):
                v_name = f"{name}[{i}]"
                v_size, v_components = get_size_detailed(v, v_name, level + 1, max_level, seen)
                size += v_size
                components.extend(v_components)
                if i < 5:  # Only show details for first few elements
                    components.append((v_name, v_size))
            if len(obj) > 5:
                components.append((f"{name}[...] ({len(obj)} items total)", 0))
    
    # Special handling for numpy arrays
    if isinstance(obj, np.ndarray):
        size += obj.nbytes
    
    return size, components

def print_object_size_breakdown(obj, name="object", max_level=2):
    """Print a breakdown of object size by component"""
    total_size, components = get_size_detailed(obj, name, max_level=max_level)
    total_mb = total_size / (1024 * 1024)
    
    print(f"Total size of {name}: {total_mb:.2f} MB")
    print("\nComponent breakdown:")
    print("-" * 80)
    print(f"{'Component':<50} {'Size (MB)':<15} {'% of Total':<10}")
    print("-" * 80)
    
    # Sort by size, largest first
    components.sort(key=lambda x: x[1], reverse=True)
    
    # Print top components (by size)
    top_n = 20  # Show top 20 components
    for i, (component_name, size) in enumerate(components[:top_n]):
        size_mb = size / (1024 * 1024)
        percentage = 100 * size / total_size if total_size > 0 else 0
        if size_mb > 0.01:  # Only show components larger than 0.01 MB
            print(f"{component_name:<50} {size_mb:<15.4f} {percentage:<10.2f}%")
    
    if len(components) > top_n:
        remaining = len(components) - top_n
        print(f"\n... and {remaining} more components not shown")

# # Example usage
# print_object_size_breakdown(model, "model", max_level=2)