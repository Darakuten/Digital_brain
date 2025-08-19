#!/usr/bin/env python3
"""
Script to analyze ECoG channel data structure from .mat files
"""

import scipy.io
import numpy as np
import os

def analyze_ecog_channel():
    # Focus on Session1/ECoG_ch1.mat first
    ecog_file = '/Users/mz/Downloads/pvrnn_sa-master/Digital Brain/Data/20120814MD_Anesthesia+and+Sleep_George_Toru+Yanagawa_mat_ECoG128/Session1/ECoG_ch1.mat'
    
    print("=== ECoG Channel 1 Data Analysis ===\n")
    
    if not os.path.exists(ecog_file):
        print(f"File not found: {ecog_file}")
        return
        
    try:
        # Load the .mat file
        print(f"Loading: {ecog_file}")
        mat_data = scipy.io.loadmat(ecog_file)
        
        # Get all non-private variables
        variables = [k for k in mat_data.keys() if not k.startswith('__')]
        print(f"Variables found: {variables}\n")
        
        for var_name in variables:
            data = mat_data[var_name]
            print(f"=== Variable: {var_name} ===")
            print(f"Type: {type(data)}")
            print(f"Data type: {data.dtype if hasattr(data, 'dtype') else 'N/A'}")
            print(f"Shape: {data.shape if hasattr(data, 'shape') else 'N/A'}")
            
            if hasattr(data, 'shape'):
                # Calculate size in memory
                if hasattr(data, 'nbytes'):
                    size_mb = data.nbytes / (1024 * 1024)
                    print(f"Memory size: {size_mb:.2f} MB")
                
                # For 1D data (time series)
                if len(data.shape) == 1:
                    n_samples = data.shape[0]
                    print(f"Number of samples: {n_samples}")
                    
                    # Estimate sampling rate based on duration
                    # From condition data, we know session spans ~3383 seconds
                    estimated_duration = 3383.3  # seconds from condition analysis
                    estimated_fs = n_samples / estimated_duration
                    print(f"Estimated sampling rate: {estimated_fs:.1f} Hz")
                    print(f"Duration: {estimated_duration:.1f} seconds ({estimated_duration/60:.1f} minutes)")
                
                # For 2D data
                elif len(data.shape) == 2:
                    rows, cols = data.shape
                    print(f"Dimensions: {rows} × {cols}")
                    if rows == 1 or cols == 1:
                        n_samples = max(rows, cols)
                        print(f"Number of samples: {n_samples}")
                        estimated_duration = 3383.3
                        estimated_fs = n_samples / estimated_duration
                        print(f"Estimated sampling rate: {estimated_fs:.1f} Hz")
                
                # Statistical analysis for numerical data
                if np.issubdtype(data.dtype, np.number):
                    flat_data = data.flatten()
                    print(f"Min value: {np.min(flat_data):.6f}")
                    print(f"Max value: {np.max(flat_data):.6f}")
                    print(f"Mean: {np.mean(flat_data):.6f}")
                    print(f"Std: {np.std(flat_data):.6f}")
                    
                    # Show first and last few samples
                    print(f"First 10 samples: {flat_data[:10]}")
                    print(f"Last 10 samples: {flat_data[-10:]}")
                    
                    # Check for any obvious patterns
                    if len(flat_data) > 1000:
                        # Check if data looks like it's in microvolts (typical ECoG range)
                        if np.abs(np.mean(flat_data)) < 1000 and np.std(flat_data) < 1000:
                            print("→ Data appears to be in microvolts (μV) - typical for ECoG")
                        elif np.abs(np.mean(flat_data)) < 1 and np.std(flat_data) < 1:
                            print("→ Data appears to be in volts (V)")
                        else:
                            print("→ Data units unclear - may need scaling")
            
            print("-" * 50)
            
    except Exception as e:
        print(f"Error analyzing file: {e}")
        import traceback
        traceback.print_exc()

def analyze_multiple_channels():
    """Analyze multiple channels to check consistency"""
    base_path = '/Users/mz/Downloads/pvrnn_sa-master/Digital Brain/Data/20120814MD_Anesthesia+and+Sleep_George_Toru+Yanagawa_mat_ECoG128/Session1/'
    
    print("\n=== Multi-Channel Consistency Check ===")
    
    # Check first 5 channels
    for ch in range(1, 6):
        file_path = os.path.join(base_path, f'ECoG_ch{ch}.mat')
        if os.path.exists(file_path):
            try:
                mat_data = scipy.io.loadmat(file_path)
                variables = [k for k in mat_data.keys() if not k.startswith('__')]
                
                for var_name in variables:
                    data = mat_data[var_name]
                    if hasattr(data, 'shape'):
                        print(f"Channel {ch}: {var_name} shape = {data.shape}")
                        
            except Exception as e:
                print(f"Error with channel {ch}: {e}")

if __name__ == "__main__":
    analyze_ecog_channel()
    analyze_multiple_channels()