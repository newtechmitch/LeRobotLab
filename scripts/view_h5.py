#!/usr/bin/env python3
"""
Script to view and explore HDF5 (.h5) files.
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


def print_h5_structure(file_path: str, max_items: int = 10, show_data: bool = False):
    """
    Print the structure and contents of an HDF5 file.
    
    Args:
        file_path: Path to the HDF5 file
        max_items: Maximum number of items to show for arrays
        show_data: Whether to show actual data values
    """
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"HDF5 File: {file_path}")
            print("=" * 60)
            _print_group(f, "", max_items, show_data)
    except Exception as e:
        print(f"Error reading HDF5 file: {e}")
        sys.exit(1)


def _print_group(group, prefix="", max_items=10, show_data=False, level=0):
    """Recursively print HDF5 group structure"""
    indent = "  " * level
    
    # Print group attributes
    if group.attrs:
        print(f"{indent}{prefix}Attributes:")
        for attr_name, attr_value in group.attrs.items():
            print(f"{indent}  {attr_name}: {attr_value}")
        print()
    
    # Print datasets and subgroups
    for key in group.keys():
        item = group[key]
        item_path = f"{prefix}/{key}" if prefix else key
        
        if isinstance(item, h5py.Group):
            print(f"{indent}📁 Group: {key}")
            _print_group(item, item_path, max_items, show_data, level + 1)
        
        elif isinstance(item, h5py.Dataset):
            print(f"{indent}📄 Dataset: {key}")
            print(f"{indent}  Shape: {item.shape}")
            print(f"{indent}  Dtype: {item.dtype}")
            
            # Print dataset attributes
            if item.attrs:
                print(f"{indent}  Attributes:")
                for attr_name, attr_value in item.attrs.items():
                    print(f"{indent}    {attr_name}: {attr_value}")
            
            # Show data preview
            if show_data and item.size > 0:
                data = item[...]
                if item.size <= max_items:
                    print(f"{indent}  Data: {data}")
                else:
                    if len(item.shape) == 1:
                        print(f"{indent}  Data preview: {data[:max_items]}...")
                    elif len(item.shape) == 2:
                        print(f"{indent}  Data preview (first {max_items} rows):")
                        for i, row in enumerate(data[:max_items]):
                            print(f"{indent}    [{i}]: {row}")
                        if len(data) > max_items:
                            print(f"{indent}    ... ({len(data) - max_items} more rows)")
                    else:
                        print(f"{indent}  Data preview: {data.flat[:max_items]}...")
            
            # Show statistics for numerical data
            if show_data and np.issubdtype(item.dtype, np.number) and item.size > 0:
                data = item[...]
                print(f"{indent}  Statistics:")
                print(f"{indent}    Min: {np.min(data):.6f}")
                print(f"{indent}    Max: {np.max(data):.6f}")
                print(f"{indent}    Mean: {np.mean(data):.6f}")
                print(f"{indent}    Std: {np.std(data):.6f}")
            
            print()


def explore_dataset(file_path: str, dataset_path: str):
    """Explore a specific dataset within the HDF5 file"""
    try:
        with h5py.File(file_path, 'r') as f:
            if dataset_path not in f:
                print(f"Dataset '{dataset_path}' not found in file")
                print("Available datasets:")
                _list_datasets(f)
                return
            
            dataset = f[dataset_path]
            if not isinstance(dataset, h5py.Dataset):
                print(f"'{dataset_path}' is not a dataset")
                return
            
            print(f"Dataset: {dataset_path}")
            print("=" * 40)
            print(f"Shape: {dataset.shape}")
            print(f"Dtype: {dataset.dtype}")
            print(f"Size: {dataset.size}")
            
            if dataset.attrs:
                print("Attributes:")
                for attr_name, attr_value in dataset.attrs.items():
                    print(f"  {attr_name}: {attr_value}")
            
            # Show data
            data = dataset[...]
            print(f"Data shape: {data.shape}")
            
            if len(data.shape) == 1:
                print(f"Data (first 20): {data[:20]}")
                if len(data) > 20:
                    print(f"... and {len(data) - 20} more values")
            elif len(data.shape) == 2:
                print("Data (first 10 rows, all columns):")
                for i, row in enumerate(data[:10]):
                    print(f"  [{i:2d}]: {row}")
                if len(data) > 10:
                    print(f"  ... and {len(data) - 10} more rows")
            else:
                print(f"Data preview: {data.flat[:50]}")
            
            if np.issubdtype(dataset.dtype, np.number):
                print(f"Statistics:")
                print(f"  Min: {np.min(data):.6f}")
                print(f"  Max: {np.max(data):.6f}")
                print(f"  Mean: {np.mean(data):.6f}")
                print(f"  Std: {np.std(data):.6f}")
                
    except Exception as e:
        print(f"Error exploring dataset: {e}")


def _list_datasets(group, prefix=""):
    """List all datasets in the HDF5 file"""
    for key in group.keys():
        item = group[key]
        item_path = f"{prefix}/{key}" if prefix else key
        
        if isinstance(item, h5py.Dataset):
            print(f"  {item_path} {item.shape} {item.dtype}")
        elif isinstance(item, h5py.Group):
            _list_datasets(item, item_path)


def main():
    parser = argparse.ArgumentParser(description="View and explore HDF5 files")
    parser.add_argument("file_path", help="Path to the HDF5 file")
    parser.add_argument("-d", "--dataset", help="Specific dataset path to explore")
    parser.add_argument("-s", "--show-data", action="store_true", 
                       help="Show actual data values")
    parser.add_argument("-n", "--max-items", type=int, default=10,
                       help="Maximum number of items to show (default: 10)")
    
    args = parser.parse_args()
    
    file_path = Path(args.file_path)
    if not file_path.exists():
        print(f"File not found: {file_path}")
        sys.exit(1)
    
    if args.dataset:
        explore_dataset(str(file_path), args.dataset)
    else:
        print_h5_structure(str(file_path), args.max_items, args.show_data)


if __name__ == "__main__":
    main()