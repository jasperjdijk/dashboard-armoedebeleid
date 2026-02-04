"""
Convert Excel data to optimized Parquet format for faster loading.

This script reads the full Excel file and creates a streamlined Parquet file
containing only the sheet and columns needed by the dashboard.

Benefits:
- Faster load times (Parquet is 5-10x faster than Excel)
- Smaller file size
- Better Cloud Run cold start performance

Usage:
    python convert_to_parquet.py                 # GUI mode (default)
    python convert_to_parquet.py --no-gui        # Command line mode with default files
    python convert_to_parquet.py --input file.xlsx --output file.parquet  # Command line with custom paths
"""

import pandas as pd
import sys
import os
from pathlib import Path

# Columns needed by the dashboard
REQUIRED_COLUMNS = [
    # Municipality information
    'Gemeentenaam',
    'GMcode',
    'Inwoners',

    # Regulation information
    'ID',
    'N4',
    'FR',

    # Filter columns
    'WB',
    'BT',
    'CAV',

    # Household type 1 (Alleenstaande)
    'WRD_HH01',
    'IG_HH01',
    'Referteperiode_HH01',

    # Household type 2 (Alleenstaande ouder met kind)
    'WRD_HH02',
    'IG_HH02',
    'Referteperiode_HH02',

    # Household type 3 (Paar)
    'WRD_HH03',
    'IG_HH03',
    'Referteperiode_HH03',

    # Household type 4 (Paar met twee kinderen)
    'WRD_HH04',
    'IG_HH04',
    'Referteperiode_HH04',
]


def select_files_gui():
    """
    Open GUI dialogs to select input Excel file and output Parquet file.

    Returns:
    --------
    tuple: (excel_path, parquet_path) or (None, None) if cancelled
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        print("✗ Error: tkinter not available. Use --no-gui flag for command line mode.", file=sys.stderr)
        return None, None

    # Create and hide the root window
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    # Select input Excel file
    print("Opening file picker for Excel input file...")
    excel_path = filedialog.askopenfilename(
        title="Select Excel file to convert",
        filetypes=[
            ("Excel files", "*.xlsx *.xls"),
            ("All files", "*.*")
        ],
        initialfile="dataoverzicht_dashboard_armoedebeleid.xlsx"
    )

    if not excel_path:
        print("✗ No input file selected. Conversion cancelled.")
        root.destroy()
        return None, None

    print(f"Selected input: {excel_path}")

    # Suggest output filename based on input
    excel_filename = Path(excel_path).stem
    suggested_output = Path(excel_path).parent / f"{excel_filename}.parquet"

    # Select output Parquet file location
    print("Opening save dialog for Parquet output file...")
    parquet_path = filedialog.asksaveasfilename(
        title="Save Parquet file as",
        defaultextension=".parquet",
        filetypes=[
            ("Parquet files", "*.parquet"),
            ("All files", "*.*")
        ],
        initialfile=suggested_output.name,
        initialdir=suggested_output.parent
    )

    root.destroy()

    if not parquet_path:
        print("✗ No output location selected. Conversion cancelled.")
        return None, None

    print(f"Selected output: {parquet_path}")

    return excel_path, parquet_path


def convert_excel_to_parquet(excel_path, output_path):
    """
    Convert Excel file to optimized Parquet format.

    Parameters:
    -----------
    excel_path : str
        Path to input Excel file
    output_path : str
        Path for output Parquet file

    Returns:
    --------
    bool: True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Converting Excel to Parquet")
    print(f"{'='*60}")
    print(f"Input:  {excel_path}")
    print(f"Output: {output_path}")
    print(f"Sheet:  Totaaloverzicht")
    print(f"Columns to keep: {len(REQUIRED_COLUMNS)}")
    print(f"{'='*60}\n")

    try:
        # Read only the required sheet and columns
        print("Reading Excel file...")
        df = pd.read_excel(
            excel_path,
            sheet_name='Totaaloverzicht',
            usecols=REQUIRED_COLUMNS
        )

        print(f"✓ Loaded {len(df):,} rows")

        # Filter out empty municipality names (same as dashboard logic)
        df = df[df['Gemeentenaam'].notna() & (df['Gemeentenaam'] != '')]
        print(f"✓ After filtering empty municipalities: {len(df):,} rows")

        # Clean data types for Parquet compatibility
        print("Cleaning data types...")
        # Ensure FR column is string (it may contain mixed types in Excel)
        df['FR'] = df['FR'].astype(str)
        # Ensure GMcode is string
        df['GMcode'] = df['GMcode'].astype(str)
        # Ensure Gemeentenaam is string
        df['Gemeentenaam'] = df['Gemeentenaam'].astype(str)
        # Ensure N4 is string
        df['N4'] = df['N4'].astype(str)
        # Ensure ID is string
        df['ID'] = df['ID'].astype(str)

        # Save to Parquet with compression
        print("Saving to Parquet format...")
        df.to_parquet(
            output_path,
            engine='pyarrow',
            compression='snappy',
            index=False
        )

        # Calculate file sizes
        excel_size = os.path.getsize(excel_path) / (1024 * 1024)  # MB
        parquet_size = os.path.getsize(output_path) / (1024 * 1024)  # MB

        print(f"\n{'='*60}")
        print(f"✓ Conversion complete!")
        print(f"{'='*60}")
        print(f"  Excel file:   {excel_size:.2f} MB")
        print(f"  Parquet file: {parquet_size:.2f} MB")
        print(f"  Size reduction: {((excel_size - parquet_size) / excel_size * 100):.1f}%")
        print(f"\n  Output saved to:\n  {output_path}")
        print(f"{'='*60}")

        # Verify the data can be read back
        print("\nVerifying Parquet file...")
        test_df = pd.read_parquet(output_path)
        print(f"✓ Verification successful: {len(test_df):,} rows, {len(test_df.columns)} columns")
        print(f"\n{'='*60}")

        return True

    except FileNotFoundError:
        print(f"\n✗ Error: Excel file not found: {excel_path}", file=sys.stderr)
        return False
    except ValueError as e:
        if "Worksheet named" in str(e):
            print(f"\n✗ Error: Sheet 'Totaaloverzicht' not found in Excel file", file=sys.stderr)
        else:
            print(f"\n✗ Error reading Excel: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert Excel data to optimized Parquet format'
    )
    parser.add_argument(
        '--input',
        help='Input Excel file path (GUI mode if not specified)'
    )
    parser.add_argument(
        '--output',
        help='Output Parquet file path (GUI mode if not specified)'
    )
    parser.add_argument(
        '--no-gui',
        action='store_true',
        help='Use command line mode with default files instead of GUI'
    )

    args = parser.parse_args()

    # Determine file paths
    if args.input and args.output:
        # Command line mode with explicit paths
        excel_path = args.input
        parquet_path = args.output
    elif args.no_gui:
        # Command line mode with defaults
        excel_path = 'dataoverzicht_dashboard_armoedebeleid.xlsx'
        parquet_path = 'dataoverzicht_dashboard_armoedebeleid.parquet'
    else:
        # GUI mode (default)
        excel_path, parquet_path = select_files_gui()
        if not excel_path or not parquet_path:
            print("\nConversion cancelled by user.")
            sys.exit(1)

    # Perform conversion
    success = convert_excel_to_parquet(excel_path, parquet_path)

    if success:
        print("\n✓ All done! You can now use the Parquet file with your dashboard.")
        input("\nPress Enter to exit...")
    else:
        print("\n✗ Conversion failed. Please check the error messages above.")
        input("\nPress Enter to exit...")

    sys.exit(0 if success else 1)
