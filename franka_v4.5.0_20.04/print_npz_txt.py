import numpy as np
import argparse
from pathlib import Path


def print_npz_contents(npz_path, output_path=None):
    """
    Read a .npz file and print all contents (complete, no truncation)
    
    Args:
        npz_path: Path to the .npz file
        output_path: Path to save the text file (optional)
    """
    # Set numpy print options to show everything
    np.set_printoptions(threshold=np.inf, linewidth=150, suppress=True)
    
    # Load the npz file
    data = np.load(npz_path)
    
    # Generate output
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append(f"NPZ File: {npz_path}")
    output_lines.append("=" * 80)
    output_lines.append("")
    
    # Print each array
    for key in data.files:
        array = data[key]
        
        output_lines.append(f"Array: '{key}'")
        output_lines.append(f"Shape: {array.shape}, Dtype: {array.dtype}")
        output_lines.append(f"Total elements: {array.size}")
        output_lines.append("")
        output_lines.append(str(array))
        output_lines.append("")
        output_lines.append("-" * 80)
        output_lines.append("")
    
    # Join all lines
    output_text = "\n".join(output_lines)
    
    # Print to console
    print(output_text)
    
    # Save to file
    if output_path is None:
        npz_file = Path(npz_path)
        output_path = npz_file.parent / f"{npz_file.stem}_contents.txt"
    
    with open(output_path, 'w') as f:
        f.write(output_text)
    
    print(f"\nContents saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Print all contents of .npz file")
    parser.add_argument("npz_file", help="Path to .npz file")
    parser.add_argument("-o", "--output", help="Output text file path (optional)")
    
    args = parser.parse_args()
    
    path = Path(args.npz_file)
    
    if not path.exists():
        print(f"Error: File not found: {args.npz_file}")
        return
    
    if not path.suffix == ".npz":
        print(f"Error: Not a .npz file: {args.npz_file}")
        return
    
    print_npz_contents(args.npz_file, args.output)


if __name__ == "__main__":
    main()