"""
generate_folder_tree.py - Generate folder structure illustrations

Usage:
    python generate_folder_tree.py <path> [options]

Examples:
    # Basic usage
    python generate_folder_tree.py W:\fMOST\251637
    
    # Limit depth
    python generate_folder_tree.py W:\fMOST\251637 --depth 3
    
    # Save to file
    python generate_folder_tree.py W:\fMOST\251637 --output structure.txt
    
    # Show only directories
    python generate_folder_tree.py W:\fMOST\251637 --dirs-only
    
    # Include file count in directories
    python generate_folder_tree.py W:\fMOST\251637 --count
"""

import os
import sys
import argparse
from pathlib import Path


def count_files_and_dirs(path):
    """Count files and directories recursively."""
    file_count = 0
    dir_count = 0
    for root, dirs, files in os.walk(path):
        file_count += len(files)
        dir_count += len(dirs)
    return file_count, dir_count


def generate_tree(path, prefix="", max_depth=10, current_depth=0, 
                  dirs_only=False, show_count=False, max_files=10):
    """
    Generate ASCII tree structure for a directory.
    
    Args:
        path: Directory path
        prefix: Current line prefix (for recursion)
        max_depth: Maximum recursion depth
        current_depth: Current depth level
        dirs_only: If True, only show directories
        show_count: If True, show file/dir counts
        max_files: Maximum files to show per directory
    
    Returns:
        List of strings (lines of the tree)
    """
    lines = []
    
    if current_depth >= max_depth:
        return lines
    
    try:
        entries = sorted(os.listdir(path))
    except (PermissionError, FileNotFoundError) as e:
        lines.append(f"{prefix}[Error: {e}]")
        return lines
    
    # Separate and sort
    dirs = [e for e in entries if os.path.isdir(os.path.join(path, e))]
    files = [] if dirs_only else [e for e in entries if os.path.isfile(os.path.join(path, e))]
    
    all_entries = dirs + files
    
    for i, name in enumerate(all_entries):
        full_path = os.path.join(path, name)
        is_dir = os.path.isdir(full_path)
        is_last = (i == len(all_entries) - 1)
        
        # Choose connector
        connector = "└── " if is_last else "├── "
        
        # Build display name
        if is_dir:
            display_name = f"{name}/"
            if show_count:
                fc, dc = count_files_and_dirs(full_path)
                display_name += f" ({fc} files, {dc} dirs)"
        else:
            display_name = name
        
        lines.append(f"{prefix}{connector}{display_name}")
        
        # Recurse into directories
        if is_dir:
            extension = "    " if is_last else "│   "
            sub_lines = generate_tree(
                full_path,
                prefix + extension,
                max_depth,
                current_depth + 1,
                dirs_only,
                show_count,
                max_files
            )
            lines.extend(sub_lines)
        
        # Limit files shown
        if not is_dir and i - len(dirs) + 1 >= max_files and i < len(all_entries) - 1:
            remaining = len(files) - max_files
            lines.append(f"{prefix}{'    ' if is_last else '│   '}└── ... and {remaining} more files")
            break
    
    return lines


def generate_markdown_tree(path, max_depth=10, dirs_only=False):
    """Generate markdown-compatible tree structure."""
    lines = []
    
    def recurse(current_path, depth=0):
        if depth >= max_depth:
            return
        
        try:
            entries = sorted(os.listdir(current_path))
        except (PermissionError, FileNotFoundError):
            return
        
        dirs = [e for e in entries if os.path.isdir(os.path.join(current_path, e))]
        files = [] if dirs_only else [e for e in entries if os.path.isfile(os.path.join(current_path, e))]
        
        indent = "  " * depth
        
        for name in dirs:
            lines.append(f"{indent}- **{name}/**")
            recurse(os.path.join(current_path, name), depth + 1)
        
        for name in files:
            lines.append(f"{indent}- {name}")
    
    lines.append(f"- **{os.path.basename(path) or path}/**")
    recurse(path, 1)
    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Generate folder structure tree illustration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s W:\\fMOST\\251637
  %(prog)s /mnt/data --depth 3 --output tree.txt
  %(prog)s . --dirs-only --count
        """
    )
    
    parser.add_argument("path", help="Directory path to visualize")
    parser.add_argument("-d", "--depth", type=int, default=10, 
                       help="Maximum depth (default: 10)")
    parser.add_argument("-o", "--output", type=str, 
                       help="Output file (default: print to stdout)")
    parser.add_argument("--dirs-only", action="store_true",
                       help="Show only directories")
    parser.add_argument("--count", action="store_true",
                       help="Show file/directory counts")
    parser.add_argument("--max-files", type=int, default=10,
                       help="Max files to show per directory (default: 10)")
    parser.add_argument("--markdown", action="store_true",
                       help="Output in markdown format")
    
    args = parser.parse_args()
    
    # Validate path
    if not os.path.exists(args.path):
        print(f"Error: Path not found: {args.path}", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.isdir(args.path):
        print(f"Error: Not a directory: {args.path}", file=sys.stderr)
        sys.exit(1)
    
    # Generate tree
    if args.markdown:
        lines = generate_markdown_tree(args.path, args.depth, args.dirs_only)
    else:
        lines = [f"{args.path}/"]
        lines.extend(generate_tree(
            args.path, 
            max_depth=args.depth,
            dirs_only=args.dirs_only,
            show_count=args.count,
            max_files=args.max_files
        ))
    
    output = "\n".join(lines)
    
    # Output
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"Tree saved to: {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
