#!/usr/bin/env python3
"""
Generate a tree graph visualization for a folder structure.

Usage:
    python generate_tree.py <folder_path> [output_file] [--max-depth N] [--format png/svg/pdf]

Examples:
    python generate_tree.py W:\fMOST\251637\cube_data_251637_INS_20260320
    python generate_tree.py /path/to/folder output_tree.png --max-depth 3
    python generate_tree.py . tree.svg --format svg
"""

import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict


def get_tree_data(folder_path, max_depth=None, current_depth=0, ignore_patterns=None):
    """
    Recursively collect tree structure data.
    
    Returns a nested dictionary representing the folder structure.
    """
    if ignore_patterns is None:
        ignore_patterns = {'.git', '__pycache__', '.pytest_cache', 'node_modules', '.venv', 'venv'}
    
    if max_depth is not None and current_depth > max_depth:
        return None
    
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        return None
    
    result = {
        'name': folder_path.name,
        'path': str(folder_path),
        'is_dir': folder_path.is_dir(),
        'size': 0,
        'children': []
    }
    
    if folder_path.is_dir():
        try:
            items = list(folder_path.iterdir())
            # Sort: directories first, then files, both alphabetically
            items.sort(key=lambda x: (not x.is_dir(), x.name.lower()))
            
            total_size = 0
            for item in items:
                # Skip hidden files and ignored patterns
                if item.name.startswith('.') or item.name in ignore_patterns:
                    continue
                    
                child_data = get_tree_data(item, max_depth, current_depth + 1, ignore_patterns)
                if child_data:
                    result['children'].append(child_data)
                    total_size += child_data.get('size', 0)
            
            # Calculate directory size
            try:
                result['size'] = total_size
            except:
                result['size'] = 0
                
        except PermissionError:
            result['name'] += " [Permission Denied]"
    else:
        try:
            result['size'] = folder_path.stat().st_size
        except:
            result['size'] = 0
    
    return result


def format_size(size_bytes):
    """Format byte size to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def count_items(tree_data):
    """Count total files and directories in tree."""
    if not tree_data:
        return 0, 0
    
    files = 0 if tree_data['is_dir'] else 1
    dirs = 1 if tree_data['is_dir'] else 0
    
    for child in tree_data.get('children', []):
        f, d = count_items(child)
        files += f
        dirs += d
    
    return files, dirs


def generate_ascii_tree(tree_data, prefix="", is_last=True, show_size=False):
    """Generate ASCII representation of the tree."""
    if not tree_data:
        return ""
    
    lines = []
    
    # Current item
    connector = "└── " if is_last else "├── "
    name = tree_data['name']
    
    if show_size and tree_data.get('size', 0) > 0:
        size_str = format_size(tree_data['size'])
        name = f"{name} ({size_str})"
    
    lines.append(prefix + connector + name)
    
    # Children
    children = tree_data.get('children', [])
    for i, child in enumerate(children):
        is_last_child = (i == len(children) - 1)
        extension = "    " if is_last else "│   "
        child_lines = generate_ascii_tree(child, prefix + extension, is_last_child, show_size)
        lines.append(child_lines)
    
    return "\n".join(lines)


def find_graphviz_bin():
    """Try to find Graphviz bin directory on Windows."""
    possible_paths = [
        r"C:\Program Files\Graphviz\bin",
        r"C:\Program Files (x86)\Graphviz\bin",
        r"C:\Tools\Graphviz\bin",
        os.path.expanduser(r"~\Graphviz\bin"),
        os.path.expanduser(r"~\AppData\Local\Graphviz\bin"),
    ]
    
    # Also check if it's already in PATH
    if os.system("where dot >nul 2>&1") == 0:
        return None  # Already in PATH
    
    for path in possible_paths:
        if os.path.exists(os.path.join(path, "dot.exe")):
            return path
    
    return None


def generate_graphviz_tree(tree_data, output_file, format_type='png'):
    """Generate a graphical tree using graphviz."""
    try:
        from graphviz import Digraph
    except ImportError:
        print("Error: graphviz package not installed.")
        print("Install it with: pip install graphviz")
        return False, "graphviz_not_installed"
    
    # Try to find Graphviz on Windows
    graphviz_bin = find_graphviz_bin()
    if graphviz_bin:
        os.environ['PATH'] = graphviz_bin + os.pathsep + os.environ.get('PATH', '')
        print(f"Found Graphviz at: {graphviz_bin}")
    
    dot = Digraph(comment='Folder Tree')
    dot.attr(rankdir='TB', size='20,20', dpi='150')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Arial')
    
    node_counter = [0]
    
    def add_nodes(data, parent_id=None):
        if not data:
            return
        
        node_id = f"node_{node_counter[0]}"
        node_counter[0] += 1
        
        # Node styling based on type
        if data['is_dir']:
            label = f"{data['name']}"
            color = '#4A90D9'  # Blue for directories
            fontcolor = 'white'
        else:
            label = f"{data['name']}"
            color = '#90EE90'  # Light green for files
            fontcolor = 'black'
        
        # Truncate long names
        if len(label) > 50:
            label = label[:47] + "..."
        
        dot.node(node_id, label, fillcolor=color, fontcolor=fontcolor)
        
        if parent_id:
            dot.edge(parent_id, node_id)
        
        for child in data.get('children', []):
            add_nodes(child, node_id)
    
    add_nodes(tree_data)
    
    # Render
    output_path = str(Path(output_file).with_suffix(''))
    try:
        dot.render(output_path, format=format_type, cleanup=True)
        print(f"Tree graph saved to: {output_path}.{format_type}")
        return True, None
    except Exception as e:
        error_msg = str(e)
        if "failed to execute" in error_msg.lower() or "dot" in error_msg.lower():
            return False, "dot_not_found"
        print(f"Error rendering graph: {e}")
        return False, "other"


def generate_matplotlib_tree(tree_data, output_file, max_display_nodes=200):
    """Generate a tree visualization using matplotlib with size limits."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("Error: matplotlib not installed.")
        print("Install it with: pip install matplotlib")
        return False
    
    # First, count total nodes
    def count_nodes(data):
        if not data:
            return 0
        count = 1
        for child in data.get('children', []):
            count += count_nodes(child)
        return count
    
    total_nodes = count_nodes(tree_data)
    print(f"Total nodes to display: {total_nodes}")
    
    if total_nodes > max_display_nodes:
        print(f"Warning: Too many nodes ({total_nodes}). Using compact representation.")
        return generate_compact_matplotlib_tree(tree_data, output_file, max_display_nodes)
    
    # Calculate tree layout
    def calculate_positions(data, x=0, y=0, level=0, positions=None, parent=None):
        if positions is None:
            positions = {}
        
        if not data:
            return positions, x
        
        children = data.get('children', [])
        
        if not children:
            # Leaf node
            positions[data['path']] = {
                'x': x,
                'y': y,
                'data': data,
                'parent': parent
            }
            return positions, x + 1
        
        # Parent node - position at average of children
        child_x_positions = []
        current_x = x
        
        for child in children:
            positions, current_x = calculate_positions(child, current_x, y - 1, level + 1, positions, data['path'])
            child_x_positions.append(positions[child['path']]['x'])
        
        parent_x = sum(child_x_positions) / len(child_x_positions)
        
        positions[data['path']] = {
            'x': parent_x,
            'y': y,
            'data': data,
            'parent': parent
        }
        
        return positions, current_x
    
    positions, max_x = calculate_positions(tree_data)
    
    if not positions:
        print("No data to visualize")
        return False
    
    # Calculate bounds
    min_y = min(p['y'] for p in positions.values())
    max_y = max(p['y'] for p in positions.values())
    min_x = min(p['x'] for p in positions.values())
    max_x_pos = max(p['x'] for p in positions.values())
    
    # Limit figure size to prevent memory issues
    width = min(65535, max(16, (max_x_pos - min_x + 1) * 2))
    height = min(65535, max(12, (max_y - min_y + 1) * 1.5))
    
    # Cap at reasonable display size
    display_width = min(40, width)
    display_height = min(30, height)
    
    fig, ax = plt.subplots(figsize=(display_width, display_height))
    
    # Scale factor for coordinates
    x_scale = display_width / width if width > 0 else 1
    y_scale = display_height / height if height > 0 else 1
    
    # Draw connections
    for path, pos in positions.items():
        if pos['parent']:
            parent_pos = positions.get(pos['parent'])
            if parent_pos:
                ax.plot([parent_pos['x'] * x_scale, pos['x'] * x_scale], 
                       [parent_pos['y'] * y_scale, pos['y'] * y_scale], 
                       'k-', linewidth=0.5, alpha=0.5, zorder=1)
    
    # Draw nodes
    for path, pos in positions.items():
        data = pos['data']
        
        # Node styling
        if data['is_dir']:
            color = '#4A90D9'
            text_color = 'white'
            box_width = 2.5 * x_scale
            box_height = 0.6 * y_scale
        else:
            color = '#90EE90'
            text_color = 'black'
            box_width = 2.0 * x_scale
            box_height = 0.5 * y_scale
        
        rect = mpatches.FancyBboxPatch(
            (pos['x'] * x_scale - box_width/2, pos['y'] * y_scale - box_height/2),
            box_width, box_height,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor='black',
            linewidth=1,
            zorder=2
        )
        ax.add_patch(rect)
        
        # Add text
        name = data['name']
        if len(name) > 25:
            name = name[:22] + "..."
        
        ax.text(pos['x'] * x_scale, pos['y'] * y_scale, name, ha='center', va='center', 
                fontsize=max(4, min(8, 1000 // total_nodes)), 
                fontweight='bold' if data['is_dir'] else 'normal',
                color=text_color, zorder=3)
    
    # Set limits and remove axes
    margin = 2
    ax.set_xlim(min_x * x_scale - margin, max_x_pos * x_scale + margin)
    ax.set_ylim(min_y * y_scale - margin, max_y * y_scale + margin)
    ax.axis('off')
    
    # Add title
    ax.text((min_x + max_x_pos) * x_scale / 2, max_y * y_scale + margin/2, 
            f"Directory Tree: {tree_data['name']}", 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Tree graph saved to: {output_file}")
    plt.close()
    return True


def generate_compact_matplotlib_tree(tree_data, output_file, max_nodes=200):
    """Generate a compact tree for large directories - show only first N nodes per level."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        return False
    
    # Limit children at each level
    def limit_tree(data, max_children=10, depth=0):
        if not data:
            return None
        
        result = {
            'name': data['name'],
            'path': data['path'],
            'is_dir': data['is_dir'],
            'size': data.get('size', 0),
            'children': []
        }
        
        children = data.get('children', [])
        
        # Sort: directories first
        children.sort(key=lambda x: (not x['is_dir'], x['name'].lower()))
        
        # Limit children
        if len(children) > max_children:
            kept = children[:max_children-1]
            # Add a placeholder for the rest
            rest_count = len(children) - max_children + 1
            placeholder = {
                'name': f'... and {rest_count} more items',
                'path': data['path'] + '/...',
                'is_dir': False,
                'size': 0,
                'children': []
            }
            kept.append(placeholder)
            children = kept
        
        for child in children:
            limited_child = limit_tree(child, max_children, depth + 1)
            if limited_child:
                result['children'].append(limited_child)
        
        return result
    
    limited_tree = limit_tree(tree_data, max_children=15)
    
    # Now use the regular matplotlib function with the limited tree
    # But we need to avoid recursion - let's do a simpler version
    
    # Calculate positions with limited tree
    def calculate_positions(data, x=0, y=0, positions=None, parent=None):
        if positions is None:
            positions = {}
        
        if not data:
            return positions, x
        
        children = data.get('children', [])
        
        if not children:
            positions[data['path']] = {
                'x': x,
                'y': y,
                'data': data,
                'parent': parent
            }
            return positions, x + 1
        
        child_x_positions = []
        current_x = x
        
        for child in children:
            positions, current_x = calculate_positions(child, current_x, y - 1, positions, data['path'])
            child_x_positions.append(positions[child['path']]['x'])
        
        parent_x = sum(child_x_positions) / len(child_x_positions)
        
        positions[data['path']] = {
            'x': parent_x,
            'y': y,
            'data': data,
            'parent': parent
        }
        
        return positions, current_x
    
    positions, max_x = calculate_positions(limited_tree)
    
    if not positions:
        return False
    
    # Use reasonable figure size
    node_count = len(positions)
    fig_width = min(32, max(12, node_count * 0.5))
    fig_height = min(24, max(8, len(set(p['y'] for p in positions.values())) * 1.5))
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    min_y = min(p['y'] for p in positions.values())
    max_y = max(p['y'] for p in positions.values())
    min_x = min(p['x'] for p in positions.values())
    max_x_pos = max(p['x'] for p in positions.values())
    
    # Draw connections
    for path, pos in positions.items():
        if pos['parent']:
            parent_pos = positions.get(pos['parent'])
            if parent_pos:
                ax.plot([parent_pos['x'], pos['x']], [parent_pos['y'], pos['y']], 
                       'k-', linewidth=0.5, alpha=0.5, zorder=1)
    
    # Draw nodes
    for path, pos in positions.items():
        data = pos['data']
        
        is_placeholder = data['name'].startswith('...')
        
        if data['is_dir']:
            color = '#4A90D9'
            text_color = 'white'
        elif is_placeholder:
            color = '#FFA500'  # Orange for placeholder
            text_color = 'black'
        else:
            color = '#90EE90'
            text_color = 'black'
        
        box_width = 2.5 if data['is_dir'] else 2.0
        box_height = 0.6
        
        rect = mpatches.FancyBboxPatch(
            (pos['x'] - box_width/2, pos['y'] - box_height/2),
            box_width, box_height,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor='black',
            linewidth=1,
            zorder=2
        )
        ax.add_patch(rect)
        
        name = data['name']
        if len(name) > 30:
            name = name[:27] + "..."
        
        ax.text(pos['x'], pos['y'], name, ha='center', va='center', 
                fontsize=7, fontweight='bold' if data['is_dir'] else 'normal',
                color=text_color, zorder=3)
    
    margin = 2
    ax.set_xlim(min_x - margin, max_x_pos + margin)
    ax.set_ylim(min_y - margin, max_y + margin)
    ax.axis('off')
    
    # Add title with note about truncation
    title = f"Directory Tree: {tree_data['name']} (showing subset of {node_count} items)"
    ax.text((min_x + max_x_pos) / 2, max_y + margin/2, title,
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Compact tree graph saved to: {output_file}")
    plt.close()
    return True


def generate_html_tree(tree_data, output_file):
    """Generate an interactive HTML tree visualization."""
    files, dirs = count_items(tree_data)
    
    def build_html(data, level=0):
        if not data:
            return ""
        
        is_dir = data['is_dir']
        icon = "📁" if is_dir else "📄"
        name = data['name']
        
        children = data.get('children', [])
        
        if children:
            children_html = "\n".join(build_html(child, level + 1) for child in children)
            return f'''
            <li>
                <span class="folder" onclick="toggle(this)">{icon} {name}</span>
                <ul class="nested">
                    {children_html}
                </ul>
            </li>
            '''
        else:
            return f'<li><span class="file">{icon} {name}</span></li>'
    
    tree_html = build_html(tree_data)
    
    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Directory Tree - {tree_data['name']}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
        }}
        .info {{
            color: #666;
            margin-bottom: 20px;
        }}
        ul, #myUL {{
            list-style-type: none;
        }}
        #myUL {{
            margin: 0;
            padding: 0;
        }}
        .caret {{
            cursor: pointer;
            user-select: none;
        }}
        .caret::before {{
            content: "\\25B6";
            color: black;
            display: inline-block;
            margin-right: 6px;
        }}
        .caret-down::before {{
            transform: rotate(90deg);
        }}
        .nested {{
            display: block;
            padding-left: 20px;
        }}
        .active {{
            display: block;
        }}
        .folder {{
            cursor: pointer;
            padding: 2px 5px;
            border-radius: 3px;
            background-color: #4A90D9;
            color: white;
            display: inline-block;
            margin: 2px 0;
        }}
        .file {{
            padding: 2px 5px;
            color: #333;
        }}
        li {{
            margin: 3px 0;
        }}
    </style>
</head>
<body>
    <h1>📂 Directory Tree: {tree_data['name']}</h1>
    <p class="info">📁 {dirs} directories | 📄 {files} files</p>
    <ul id="myUL">
        {tree_html}
    </ul>
    
    <script>
        function toggle(element) {{
            var nested = element.nextElementSibling;
            if (nested) {{
                if (nested.style.display === "none") {{
                    nested.style.display = "block";
                    element.style.backgroundColor = "#4A90D9";
                }} else {{
                    nested.style.display = "none";
                    element.style.backgroundColor = "#357ABD";
                }}
            }}
        }}
    </script>
</body>
</html>
'''
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Interactive HTML tree saved to: {output_file}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Generate tree graph visualization for a folder structure.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_tree.py W:\\fMOST\\251637\\cube_data_251637_INS_20260320
  python generate_tree.py /path/to/folder output.png --max-depth 3
  python generate_tree.py . tree.txt --ascii --show-size
  python generate_tree.py . tree.svg --format svg
  python generate_tree.py . tree.html  (interactive HTML tree)
        """
    )
    
    parser.add_argument('folder', help='Path to the folder to visualize')
    parser.add_argument('output', nargs='?', default=None, 
                       help='Output file path (default: tree.txt for ASCII, tree.png for graph)')
    parser.add_argument('--max-depth', '-d', type=int, default=None,
                       help='Maximum depth to traverse (default: unlimited)')
    parser.add_argument('--format', '-f', choices=['png', 'svg', 'pdf', 'txt', 'html'], 
                       default=None, help='Output format')
    parser.add_argument('--ascii', '-a', action='store_true',
                       help='Generate ASCII text tree instead of graphical')
    parser.add_argument('--show-size', '-s', action='store_true',
                       help='Show file/directory sizes')
    parser.add_argument('--matplotlib', '-m', action='store_true',
                       help='Use matplotlib instead of graphviz for visualization')
    parser.add_argument('--max-nodes', '-n', type=int, default=200,
                       help='Maximum nodes to display in matplotlib (default: 200)')
    
    args = parser.parse_args()
    
    # Resolve folder path
    folder_path = Path(args.folder).resolve()
    
    if not folder_path.exists():
        print(f"Error: Folder does not exist: {folder_path}")
        sys.exit(1)
    
    if not folder_path.is_dir():
        print(f"Error: Path is not a directory: {folder_path}")
        sys.exit(1)
    
    print(f"Scanning folder: {folder_path}")
    
    # Collect tree data
    tree_data = get_tree_data(folder_path, max_depth=args.max_depth)
    
    if not tree_data:
        print("Error: Could not read folder structure")
        sys.exit(1)
    
    # Count items
    files, dirs = count_items(tree_data)
    print(f"Found {files} files and {dirs} directories")
    
    # Determine output format and file
    if args.ascii or (args.format == 'txt'):
        # ASCII output
        output_file = args.output or 'tree.txt'
        ascii_tree = generate_ascii_tree(tree_data, show_size=args.show_size)
        
        # Add header
        header = f"Directory Tree: {folder_path}\n"
        header += f"Files: {files}, Directories: {dirs}\n"
        header += "=" * 60 + "\n\n"
        
        full_output = header + ascii_tree
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_output)
        
        print(f"ASCII tree saved to: {output_file}")
        print("\nPreview:")
        print(full_output[:2000] + "..." if len(full_output) > 2000 else full_output)
        
    elif args.format == 'html' or (args.output and args.output.endswith('.html')):
        # HTML output
        output_file = args.output or 'tree.html'
        generate_html_tree(tree_data, output_file)
        
    else:
        # Graphical output
        if args.format:
            format_type = args.format
        elif args.output:
            format_type = Path(args.output).suffix.lstrip('.') or 'png'
        else:
            format_type = 'png'
        
        output_file = args.output or f'tree.{format_type}'
        
        if args.matplotlib:
            success = generate_matplotlib_tree(tree_data, output_file, max_display_nodes=args.max_nodes)
            if not success:
                print("\nFalling back to ASCII tree...")
                output_file = str(Path(output_file).with_suffix('.txt'))
                ascii_tree = generate_ascii_tree(tree_data, show_size=args.show_size)
                
                header = f"Directory Tree: {folder_path}\n"
                header += f"Files: {files}, Directories: {dirs}\n"
                header += "=" * 60 + "\n\n"
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(header + ascii_tree)
                
                print(f"ASCII tree saved to: {output_file}")
        else:
            success, error_type = generate_graphviz_tree(tree_data, output_file, format_type)
            
            if not success:
                if error_type == "dot_not_found":
                    print("\n" + "="*60)
                    print("Graphviz executable not found!")
                    print("="*60)
                    print("\nTo fix this, you have several options:")
                    print("\n1. Install Graphviz (recommended):")
                    print("   - Download from: https://graphviz.org/download/")
                    print("   - Install and add to PATH")
                    print("\n2. Use matplotlib instead (no extra system install needed):")
                    print(f"   python generate_tree.py \"{args.folder}\" {output_file} --matplotlib")
                    print("\n3. Use ASCII output:")
                    print(f"   python generate_tree.py \"{args.folder}\" tree.txt --ascii")
                    print("\n4. Use interactive HTML output:")
                    print(f"   python generate_tree.py \"{args.folder}\" tree.html")
                    print("\n" + "="*60)
                    
                    # Auto-fallback to matplotlib
                    print("\nAuto-falling back to matplotlib...")
                    success = generate_matplotlib_tree(tree_data, output_file, max_display_nodes=args.max_nodes)
                    
                    if not success:
                        # Final fallback to ASCII
                        print("\nFalling back to ASCII tree...")
                        output_file = str(Path(output_file).with_suffix('.txt'))
                        ascii_tree = generate_ascii_tree(tree_data, show_size=args.show_size)
                        
                        header = f"Directory Tree: {folder_path}\n"
                        header += f"Files: {files}, Directories: {dirs}\n"
                        header += "=" * 60 + "\n\n"
                        
                        with open(output_file, 'w', encoding='utf-8') as f:
                            f.write(header + ascii_tree)
                        
                        print(f"ASCII tree saved to: {output_file}")
                else:
                    # Other error - fallback to ASCII
                    print("\nFalling back to ASCII tree...")
                    output_file = str(Path(output_file).with_suffix('.txt'))
                    ascii_tree = generate_ascii_tree(tree_data, show_size=args.show_size)
                    
                    header = f"Directory Tree: {folder_path}\n"
                    header += f"Files: {files}, Directories: {dirs}\n"
                    header += "=" * 60 + "\n\n"
                    
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(header + ascii_tree)
                    
                    print(f"ASCII tree saved to: {output_file}")


if __name__ == '__main__':
    main()
