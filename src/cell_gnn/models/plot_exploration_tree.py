"""
Plot UCB Exploration Tree from ucb_scores.txt

Visualizes the exploration tree structure with:
- Circles for regular nodes, crosses for leaf nodes
- Node ID, visits, and RMSE displayed

Usage:
    python plot_exploration_tree.py <ucb_scores.txt> [--output <output.png>]
"""

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


@dataclass
class UCBNode:
    """Represents a node from ucb_scores.txt."""
    id: int
    ucb: float
    parent: Optional[int]
    visits: int
    rmse: float
    g_phi_r2: float = 0.0
    external_input_r2: float = -1.0  # -1 means not available
    cluster_accuracy: float = -1.0  # -1 means not available
    training_time_min: float = -1.0  # -1 means not available
    mutation: str = ""


def parse_ucb_scores(filepath: str) -> list[UCBNode]:
    """Parse ucb_scores.txt into a list of UCBNode objects."""
    nodes = []

    with open(filepath, 'r') as f:
        content = f.read()

    # Pattern: Node N: UCB=X.XXX, parent=P|root, visits=V, RMSE=X.XXX, g_phi_R2=X.XXX, ...
    # All fields after RMSE are optional for backward compatibility
    pattern = r'Node (\d+): UCB=([\d.]+), parent=(\d+|root), visits=(\d+), RMSE=([\d.]+)(?:, g_phi_R2=([\d.]+))?(?:, External_input_R2=([\d.]+))?(?:, Cluster=([\d.]+))?(?:, Time=([\d.]+))?(?:, Mutation=([^\n\[]+))?'

    for match in re.finditer(pattern, content):
        node_id = int(match.group(1))
        ucb = float(match.group(2))
        parent_str = match.group(3)
        parent = None if parent_str == 'root' else int(parent_str)
        visits = int(match.group(4))
        rmse = float(match.group(5))
        g_phi_r2 = float(match.group(6)) if match.group(6) else 0.0
        external_input_r2 = float(match.group(7)) if match.group(7) else -1.0
        cluster_accuracy = float(match.group(8)) if match.group(8) else -1.0
        training_time_min = float(match.group(9)) if match.group(9) else -1.0
        mutation = match.group(10).strip() if match.group(10) else ""

        nodes.append(UCBNode(
            id=node_id,
            ucb=ucb,
            parent=parent,
            visits=visits,
            rmse=rmse,
            g_phi_r2=g_phi_r2,
            external_input_r2=external_input_r2,
            cluster_accuracy=cluster_accuracy,
            training_time_min=training_time_min,
            mutation=mutation
        ))

    return nodes


def build_tree(nodes: list[UCBNode]) -> dict:
    """Build tree structure: children dict and find roots."""
    children = defaultdict(list)
    node_map = {n.id: n for n in nodes}

    for node in nodes:
        if node.parent is not None:
            children[node.parent].append(node.id)

    # Sort children by id for consistent layout
    for parent_id in children:
        children[parent_id].sort()

    # Find root nodes (nodes with no parent or parent not in node_map)
    roots = [n.id for n in nodes if n.parent is None or n.parent not in node_map]

    return {
        'children': children,
        'node_map': node_map,
        'roots': roots
    }


def compute_layout(tree: dict) -> dict[int, tuple[float, float]]:
    """
    Compute x,y positions for tree visualization.
    x = depth from root
    y = vertical spread within depth level
    """
    children = tree['children']
    roots = tree['roots']

    depth_map = {}
    y_positions = {}

    # Compute depths using BFS
    def compute_depth(node_id, current_depth=0):
        depth_map[node_id] = current_depth
        for child_id in children.get(node_id, []):
            compute_depth(child_id, current_depth + 1)

    for root in roots:
        compute_depth(root, 0)

    # Assign y positions: leaves get sequential positions, parents center on children
    leaf_counter = [0]

    def assign_y_dfs(node_id):
        child_list = children.get(node_id, [])
        if not child_list:
            # Leaf node
            y_positions[node_id] = leaf_counter[0]
            leaf_counter[0] += 1
        else:
            # Process children first
            for child_id in child_list:
                assign_y_dfs(child_id)
            # Parent y = center of children
            y_positions[node_id] = np.mean([y_positions[c] for c in child_list])

    for root in roots:
        assign_y_dfs(root)

    # Combine into positions dict
    positions = {}
    for node_id in depth_map:
        if node_id in y_positions:
            positions[node_id] = (depth_map[node_id], y_positions[node_id])

    return positions


def plot_ucb_tree(nodes: list[UCBNode],
                  output_path: Optional[str] = None,
                  title: str = "UCB Exploration Tree",
                  simulation_info: Optional[str] = None):
    """
    Plot the UCB exploration tree.

    - Circle (o) for nodes with children
    - Cross (x) for leaf nodes
    - Shows node ID, visits, and RMSE
    - Shows simulation parameters near root node

    Note: For RMSE, lower is better, so color coding is reversed
    compared to R2: low RMSE = green, high RMSE = red.
    """
    if not nodes:
        print("No nodes to plot")
        return

    tree = build_tree(nodes)
    positions = compute_layout(tree)
    children = tree['children']
    node_map = tree['node_map']

    # Color based on RMSE value (lower is better: green=low, red=high)
    def get_color(rmse):
        if rmse <= 0.1:
            return '#2ecc71'  # green - excellent
        elif rmse <= 0.5:
            return '#f39c12'  # orange - moderate
        else:
            return '#e74c3c'  # red - poor

    fig, ax = plt.subplots(figsize=(16, 12))

    # Set white background explicitly
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Draw edges first (so they're behind nodes)
    for node in nodes:
        if node.parent is not None and node.parent in positions and node.id in positions:
            x1, y1 = positions[node.parent]
            x2, y2 = positions[node.id]
            ax.plot([x1, x2], [y1, y2], color='#34495e', linestyle='-',
                   linewidth=1.5, alpha=0.6, zorder=1)

    # Compute UCB range for size scaling
    ucb_values = [n.ucb for n in nodes]
    min_ucb = min(ucb_values)
    max_ucb = max(ucb_values)
    ucb_range = max_ucb - min_ucb if max_ucb > min_ucb else 1.0

    # Draw nodes
    for node in nodes:
        if node.id not in positions:
            continue

        x, y = positions[node.id]
        color = get_color(node.rmse)

        # Size proportional to UCB (larger base size to fit node numbers)
        size = 400 + 200 * (node.ucb - min_ucb) / ucb_range

        # Determine if leaf node (no children)
        is_leaf = len(children.get(node.id, [])) == 0

        if is_leaf:
            # Cross marker for leaf nodes
            ax.scatter(x, y, c=color, s=size, marker='x', linewidths=3, zorder=2)
        else:
            # Circle marker for internal nodes
            ax.scatter(x, y, c=color, s=size, marker='o',
                      edgecolors='black', linewidths=0.5, zorder=2)

        # Label: node id inside/near the marker (always black)
        ax.annotate(str(node.id), (x, y), ha='center', va='center',
                   fontsize=9, color='black', zorder=3)

        # Mutation above the node (for nodes with id > 1)
        if node.id > 1 and node.mutation:
            # Remove parenthesis part from mutation text (e.g., "(2x increase exploring lr_W)")
            mutation_text = re.sub(r'\s*\([^)]*\)\s*$', '', node.mutation).strip()
            # Skip simulation change messages (they clutter the plot)
            if not mutation_text.startswith('simulation changed'):
                ax.annotate(mutation_text, (x, y), ha='left', va='bottom',
                           fontsize=6, xytext=(5, 14), textcoords='offset points',
                           color='#333333', zorder=3, rotation=45)

        # Annotation: UCB/V and RMSE/g_phi_R2 below the node
        label_text = f"UCB={node.ucb:.2f} V={node.visits}\nRMSE={node.rmse:.3f} g_phi_R\u00b2={node.g_phi_r2:.2f}"
        # Add external input R2 if available (>= 0)
        if node.external_input_r2 >= 0:
            label_text += f"\nR\u00b2_ext={node.external_input_r2:.2f}"
        # Add cluster accuracy (Acc) and training time (T) on same line if available
        acc_t_parts = []
        if node.cluster_accuracy >= 0:
            acc_t_parts.append(f"Acc={node.cluster_accuracy:.2f}")
        if node.training_time_min >= 0:
            acc_t_parts.append(f"T={node.training_time_min:.1f}")
        if acc_t_parts:
            label_text += f"\n{', '.join(acc_t_parts)}"
        ax.annotate(label_text, (x, y), ha='center', va='top',
                   fontsize=8, xytext=(0, -14), textcoords='offset points',
                   color='#555555', zorder=3)

    # Add simulation info at top left of figure
    if simulation_info:
        # Format: remove "Simulation: " prefix, replace underscores with spaces
        sim_text = simulation_info.replace('Simulation:', '').strip()
        sim_text = sim_text.replace('_', ' ')
        # Split by comma and join with newlines
        sim_lines = [p.strip() for p in sim_text.split(',')]
        # Filter lines: prioritize key simulation parameters
        filtered_lines = []
        # First pass: n_neurons and n_frames
        for line in sim_lines:
            if 'n neurons' in line or 'n frames' in line:
                filtered_lines.append(line)
        # Second pass: time step and recurrent info
        for line in sim_lines:
            if 'time step' in line or 'recurrent' in line:
                filtered_lines.append(line)
        # Third pass: connectivity and dynamics parameters
        for line in sim_lines:
            if 'connectivity type' in line:
                filtered_lines.append(line)
            elif 'Dale law=' in line:
                filtered_lines.append(line)
            elif 'E/I=' in line:
                filtered_lines.append(line)
            elif 'noise model level' in line:
                filtered_lines.append(line)
            elif 'connectivity rank' in line and 'low rank' in sim_text.lower():
                filtered_lines.append(line)
            elif 'connectivity filling' in line or 'filling factor' in line:
                filtered_lines.append(line)
            elif 'n neuron types' in line or 'n types' in line:
                filtered_lines.append(line)
            elif 'g=' in line or 'gain' in line.lower():
                filtered_lines.append(line)
            elif 'low rank factorization' in line:
                filtered_lines.append(line)
            elif 'low rank=' in line:
                filtered_lines.append(line)
        # Format with two parameters per line for compactness
        if filtered_lines:
            paired_lines = []
            for i in range(0, len(filtered_lines), 2):
                if i + 1 < len(filtered_lines):
                    paired_lines.append(f"{filtered_lines[i]}, {filtered_lines[i+1]}")
                else:
                    paired_lines.append(filtered_lines[i])
            sim_formatted = '\n'.join(paired_lines)
        else:
            sim_formatted = '\n'.join(sim_lines)
        # Place at top left using axes coordinates (0,1 = top left)
        ax.text(0.02, 0.98, sim_formatted, transform=ax.transAxes,
                fontsize=11, ha='left', va='top', color='#333333')

    # Remove axis labels and ticks
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])

    # Set axis limits with padding
    if positions:
        x_vals = [p[0] for p in positions.values()]
        y_vals = [p[1] for p in positions.values()]
        ax.set_xlim(min(x_vals) - 0.5, max(x_vals) + 0.5)
        ax.set_ylim(min(y_vals) - 1, max(y_vals) + 1)

    ax.grid(False)
    ax.axis('off')

    # Legend (RMSE: lower is better)
    legend_elements = [
        mpatches.Patch(color='#2ecc71', label='RMSE \u2264 0.1'),
        mpatches.Patch(color='#f39c12', label='RMSE \u2264 0.5'),
        mpatches.Patch(color='#e74c3c', label='RMSE > 0.5'),
        plt.Line2D([0], [0], marker='o', color='gray', label='Internal node',
                   markerfacecolor='gray', markersize=8, linestyle='None'),
        plt.Line2D([0], [0], marker='x', color='gray', label='Leaf node',
                   markerfacecolor='gray', markersize=8, linestyle='None', markeredgewidth=2),
    ]
    legend = ax.legend(handles=legend_elements, loc='upper right', fontsize=9,
                       facecolor='white', edgecolor='black', framealpha=1.0)
    # Ensure legend text is black
    for text in legend.get_texts():
        text.set_color('black')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def print_summary(nodes: list[UCBNode]):
    """Print summary statistics."""
    if not nodes:
        print("No nodes found")
        return

    print("\n=== UCB Tree Summary ===")
    print(f"total nodes: {len(nodes)}")
    print(f"UCB range: {min(n.ucb for n in nodes):.3f} - {max(n.ucb for n in nodes):.3f}")
    print(f"Visits range: {min(n.visits for n in nodes)} - {max(n.visits for n in nodes)}")
    print(f"RMSE range: {min(n.rmse for n in nodes):.4f} - {max(n.rmse for n in nodes):.4f}")

    # Find highest UCB nodes (most promising to explore)
    sorted_by_ucb = sorted(nodes, key=lambda n: n.ucb, reverse=True)[:5]
    print("\nTop 5 by UCB (most promising):")
    for n in sorted_by_ucb:
        print(f"  Node {n.id}: UCB={n.ucb:.3f}, visits={n.visits}, RMSE={n.rmse:.4f}")


def main():
    # Default path to ucb_scores.txt (relative to this script's location)
    script_dir = Path(__file__).parent
    default_input = script_dir / "ucb_scores.txt"
    default_output = script_dir / "ucb_scores_tree.png"

    parser = argparse.ArgumentParser(
        description='Visualize UCB exploration tree from ucb_scores.txt'
    )
    parser.add_argument('input', type=str, nargs='?', default=str(default_input),
                       help=f'Path to ucb_scores.txt file (default: {default_input})')
    parser.add_argument('--output', '-o', type=str, default=str(default_output),
                       help=f'Output path for visualization (default: {default_output})')

    args = parser.parse_args()

    # Parse UCB scores
    print(f"Parsing {args.input}...")
    nodes = parse_ucb_scores(args.input)
    print(f"Found {len(nodes)} nodes")

    if not nodes:
        print("No nodes found in the file")
        return

    # Print summary
    print_summary(nodes)

    # Plot
    print("\nGenerating UCB tree visualization...")
    plot_ucb_tree(nodes, args.output,
                  title=f"UCB Exploration Tree ({len(nodes)} nodes)")

    print("\nDone!")


if __name__ == '__main__':
    main()
