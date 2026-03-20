"""
Exploration Tree Visualization for Experiment Logs

This script parses experiment analysis markdown files and visualizes
the parameter exploration as a tree structure, showing how different
hyperparameter choices led to different outcomes.

Usage:
    python exploration_tree.py <analysis_file.md> [--output <output.png>]

Example:
    python exploration_tree.py analysis_experiment_convergence.md --output tree.png
"""

import math
import os
import re
import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np


@dataclass
class ExperimentNode:
    """Represents a single experiment iteration."""
    iteration: int
    status: str  # converged, partial, failed
    config: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    observation: str = ""
    change: str = ""
    changed_param: Optional[str] = None
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    parent: Optional[int] = None  # iteration number of parent
    children: list = field(default_factory=list)


def parse_experiment_log(filepath: str) -> list[ExperimentNode]:
    """Parse an experiment log markdown file into a list of nodes."""

    with open(filepath, 'r') as f:
        content = f.read()

    # Split by iteration headers
    iter_pattern = r'## Iter (\d+): (\w+)'
    config_pattern = r'Config: (.+)'
    metrics_pattern = r'Metrics: (.+)'
    observation_pattern = r'Observation: (.+)'
    change_pattern = r'Change: (.+)'

    nodes = []

    # Find all iterations
    sections = re.split(r'(?=## Iter \d+:)', content)

    for section in sections:
        if not section.strip():
            continue

        # Parse iteration header
        iter_match = re.search(iter_pattern, section)
        if not iter_match:
            continue

        iteration = int(iter_match.group(1))
        status = iter_match.group(2).lower()

        node = ExperimentNode(iteration=iteration, status=status)

        # Parse config
        config_match = re.search(config_pattern, section)
        if config_match:
            config_str = config_match.group(1)
            # Parse key=value pairs
            for pair in re.findall(r'(\w+)=([^,\s]+)', config_str):
                node.config[pair[0]] = pair[1]

        # Parse metrics
        metrics_match = re.search(metrics_pattern, section)
        if metrics_match:
            metrics_str = metrics_match.group(1)
            for pair in re.findall(r'(\w+)=([\d.eE+-]+)', metrics_str):
                try:
                    node.metrics[pair[0]] = float(pair[1])
                except ValueError:
                    node.metrics[pair[0]] = pair[1]

        # Parse observation
        obs_match = re.search(observation_pattern, section)
        if obs_match:
            node.observation = obs_match.group(1).strip()

        # Parse change
        change_match = re.search(change_pattern, section)
        if change_match:
            node.change = change_match.group(1).strip()
            # Try to parse the change into param, old, new
            param_change = re.search(r'(\w+):\s*([^\s]+)\s*->\s*([^\s\(]+)', node.change)
            if param_change:
                node.changed_param = param_change.group(1)
                node.old_value = param_change.group(2)
                node.new_value = param_change.group(3)

        nodes.append(node)

    return nodes


def build_tree_structure(nodes: list[ExperimentNode]) -> list[ExperimentNode]:
    """Build parent-child relationships based on parameter changes."""

    if not nodes:
        return nodes

    # First node has no parent
    nodes[0].parent = None

    for i, node in enumerate(nodes[1:], 1):
        prev_node = nodes[i-1]

        # Check if this is a validation run (same config, no change)
        if node.change.lower().startswith('none') or 'validation' in node.change.lower():
            # Same parent as previous or itself
            node.parent = prev_node.parent if prev_node.parent is not None else prev_node.iteration
        else:
            # New exploration branch - parent is previous node
            node.parent = prev_node.iteration

    # Build children lists
    node_map = {n.iteration: n for n in nodes}
    for node in nodes:
        if node.parent is not None and node.parent in node_map:
            parent = node_map[node.parent]
            if node.iteration not in parent.children:
                parent.children.append(node.iteration)

    return nodes


def compute_tree_layout(nodes: list[ExperimentNode]) -> dict[int, tuple[float, float]]:
    """Compute x,y positions for each node in the tree visualization."""

    if not nodes:
        return {}

    node_map = {n.iteration: n for n in nodes}
    positions = {}

    # Group nodes by "branches" - sequences of exploration
    # Track which parameter was changed to identify branch points

    # Simple layout: x = iteration, y based on status and rollout_RMSE_mean
    # For RMSE (lower is better), invert: y = 1 - RMSE so higher y = better
    for node in nodes:
        x = node.iteration

        # y based on rollout_RMSE_mean if available (inverted: lower RMSE = higher y)
        rmse = node.metrics.get('rollout_RMSE_mean', 0.5)
        y = max(0.0, 1.0 - rmse)

        positions[node.iteration] = (x, y)

    return positions


def plot_exploration_tree(nodes: list[ExperimentNode],
                          output_path: Optional[str] = None,
                          title: str = "Parameter Exploration Tree",
                          param_x: str = 'lr_W',
                          param_y: str = 'lr'):
    """Create visualization of the exploration tree with all panels in one figure."""

    if not nodes:
        print("No nodes to plot")
        return

    node_map = {n.iteration: n for n in nodes}

    # Color mapping for status
    status_colors = {
        'converged': '#2ecc71',  # green
        'partial': '#f39c12',    # orange
        'failed': '#e74c3c',     # red
        'oscillatory': '#3498db', # blue
        'steady': '#9b59b6',     # purple
        'good': '#2ecc71',       # green
        'moderate': '#f39c12',   # orange
        'poor': '#e74c3c',       # red
    }

    # Create figure with three subplots: 2 columns on top, 1 spanning bottom
    fig = plt.figure(figsize=(18, 14))

    # Top left: Exploration trajectory (rollout_RMSE_mean over iterations)
    ax1 = fig.add_subplot(2, 2, 1)
    # Top right: Parameter space exploration (2D)
    ax2 = fig.add_subplot(2, 2, 2)
    # Bottom: Parameter changes timeline (spans full width)
    ax3 = fig.add_subplot(2, 1, 2)

    # --- Top left: Exploration trajectory ---
    positions = compute_tree_layout(nodes)

    # Draw edges (connections between iterations)
    for node in nodes:
        if node.parent is not None and node.parent in positions:
            x1, y1 = positions[node.parent]
            x2, y2 = positions[node.iteration]

            # Color edge based on whether it's a validation run or exploration
            if 'none' in node.change.lower() or 'validation' in node.change.lower():
                edge_color = '#bdc3c7'  # gray for validation
                linestyle = '--'
            else:
                edge_color = '#34495e'  # dark for exploration
                linestyle = '-'

            ax1.plot([x1, x2], [y1, y2], color=edge_color,
                    linestyle=linestyle, linewidth=1.5, alpha=0.6, zorder=1)

    # Draw nodes
    for node in nodes:
        x, y = positions[node.iteration]
        color = status_colors.get(node.status, '#95a5a6')

        # Size based on validation status
        if 'validation' in node.observation.lower() or 'robust' in node.observation.lower():
            size = 200
            marker = 's'  # square for validation
        else:
            size = 100
            marker = 'o'

        ax1.scatter(x, y, c=color, s=size, marker=marker,
                   edgecolors='black', linewidths=0.5, zorder=2)

        # Add iteration number
        ax1.annotate(str(node.iteration), (x, y),
                    ha='center', va='center', fontsize=6, fontweight='bold')

    # Add horizontal lines for thresholds (inverted for RMSE: lower RMSE = better)
    ax1.axhline(y=0.9, color='green', linestyle=':', alpha=0.5, label='Good (RMSE<0.1)')
    ax1.axhline(y=0.1, color='red', linestyle=':', alpha=0.5, label='Poor (RMSE>0.9)')

    ax1.set_xlabel('Iteration', fontsize=11)
    ax1.set_ylabel('1 - rollout_RMSE_mean', fontsize=11)
    ax1.set_title('Convergence Trajectory', fontsize=12)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)

    # Add legend for status colors
    legend_patches = [mpatches.Patch(color=c, label=s.capitalize())
                     for s, c in status_colors.items() if s in [n.status for n in nodes]]
    ax1.legend(handles=legend_patches, loc='lower right', fontsize=8)

    # --- Top right: Parameter space exploration (2D) ---
    x_vals = []
    y_vals = []
    colors = []
    sizes = []
    labels = []

    for node in nodes:
        if param_x in node.config and param_y in node.config:
            try:
                x = float(node.config[param_x])
                y = float(node.config[param_y])
                x_vals.append(x)
                y_vals.append(y)
                colors.append(status_colors.get(node.status, '#95a5a6'))

                # Size based on rollout_RMSE_mean (lower RMSE = larger marker)
                rmse = node.metrics.get('rollout_RMSE_mean', 0.5)
                quality = max(0.0, 1.0 - rmse)
                sizes.append(50 + quality * 150)
                labels.append(node.iteration)
            except (ValueError, TypeError):
                continue

    if x_vals:
        # Draw exploration path
        for i in range(1, len(x_vals)):
            ax2.annotate('', xy=(x_vals[i], y_vals[i]),
                       xytext=(x_vals[i-1], y_vals[i-1]),
                       arrowprops=dict(arrowstyle='->', color='gray', alpha=0.3))

        # Scatter plot
        ax2.scatter(x_vals, y_vals, c=colors, s=sizes,
                   edgecolors='black', linewidths=0.5, alpha=0.8)

        # Add iteration labels
        for i, (x, y, label) in enumerate(zip(x_vals, y_vals, labels)):
            ax2.annotate(str(label), (x, y), ha='center', va='center',
                        fontsize=6, fontweight='bold')

        ax2.set_xlabel(param_x, fontsize=11)
        ax2.set_ylabel(param_y, fontsize=11)
        ax2.set_title(f'Parameter Space: {param_x} vs {param_y}', fontsize=12)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, f'Parameters {param_x} and {param_y}\nnot found in config',
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title(f'Parameter Space: {param_x} vs {param_y}', fontsize=12)

    # --- Bottom: Parameter changes timeline ---
    param_names = set()
    for node in nodes:
        param_names.update(node.config.keys())

    # Focus on key parameters
    key_params = ['lr_W', 'lr', 'coeff_W_L1', 'batch_size', 'factor', 'gain', 'n_types', 'Dale_law']
    key_params = [p for p in key_params if p in param_names]

    if key_params:
        # Create parameter change annotations
        y_positions = {p: i for i, p in enumerate(key_params)}

        for node in nodes:
            x = node.iteration

            # Mark changed parameter
            if node.changed_param and node.changed_param in y_positions:
                y = y_positions[node.changed_param]
                color = status_colors.get(node.status, '#95a5a6')
                ax3.scatter(x, y, c=color, s=80, marker='>', zorder=2)

                # Add value annotation
                if node.new_value:
                    ax3.annotate(node.new_value, (x, y + 0.15),
                                ha='center', va='bottom', fontsize=6, rotation=45)

        ax3.set_yticks(range(len(key_params)))
        ax3.set_yticklabels(key_params)
        ax3.set_xlabel('Iteration', fontsize=11)
        ax3.set_ylabel('Changed Parameter', fontsize=11)
        ax3.set_title('Parameter Changes Over Time', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, max(n.iteration for n in nodes) + 1)
        ax3.set_ylim(-0.5, len(key_params) - 0.5)

    # Add overall title
    fig.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved exploration tree to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_data_exploration(nodes: list[ExperimentNode],
                          output_path: Optional[str] = None,
                          title: str = "Data Exploration",
                          ucb_path: Optional[str] = None):
    """
    Visualize data exploration with 2 panels:

    Panels:
    - Left: svd_rank vs spectral_radius (colored by RMSE: red/orange/green)
    - Right: Parent-child UCB tree structure

    Args:
        nodes: List of ExperimentNode objects
        output_path: Path to save the plot
        title: Plot title
        ucb_path: Path to ucb_scores.txt file (optional)
    """
    if not nodes:
        print("No nodes to plot")
        return

    status_colors = {
        'converged': '#2ecc71',  # green
        'partial': '#f39c12',    # orange
        'failed': '#e74c3c',     # red
    }

    # marker based on connectivity_type
    connectivity_markers = {
        'chaotic': 'o',    # circle
        'low_rank': 's',   # square
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax1, ax2 = axes

    # extract metrics from nodes
    svd_ranks = []
    spectral_radii = []
    rmse_vals = []
    colors = []
    iterations = []
    conn_types = []

    for node in nodes:
        svd_rank = node.metrics.get('svd_rank', None)
        spectral_radius = node.metrics.get('spectral_radius', None)
        rmse = node.metrics.get('rollout_RMSE_mean', None)
        conn_type = node.config.get('connectivity_type', 'chaotic')

        # Debug: print node metrics
        print(f"[plot_data_exploration] Node {node.iteration}: svd_rank={svd_rank}, spectral_radius={spectral_radius}, status={node.status}")

        if svd_rank is not None and spectral_radius is not None:
            svd_ranks.append(float(svd_rank))
            spectral_radii.append(float(spectral_radius))
            rmse_vals.append(float(rmse) if rmse is not None else 1.0)
            colors.append(status_colors.get(node.status, '#95a5a6'))
            iterations.append(node.iteration)
            conn_types.append(conn_type)

    # --- Panel 1: svd_rank vs spectral_radius (colored by RMSE) ---
    # Debug: print what we found
    print(f"[plot_data_exploration] Found {len(svd_ranks)} nodes with svd_rank/spectral_radius data")
    if svd_ranks:
        for sr, spec, rmse, c, it, ct in zip(svd_ranks, spectral_radii, rmse_vals, colors, iterations, conn_types):
            marker = connectivity_markers.get(ct, 'o')
            ax1.scatter(sr, spec, c=c, s=120, marker=marker, edgecolors='black', linewidths=0.5, alpha=0.8, zorder=2)
            ax1.annotate(str(it), (sr, spec), ha='center', va='bottom', fontsize=7, fontweight='bold', xytext=(0, 5), textcoords='offset points')

        ax1.axhline(y=1.0, color='orange', linestyle=':', alpha=0.5, label='edge of chaos')
        ax1.axvline(x=10, color='blue', linestyle=':', alpha=0.5, label='min complexity')
        ax1.set_xlabel('svd_rank (activity complexity)', fontsize=11)
        ax1.set_ylabel('spectral_radius', fontsize=11)
        ax1.set_title('Activity Complexity vs Dynamics Stability', fontsize=12)
        ax1.grid(True, alpha=0.3)

        # legend for status colors
        legend_elements = []
        for s, c in status_colors.items():
            if s in [n.status for n in nodes]:
                legend_elements.append(mpatches.Patch(color=c, label=s.capitalize()))
        legend_elements.append(Line2D([0], [0], marker='o', color='gray', label='chaotic',
                                       markerfacecolor='gray', markersize=8, linestyle='None'))
        legend_elements.append(Line2D([0], [0], marker='s', color='gray', label='low_rank',
                                       markerfacecolor='gray', markersize=8, linestyle='None'))
        ax1.legend(handles=legend_elements, loc='upper right', fontsize=8)
    else:
        ax1.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Activity Complexity vs Dynamics Stability', fontsize=12)

    # --- Panel 2: UCB Tree Structure (parent-child relationships) ---
    # Build tree structure from parent field in analysis log
    node_map = {n.iteration: n for n in nodes}

    # Parse parent from Node line in analysis (stored during parse_experiment_log)
    # Build children dict from the parent relationships
    children = defaultdict(list)
    parent_map = {}

    for node in nodes:
        # Get parent from the node's parent field (set during build_tree_structure)
        parent_id = node.parent
        parent_map[node.iteration] = parent_id
        if parent_id is not None:
            children[parent_id].append(node.iteration)

    # Compute UCB for display
    def get_subtree_stats(node_id):
        if node_id not in node_map:
            return 0, 0.0
        visits = 1
        rmse = node_map[node_id].metrics.get('rollout_RMSE_mean', 1.0)
        rewards = max(0.0, 1.0 - rmse * 10.0)
        for child_id in children[node_id]:
            child_visits, child_rewards = get_subtree_stats(child_id)
            visits += child_visits
            rewards += child_rewards
        return visits, rewards

    n_total = len(nodes)
    c = 1.414

    ucb_values = {}
    for node in nodes:
        visits, sum_rewards = get_subtree_stats(node.iteration)
        mean_reward = sum_rewards / visits if visits > 0 else 0.0
        if visits > 0 and n_total > 1:
            exploration_term = c * math.sqrt(math.log(n_total) / visits)
        else:
            exploration_term = float('inf')
        ucb = mean_reward + exploration_term
        ucb_values[node.iteration] = {
            'ucb': ucb if ucb != float('inf') else 10.0,
            'visits': visits
        }

    # Compute tree layout: x = depth from root, y = spread within depth level
    depth_map = {}

    def compute_depth(node_id, current_depth=0):
        depth_map[node_id] = current_depth
        for child_id in children[node_id]:
            compute_depth(child_id, current_depth + 1)

    # Find root nodes
    root_nodes = [n.iteration for n in nodes if parent_map.get(n.iteration) is None]
    for root in root_nodes:
        compute_depth(root, 0)

    # Assign y positions: leaves get sequential positions, parents center on children
    y_positions = {}
    leaf_counter = [0]

    def assign_y_dfs(node_id):
        child_list = sorted(children.get(node_id, []))
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

    for root in root_nodes:
        assign_y_dfs(root)

    # Draw edges (parent -> child)
    for node in nodes:
        parent_id = parent_map.get(node.iteration)
        if parent_id is not None and parent_id in depth_map and node.iteration in depth_map:
            x1, y1 = depth_map[parent_id], y_positions[parent_id]
            x2, y2 = depth_map[node.iteration], y_positions[node.iteration]
            ax2.plot([x1, x2], [y1, y2], color='#34495e', linestyle='-', linewidth=2, alpha=0.7, zorder=1)

    # Draw nodes
    max_ucb = max(ucb_values[n.iteration]['ucb'] for n in nodes)
    min_ucb = min(ucb_values[n.iteration]['ucb'] for n in nodes)
    ucb_range = max_ucb - min_ucb if max_ucb > min_ucb else 1.0

    for node in nodes:
        if node.iteration not in depth_map:
            continue
        x = depth_map[node.iteration]
        y = y_positions[node.iteration]
        color = status_colors.get(node.status, '#95a5a6')
        ucb = ucb_values[node.iteration]['ucb']

        # Size proportional to UCB
        size = 100 + 150 * (ucb - min_ucb) / ucb_range

        conn_type = node.config.get('connectivity_type', 'chaotic')
        marker = connectivity_markers.get(conn_type, 'o')
        ax2.scatter(x, y, c=color, s=size, marker=marker, edgecolors='black', linewidths=1, zorder=2)

        # Label: node id on top, UCB below
        ax2.annotate(str(node.iteration), (x, y), ha='center', va='center',
                    fontsize=7, fontweight='bold', color='white')

    # Add UCB values as text next to nodes (closer to dot)
    for node in nodes:
        if node.iteration not in depth_map:
            continue
        x = depth_map[node.iteration]
        y = y_positions[node.iteration]
        ucb = ucb_values[node.iteration]['ucb']
        ax2.annotate(f'{ucb:.1f}', (x, y), ha='left', va='center', fontsize=6, xytext=(8, 0), textcoords='offset points')

    ax2.set_xlabel('Tree Depth', fontsize=11)
    ax2.set_ylabel('Branch', fontsize=11)
    ax2.set_title('Parent-Child Tree (size = UCB)', fontsize=12)
    ax2.set_xlim(-0.5, max(depth_map.values()) + 0.5 if depth_map else 1)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved data exploration plot to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_parameter_space(nodes: list[ExperimentNode],
                         param_x: str = 'lr_W',
                         param_y: str = 'lr',
                         output_path: Optional[str] = None):
    """Plot exploration in 2D parameter space."""

    if not nodes:
        return

    status_colors = {
        'converged': '#2ecc71',
        'partial': '#f39c12',
        'failed': '#e74c3c',
    }

    fig, ax = plt.subplots(figsize=(12, 10))

    # Extract parameter values
    x_vals = []
    y_vals = []
    colors = []
    sizes = []
    labels = []

    for node in nodes:
        if param_x in node.config and param_y in node.config:
            try:
                x = float(node.config[param_x])
                y = float(node.config[param_y])
                x_vals.append(x)
                y_vals.append(y)
                colors.append(status_colors.get(node.status, '#95a5a6'))

                # Size based on rollout_RMSE_mean (lower RMSE = larger marker)
                rmse = node.metrics.get('rollout_RMSE_mean', 0.5)
                quality = max(0.0, 1.0 - rmse)
                sizes.append(50 + quality * 150)
                labels.append(node.iteration)
            except (ValueError, TypeError):
                continue

    if not x_vals:
        print(f"No data points found for parameters {param_x} and {param_y}")
        return

    # Draw exploration path
    for i in range(1, len(x_vals)):
        ax.annotate('', xy=(x_vals[i], y_vals[i]),
                   xytext=(x_vals[i-1], y_vals[i-1]),
                   arrowprops=dict(arrowstyle='->', color='gray', alpha=0.3))

    # Scatter plot
    scatter = ax.scatter(x_vals, y_vals, c=colors, s=sizes,
                        edgecolors='black', linewidths=0.5, alpha=0.8)

    # Add iteration labels
    for i, (x, y, label) in enumerate(zip(x_vals, y_vals, labels)):
        ax.annotate(str(label), (x, y), ha='center', va='center',
                   fontsize=7, fontweight='bold')

    ax.set_xlabel(param_x, fontsize=12)
    ax.set_ylabel(param_y, fontsize=12)
    ax.set_title(f'Parameter Space Exploration: {param_x} vs {param_y}', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # Legend
    legend_patches = [mpatches.Patch(color=c, label=s.capitalize())
                     for s, c in status_colors.items()]
    ax.legend(handles=legend_patches, loc='best')

    plt.tight_layout()

    if output_path:
        # Modify output path for parameter space plot
        base = Path(output_path).stem
        suffix = Path(output_path).suffix
        param_output = str(Path(output_path).parent / f"{base}_param_space{suffix}")
        plt.savefig(param_output, dpi=150, bbox_inches='tight')
        print(f"Saved parameter space plot to {param_output}")
    else:
        plt.show()

    plt.close()


def generate_summary_stats(nodes: list[ExperimentNode]) -> dict:
    """Generate summary statistics from the exploration."""

    stats = {
        'total_iterations': len(nodes),
        'converged': sum(1 for n in nodes if n.status == 'converged'),
        'partial': sum(1 for n in nodes if n.status == 'partial'),
        'failed': sum(1 for n in nodes if n.status == 'failed'),
        'best_rollout_RMSE_mean': float('inf'),
        'best_iteration': None,
        'parameters_explored': set(),
        'unique_configs': 0,
    }

    configs_seen = set()
    for node in nodes:
        rmse = node.metrics.get('rollout_RMSE_mean', float('inf'))
        if rmse < stats['best_rollout_RMSE_mean']:
            stats['best_rollout_RMSE_mean'] = rmse
            stats['best_iteration'] = node.iteration

        if node.changed_param:
            stats['parameters_explored'].add(node.changed_param)

        config_tuple = tuple(sorted(node.config.items()))
        configs_seen.add(config_tuple)

    stats['unique_configs'] = len(configs_seen)
    stats['parameters_explored'] = list(stats['parameters_explored'])

    return stats


def compute_ucb_scores(analysis_path, ucb_path, c=1.0, current_log_path=None, current_iteration=None, block_size=12):
    """
    Parse analysis file, build exploration tree, compute UCB scores.

    Args:
        analysis_path: Path to analysis_experiment_*.md file
        ucb_path: Path to write UCB scores output
        c: Exploration constant (default sqrt(2) ~= 1.414)
        current_log_path: Path to current iteration's analysis.log (optional)
        current_iteration: Current iteration number (optional)
        block_size: Size of each simulation block (default 12)

    Returns:
        True if UCB scores were computed, False if no nodes found

    Note:
        When block_size > 0 and current_iteration is provided, only nodes
        from the current block are included in UCB scores. Block N covers
        iterations (N*block_size)+1 to (N+1)*block_size.
    """
    nodes = {}
    next_parent_map = {}  # maps iteration N -> parent for iteration N+1 (from "Next: parent=P")

    # parse previous iterations from analysis markdown file
    if os.path.exists(analysis_path):
        with open(analysis_path, 'r') as f:
            content = f.read()

        # Parse nodes from analysis file
        # Format: Node: id=N, parent=P, V=1, N_total=N
        # Metrics: ..., rollout_RMSE_mean=V, ...
        # Next: parent=P (specifies parent for next iteration)
        current_node = None

        lines = content.split('\n')
        for i, line in enumerate(lines):
            # Match iteration header: ## Iter N: [status] or ### Iter N: [status]
            # When we hit a new iteration, save the previous node if complete
            iter_match = re.match(r'##+ Iter (\d+):', line)
            if iter_match:
                # Save previous node if it has required fields
                if current_node is not None and 'id' in current_node and 'rollout_RMSE_mean' in current_node:
                    nodes[current_node['id']] = current_node
                current_iter = int(iter_match.group(1))
                current_node = {'iter': current_iter}
                continue

            # Match Node line
            node_match = re.match(r'Node: id=(\d+), parent=(\d+|None|root)', line)
            if node_match and current_node is not None:
                current_node['id'] = int(node_match.group(1))
                parent_str = node_match.group(2)
                # Treat parent=0, parent=None, or parent=root as root (no parent)
                if parent_str in ('None', '0', 'root'):
                    current_node['parent'] = None
                else:
                    current_node['parent'] = int(parent_str)
                continue

            # Match Next line: specifies parent for the NEXT iteration
            next_match = re.match(r'Next: parent=(\d+|root)', line)
            if next_match and current_node is not None:
                next_parent_str = next_match.group(1)
                if next_parent_str == 'root':
                    next_parent_map[current_node['iter']] = None
                else:
                    next_parent_map[current_node['iter']] = int(next_parent_str)
                continue

            # Match Mutation line
            mutation_match = re.match(r'Mutation: (.+)', line)
            if mutation_match and current_node is not None:
                current_node['mutation'] = mutation_match.group(1).strip()
                continue

            # Match Metrics line for rollout_RMSE_mean, training_g_phi_R2, training_accuracy, and training_final_loss
            metrics_match = re.search(r'rollout_RMSE_mean=([\d.eE+-]+|nan)', line)
            if metrics_match and current_node is not None:
                rmse_str = metrics_match.group(1).rstrip('.')  # Strip trailing period
                current_node['rollout_RMSE_mean'] = float(rmse_str) if rmse_str != 'nan' else 1.0
                # Also extract training_g_phi_R2 from same line
                g_phi_match = re.search(r'training_g_phi_R2=([\d.eE+-]+|nan)', line)
                if g_phi_match:
                    p_str = g_phi_match.group(1).rstrip('.')  # Strip trailing period
                    current_node['training_g_phi_R2'] = float(p_str) if p_str != 'nan' else 0.0
                else:
                    current_node['training_g_phi_R2'] = 0.0
                # Also extract training_accuracy (cluster_accuracy) from same line
                cluster_match = re.search(r'training_accuracy=([\d.eE+-]+|nan)', line)
                if cluster_match:
                    c_str = cluster_match.group(1).rstrip('.')  # Strip trailing period
                    current_node['cluster_accuracy'] = float(c_str) if c_str != 'nan' else -1.0
                else:
                    current_node['cluster_accuracy'] = -1.0
                # Also extract training_final_loss from same line
                loss_match = re.search(r'training_final_loss=([\d.eE+-]+|nan)', line)
                if loss_match:
                    l_str = loss_match.group(1).rstrip('.')
                    current_node['training_final_loss'] = float(l_str) if l_str != 'nan' else -1.0
                else:
                    current_node['training_final_loss'] = -1.0
                # Don't store node yet - continue collecting fields (Mutation comes after Metrics)
                continue

        # Save the last node if complete
        if current_node is not None and 'id' in current_node and 'rollout_RMSE_mean' in current_node:
            nodes[current_node['id']] = current_node

    # Apply next_parent_map: if iteration N-1 specified "Next: parent=P", use P as parent for node N
    for node_id, node in nodes.items():
        prev_iter = node_id - 1
        if prev_iter in next_parent_map:
            new_parent = next_parent_map[prev_iter]
            # Skip self-references (parallel mode: sibling slot pointing to itself)
            if new_parent == node_id:
                continue
            node['parent'] = new_parent

    # Add current iteration from analysis.log if not yet in markdown
    # Use parent from next_parent_map (from previous iteration's "Next: parent=P")
    if current_log_path and current_iteration and os.path.exists(current_log_path):
        with open(current_log_path, 'r') as f:
            log_content = f.read()

        # parse rollout_RMSE_mean from analysis.log (handles both = and : formats)
        rmse_match = re.search(r'rollout_RMSE_mean[=:]\s*([\d.eE+-]+|nan)', log_content)
        if rmse_match:
            rmse_str = rmse_match.group(1)
            rmse_value = float(rmse_str) if rmse_str != 'nan' else 1.0

            # parse training_g_phi_R2 from analysis.log
            g_phi_value = 0.0
            g_phi_match = re.search(r'training_g_phi_R2[=:]\s*([\d.eE+-]+|nan)', log_content)
            if g_phi_match:
                p_str = g_phi_match.group(1)
                g_phi_value = float(p_str) if p_str != 'nan' else 0.0

            # parse cluster_accuracy from analysis.log
            cluster_value = -1.0
            cluster_match = re.search(r'cluster_accuracy[=:]\s*([\d.eE+-]+|nan)', log_content)
            if cluster_match:
                c_str = cluster_match.group(1)
                cluster_value = float(c_str) if c_str != 'nan' else -1.0

            # parse training_time_min from analysis.log
            training_time_value = -1.0
            time_match = re.search(r'training_time_min[=:]\s*([\d.]+)', log_content)
            if time_match:
                training_time_value = float(time_match.group(1))

            # parse training_final_loss from analysis.log
            training_final_loss_value = -1.0
            loss_match = re.search(r'training_final_loss[=:]\s*([\d.eE+-]+|nan)', log_content)
            if loss_match:
                l_str = loss_match.group(1)
                training_final_loss_value = float(l_str) if l_str != 'nan' else -1.0

            if current_iteration in nodes:
                # Update existing node's metrics
                nodes[current_iteration]['rollout_RMSE_mean'] = rmse_value
                nodes[current_iteration]['training_g_phi_R2'] = g_phi_value
                nodes[current_iteration]['cluster_accuracy'] = cluster_value
                nodes[current_iteration]['training_time_min'] = training_time_value
                nodes[current_iteration]['training_final_loss'] = training_final_loss_value
            else:
                # Create new node using parent from previous iteration's "Next: parent=P"
                prev_iter = current_iteration - 1
                parent = next_parent_map.get(prev_iter, prev_iter if prev_iter in nodes else None)
                nodes[current_iteration] = {
                    'iter': current_iteration,
                    'id': current_iteration,
                    'parent': parent,
                    'rollout_RMSE_mean': rmse_value,
                    'training_g_phi_R2': g_phi_value,
                    'cluster_accuracy': cluster_value,
                    'training_time_min': training_time_value,
                    'training_final_loss': training_final_loss_value,
                }

    if not nodes:
        return False

    # Filter nodes to current block if block_size > 0 and current_iteration is provided
    # Block N covers iterations (N*block_size)+1 to (N+1)*block_size
    if block_size > 0 and current_iteration is not None:
        current_block = (current_iteration - 1) // block_size
        block_start = current_block * block_size + 1
        block_end = (current_block + 1) * block_size

        # Filter nodes to only include those in current block
        nodes = {node_id: node for node_id, node in nodes.items()
                 if block_start <= node_id <= block_end}

        # Update parent references: if parent is outside block, set to None (root)
        for node_id, node in nodes.items():
            if node['parent'] is not None and node['parent'] not in nodes:
                node['parent'] = None

    if not nodes:
        return False

    # Build tree structure: for each node, track children
    children = defaultdict(list)
    for node_id, node in nodes.items():
        if node['parent'] is not None:
            children[node['parent']].append(node_id)

    # Total number of nodes
    n_total = len(nodes)

    # Compute visits using PUCT backpropagation semantics:
    # - Each node starts with V=1 (its own creation visit)
    # - When a child is created, parent and all ancestors get V += 1
    # This means V(node) = 1 + number of descendants
    visits = {node_id: 1 for node_id in nodes}

    # Sort nodes by id to process in creation order (children after parents)
    sorted_node_ids = sorted(nodes.keys())

    # Backpropagate: for each node, increment all ancestors
    for node_id in sorted_node_ids:
        parent_id = nodes[node_id]['parent']
        visited = set()  # Protect against circular parent references
        while parent_id is not None and parent_id in nodes and parent_id not in visited:
            visited.add(parent_id)
            visits[parent_id] += 1
            parent_id = nodes[parent_id]['parent']
        if parent_id in visited:
            print(f"Warning: circular parent reference detected at node {node_id}")

    # Compute UCB for each node
    # Google PUCT formula: UCB(u) = RankScore(u) + c * sqrt(N_total) / (1 + V(u))
    # For cell-gnn: reward = max(0, 1 - rmse * 10) to normalize RMSE to 0-1 range
    ucb_scores = []
    for node_id, node in nodes.items():
        v = visits[node_id]
        rmse = node.get('rollout_RMSE_mean', 1.0)
        reward = max(0.0, 1.0 - rmse * 10.0)

        # PUCT exploration term: c * sqrt(N_total) / (1 + V)
        exploration_term = c * math.sqrt(n_total) / (1 + v)

        ucb = reward + exploration_term

        ucb_scores.append({
            'id': node_id,
            'parent': node['parent'],
            'visits': v,
            'mean_reward': reward,
            'ucb': ucb,
            'rollout_RMSE_mean': rmse,
            'training_g_phi_R2': node.get('training_g_phi_R2', 0.0),
            'cluster_accuracy': node.get('cluster_accuracy', -1.0),
            'training_time_min': node.get('training_time_min', -1.0),
            'training_final_loss': node.get('training_final_loss', -1.0),
            'mutation': node.get('mutation', ''),
            'is_current': node_id == current_iteration
        })

    # Sort by UCB descending (highest UCB = most promising to explore)
    ucb_scores.sort(key=lambda x: x['ucb'], reverse=True)

    # Write UCB scores to file
    with open(ucb_path, 'w') as f:
        # Include block information if block_size > 0
        if block_size > 0 and current_iteration is not None:
            current_block = (current_iteration - 1) // block_size
            block_start = current_block * block_size + 1
            block_end = (current_block + 1) * block_size
            f.write(f"=== UCB Scores (Simulation block {current_block}, iters {block_start}-{block_end}, N={n_total}, c={c}) ===\n\n")
        else:
            f.write(f"=== UCB Scores (N_total={n_total}, c={c}) ===\n\n")
        for score in ucb_scores:
            parent_str = score['parent'] if score['parent'] is not None else 'root'
            mutation_str = score.get('mutation', '')
            cluster_acc = score.get('cluster_accuracy', -1.0)
            training_time = score.get('training_time_min', -1.0)
            training_final_loss = score.get('training_final_loss', -1.0)
            line = (f"Node {score['id']}: UCB={score['ucb']:.3f}, "
                    f"parent={parent_str}, visits={score['visits']}, "
                    f"RMSE={score['rollout_RMSE_mean']:.4f}, "
                    f"g_phi_R2={score['training_g_phi_R2']:.3f}")
            # Add cluster accuracy if available (>= 0)
            if cluster_acc >= 0:
                line += f", Cluster={cluster_acc:.3f}"
            # Add training time if available (>= 0)
            if training_time >= 0:
                line += f", Time={training_time:.1f}"
            # Add training_final_loss if available (>= 0)
            if training_final_loss >= 0:
                line += f", Loss={training_final_loss:.4f}"
            if mutation_str:
                line += f", Mutation={mutation_str}"
            f.write(line + "\n")

    return True


def print_recommendations():
    """Print recommendations for improving the log format."""

    recommendations = """
=== RECOMMENDATIONS FOR IMPROVED LOG FORMAT ===

To enable better decision tree visualization and analysis, consider adding:

1. **Parent Iteration Reference**:
   Add an explicit parent reference to track true branching:
   ```
   ## Iter N: [status]
   Parent: M  # which iteration this branched from
   ```

2. **Branch Type Annotation**:
   Distinguish between exploration types:
   ```
   Branch: exploration|validation|refinement|boundary_search
   ```

3. **Decision Rationale**:
   Add structured reasoning for parameter choices:
   ```
   Rationale: [increase|decrease|reset] [param] because [reason]
   ```

4. **Exploration Phase**:
   Track the exploration strategy phase:
   ```
   Phase: initial_sweep|optimization|validation|boundary_mapping
   ```

5. **Cumulative Best**:
   Track the best config found so far:
   ```
   Best_so_far: iter=X, rollout_RMSE_mean=Y
   ```

6. **Parameter Delta**:
   Show relative change magnitude:
   ```
   Delta: lr_W *2 (doubled), lr /2 (halved)
   ```

Example improved format:
```
## Iter 15: converged
Parent: 14
Phase: optimization
Branch: exploration
Config: lr_W=5.0E-3, lr=3.5E-4, coeff_W_L1=0
Metrics: rollout_RMSE_mean=0.0123, training_g_phi_R2=0.9864, cluster_accuracy=0.85, training_final_loss=2.6281e+02
Observation: lr_W=5E-3 achieves low rollout RMSE
Rationale: increase lr_W to probe upper bound
Change: lr_W: 4.0E-3 -> 5.0E-3 (*1.25)
Best_so_far: iter=14, rollout_RMSE_mean=0.0098
```

This enables:
- True tree structure reconstruction
- Strategy phase analysis
- Decision pattern learning
- Automatic boundary detection
"""
    print(recommendations)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize experiment exploration as a tree structure'
    )
    parser.add_argument('input', type=str, nargs='?', default=None,
                       help='Path to analysis_experiment_*.md file')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output path for visualization (default: display)')
    parser.add_argument('--param-x', type=str, default='lr_W',
                       help='X-axis parameter for 2D plot (default: lr_W)')
    parser.add_argument('--param-y', type=str, default='lr',
                       help='Y-axis parameter for 2D plot (default: lr)')
    parser.add_argument('--recommendations', '-r', action='store_true',
                       help='Print recommendations for improved log format')

    args = parser.parse_args()

    if args.recommendations:
        print_recommendations()
        return

    if args.input is None:
        parser.error("input file is required unless --recommendations is used")

    # Parse the log file
    print(f"Parsing {args.input}...")
    nodes = parse_experiment_log(args.input)
    print(f"Found {len(nodes)} iterations")

    if not nodes:
        print("No iterations found in the file")
        return

    # Build tree structure
    nodes = build_tree_structure(nodes)

    # Generate summary
    stats = generate_summary_stats(nodes)
    print("\n=== Summary ===")
    print(f"Total iterations: {stats['total_iterations']}")
    print(f"Converged: {stats['converged']}, Partial: {stats['partial']}, Failed: {stats['failed']}")
    print(f"Best rollout_RMSE_mean: {stats['best_rollout_RMSE_mean']:.4f} (iter {stats['best_iteration']})")
    print(f"Parameters explored: {', '.join(stats['parameters_explored'])}")
    print(f"Unique configurations tested: {stats['unique_configs']}")

    # Set output path
    if args.output:
        output_path = args.output
    else:
        # Auto-generate output path
        input_path = Path(args.input)
        output_path = str(input_path.parent / f"{input_path.stem}_tree.png")

    # Create combined visualization (all panels in one figure)
    print("\nGenerating exploration tree visualization...")
    plot_exploration_tree(nodes, output_path,
                         title=f"Exploration Tree: {Path(args.input).stem}",
                         param_x=args.param_x,
                         param_y=args.param_y)

    print("\nDone!")
    print("\nFor recommendations on improving the log format, run with --recommendations")


if __name__ == '__main__':
    main()
