# /src/mlx_rl_trainer/monitoring/chart_generator.py
# Revision: 001
# Goal: Advanced chart generation with seaborn and matplotlib
# Type: New Code
# Description: Comprehensive visualization system for training metrics

import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

try:
    import seaborn as sns
    sns.set_theme(style="darkgrid")
    sns.set_palette("husl")
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

logger = logging.getLogger(__name__)


class ChartGenerator:
    """
    Advanced chart generation with seaborn and matplotlib.

    Features:
    - Multiple chart types (line, scatter, distribution, heatmap)
    - Seaborn-styled visualizations
    - Correlation matrices
    - Distribution plots
    - Time-series analysis
    - Multi-panel dashboards
    """

    def __init__(
        self,
        output_dir: Path,
        style: str = 'darkgrid',
        palette: str = 'husl',
        dpi: int = 150,
        figsize: Tuple[int, int] = (12, 8)
    ):
        """
        Initialize chart generator.

        Args:
            output_dir: Directory to save charts
            style: Seaborn style
            palette: Color palette
            dpi: Image resolution
            figsize: Default figure size
        """
        if not MPL_AVAILABLE:
            raise ImportError("matplotlib is required for ChartGenerator")

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.style = style
        self.palette = palette
        self.dpi = dpi
        self.figsize = figsize

        if SEABORN_AVAILABLE:
            sns.set_theme(style=style)
            sns.set_palette(palette)

        logger.info(f"Initialized ChartGenerator: {self.output_dir}")

    def plot_training_curves(
        self,
        data: Dict[str, List[float]],
        title: str = "Training Metrics",
        filename: str = "training_curves.png",
        smooth: bool = True,
        window: int = 10
    ) -> Path:
        """
        Plot multiple training curves.

        Args:
            data: Dictionary of metric_name -> values
            title: Plot title
            filename: Output filename
            smooth: Whether to apply smoothing
            window: Smoothing window size
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')

        metrics = list(data.keys())
        for idx, (ax, metric) in enumerate(zip(axes.flat, metrics[:4])):
            values = np.array(data[metric])
            steps = np.arange(len(values))

            # Plot raw data
            ax.plot(steps, values, alpha=0.3, label='Raw', linewidth=1)

            # Plot smoothed data
            if smooth and len(values) > window:
                smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
                smooth_steps = steps[window-1:]
                ax.plot(smooth_steps, smoothed, label='Smoothed', linewidth=2)

            ax.set_xlabel('Step')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(metric.replace('_', ' ').title())
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / filename
        fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Saved training curves: {output_path}")
        return output_path

    def plot_reward_distribution(
        self,
        rewards: List[float],
        filename: str = "reward_distribution.png",
        bins: int = 50
    ) -> Path:
        """Plot reward distribution with statistics."""
        if not SEABORN_AVAILABLE:
            logger.warning("Seaborn not available, using basic histogram")
            return self._plot_basic_distribution(rewards, filename, bins)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Histogram with KDE
        sns.histplot(rewards, bins=bins, kde=True, ax=axes[0])
        axes[0].set_xlabel('Reward')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Reward Distribution with KDE')
        axes[0].axvline(np.mean(rewards), color='r', linestyle='--', label=f'Mean: {np.mean(rewards):.3f}')
        axes[0].axvline(np.median(rewards), color='g', linestyle='--', label=f'Median: {np.median(rewards):.3f}')
        axes[0].legend()

        # Box plot
        sns.boxplot(y=rewards, ax=axes[1])
        axes[1].set_ylabel('Reward')
        axes[1].set_title('Reward Box Plot')

        plt.tight_layout()
        output_path = self.output_dir / filename
        fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Saved reward distribution: {output_path}")
        return output_path

    def plot_correlation_matrix(
        self,
        data: Dict[str, List[float]],
        filename: str = "correlation_matrix.png"
    ) -> Path:
        """Plot correlation matrix between metrics."""
        if not SEABORN_AVAILABLE or not PANDAS_AVAILABLE:
            logger.warning("Seaborn/Pandas not available, skipping correlation matrix")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Calculate correlation
        corr = df.corr()

        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            corr,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        ax.set_title('Metric Correlation Matrix', fontsize=16, fontweight='bold')

        plt.tight_layout()
        output_path = self.output_dir / filename
        fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Saved correlation matrix: {output_path}")
        return output_path

    def plot_gradient_flow(
        self,
        gradient_norms: Dict[str, List[float]],
        filename: str = "gradient_flow.png"
    ) -> Path:
        """Plot gradient flow across layers."""
        fig, ax = plt.subplots(figsize=(15, 6))

        for layer_name, norms in gradient_norms.items():
            steps = np.arange(len(norms))
            ax.plot(steps, norms, label=layer_name, alpha=0.7)

        ax.set_xlabel('Step')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Flow Across Layers', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        plt.tight_layout()
        output_path = self.output_dir / filename
        fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Saved gradient flow: {output_path}")
        return output_path

    def plot_memory_usage(
        self,
        memory_data: Dict[str, List[float]],
        filename: str = "memory_usage.png"
    ) -> Path:
        """Plot memory usage over time."""
        fig, ax = plt.subplots(figsize=(12, 6))

        for mem_type, values in memory_data.items():
            steps = np.arange(len(values))
            ax.plot(steps, values, label=mem_type, linewidth=2)

        ax.set_xlabel('Step')
        ax.set_ylabel('Memory (MB)')
        ax.set_title('Memory Usage Over Time', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add threshold line if memory exceeds 8GB
        max_memory = max(max(v) for v in memory_data.values())
        if max_memory > 8000:
            ax.axhline(8000, color='r', linestyle='--', label='8GB Threshold')

        plt.tight_layout()
        output_path = self.output_dir / filename
        fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Saved memory usage: {output_path}")
        return output_path

    def plot_token_distribution(
        self,
        thinking_tokens: List[int],
        answer_tokens: List[int],
        filename: str = "token_distribution.png"
    ) -> Path:
        """Plot distribution of thinking vs answer tokens."""
        if not SEABORN_AVAILABLE:
            return self._plot_basic_token_dist(thinking_tokens, answer_tokens, filename)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Thinking tokens histogram
        sns.histplot(thinking_tokens, bins=30, kde=True, ax=axes[0, 0])
        axes[0, 0].set_xlabel('Thinking Tokens')
        axes[0, 0].set_title('Thinking Token Distribution')
        axes[0, 0].axvline(np.mean(thinking_tokens), color='r', linestyle='--', label=f'Mean: {np.mean(thinking_tokens):.1f}')
        axes[0, 0].legend()

        # Answer tokens histogram
        sns.histplot(answer_tokens, bins=30, kde=True, ax=axes[0, 1])
        axes[0, 1].set_xlabel('Answer Tokens')
        axes[0, 1].set_title('Answer Token Distribution')
        axes[0, 1].axvline(np.mean(answer_tokens), color='r', linestyle='--', label=f'Mean: {np.mean(answer_tokens):.1f}')
        axes[0, 1].legend()

        # Scatter plot
        axes[1, 0].scatter(thinking_tokens, answer_tokens, alpha=0.5)
        axes[1, 0].set_xlabel('Thinking Tokens')
        axes[1, 0].set_ylabel('Answer Tokens')
        axes[1, 0].set_title('Thinking vs Answer Tokens')

        # Ratio distribution
        ratios = [t/max(a, 1) for t, a in zip(thinking_tokens, answer_tokens)]
        sns.histplot(ratios, bins=30, kde=True, ax=axes[1, 1])
        axes[1, 1].set_xlabel('Thinking/Answer Ratio')
        axes[1, 1].set_title('Token Ratio Distribution')

        plt.tight_layout()
        output_path = self.output_dir / filename
        fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Saved token distribution: {output_path}")
        return output_path

    def create_dashboard(
        self,
        stats_data: Dict[str, Any],
        filename: str = "training_dashboard.png"
    ) -> Path:
        """Create comprehensive training dashboard."""
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig)

        # Loss plot
        ax1 = fig.add_subplot(gs[0, :2])
        if 'loss' in stats_data:
            loss_data = stats_data['loss']
            steps = np.arange(len(loss_data))
            ax1.plot(steps, loss_data, linewidth=2)
            ax1.set_title('Training Loss', fontweight='bold')
            ax1.set_xlabel('Step')
            ax1.set_ylabel('Loss')
            ax1.grid(True, alpha=0.3)

        # Reward plot
        ax2 = fig.add_subplot(gs[1, :2])
        if 'reward_mean' in stats_data:
            reward_data = stats_data['reward_mean']
            steps = np.arange(len(reward_data))
            ax2.plot(steps, reward_data, color='green', linewidth=2)
            ax2.set_title('Average Reward', fontweight='bold')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Reward')
            ax2.grid(True, alpha=0.3)

        # Memory plot
        ax3 = fig.add_subplot(gs[2, :2])
        if 'memory_allocated_mb' in stats_data:
            mem_data = stats_data['memory_allocated_mb']
            steps = np.arange(len(mem_data))
            ax3.plot(steps, mem_data, color='red', linewidth=2)
            ax3.set_title('Memory Usage', fontweight='bold')
            ax3.set_xlabel('Step')
            ax3.set_ylabel('Memory (MB)')
            ax3.grid(True, alpha=0.3)

        # Statistics panel
        ax4 = fig.add_subplot(gs[0, 2])
        ax4.axis('off')
        stats_text = "Training Statistics\n\n"
        if 'loss' in stats_data and len(stats_data['loss']) > 0:
            stats_text += f"Final Loss: {stats_data['loss'][-1]:.4f}\n"
        if 'reward_mean' in stats_data and len(stats_data['reward_mean']) > 0:
            stats_text += f"Final Reward: {stats_data['reward_mean'][-1]:.4f}\n"
        ax4.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')

        # Reward distribution
        ax5 = fig.add_subplot(gs[1, 2])
        if 'reward_mean' in stats_data and len(stats_data['reward_mean']) > 0:
            if SEABORN_AVAILABLE:
                sns.histplot(stats_data['reward_mean'], bins=20, kde=True, ax=ax5)
            else:
                ax5.hist(stats_data['reward_mean'], bins=20, alpha=0.7)
            ax5.set_title('Reward Distribution', fontweight='bold')
            ax5.set_xlabel('Reward')

        # Learning rate
        ax6 = fig.add_subplot(gs[2, 2])
        if 'learning_rate' in stats_data:
            lr_data = stats_data['learning_rate']
            steps = np.arange(len(lr_data))
            ax6.plot(steps, lr_data, color='purple', linewidth=2)
            ax6.set_title('Learning Rate', fontweight='bold')
            ax6.set_xlabel('Step')
            ax6.set_ylabel('LR')
            ax6.set_yscale('log')

        plt.tight_layout()
        output_path = self.output_dir / filename
        fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Saved training dashboard: {output_path}")
        return output_path

    def _plot_basic_distribution(
        self,
        data: List[float],
        filename: str,
        bins: int
    ) -> Path:
        """Fallback basic distribution plot."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(data, bins=bins, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution')
        ax.axvline(np.mean(data), color='r', linestyle='--', label=f'Mean: {np.mean(data):.3f}')
        ax.legend()

        output_path = self.output_dir / filename
        fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)

        return output_path

    def _plot_basic_token_dist(
        self,
        thinking: List[int],
        answer: List[int],
        filename: str
    ) -> Path:
        """Fallback basic token distribution."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].hist(thinking, bins=30, alpha=0.7)
        axes[0].set_xlabel('Thinking Tokens')
        axes[0].set_title('Thinking Distribution')

        axes[1].hist(answer, bins=30, alpha=0.7)
        axes[1].set_xlabel('Answer Tokens')
        axes[1].set_title('Answer Distribution')

        output_path = self.output_dir / filename
        fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)

        return output_path

    def cleanup(self) -> None:
        """Clean up resources."""
        plt.close('all')


# Dependencies: matplotlib, seaborn, pandas, numpy
# Install: pip install matplotlib seaborn pandas numpy
# Usage: chart_gen = ChartGenerator(output_dir=Path("./charts"))
#        chart_gen.plot_training_curves(data={'loss': [...]})
# Status: Complete and commit-ready
