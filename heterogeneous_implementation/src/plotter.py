"""Stable-layout network drawing, time-series dashboard and agent distributions."""
from pathlib import Path
import math

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import networkx as nx
import numpy as np


class NetworkPlotter:
    def __init__(self, social_network, animate=True):
        self.social_network = social_network
        self.G = social_network.G
        self.animate = animate
        self.network_fig = self.network_ax = None
        self.result_fig = None
        self.heterogeneity_fig = None
        self.structure_figures = {}
        self.pos = None
        self.frames_drawn = 0
        self.panels = []

    def draw_network_graph(self, title='Social network', label_type=None):
        if self.network_fig is None:
            self.network_fig, self.network_ax = plt.subplots(figsize=(10, 8))
            self.pos = nx.spring_layout(self.G, seed=42, iterations=150)
        if not plt.fignum_exists(self.network_fig.number):
            return  # Closing the animation window does not interrupt the model.
        colors = {'Unaware': '#cbd5e1', 'TrueBeliever': '#22a884',
                  'FalseBeliever': '#e76f51', 'Corrected': '#6d5bd0',
                  'Discarded': '#555555'}
        agents = self.social_network.social_agents
        nodes = list(self.G.nodes())
        self.network_ax.clear()
        nx.draw_networkx_edges(self.G, self.pos, ax=self.network_ax, alpha=0.25)

        degrees = dict(self.G.degree())
        min_degree = min(degrees.values())
        max_degree = max(degrees.values())
        nx.draw_networkx_nodes(
            self.G, self.pos, ax=self.network_ax,
            # node_color=[colors[agents[n].message_state.name] for n in nodes],
            

            node_size = [
                100 + 450 * (
                    (degrees[n] - min_degree) / (max_degree - min_degree)
                    if max_degree > min_degree else 0.5
                )
                for n in nodes
            ],
                                
            edgecolors=['#f6a800' if agents[n].is_influencer else 'white' for n in nodes],
            linewidths=[2.5 if agents[n].is_influencer else 0.5 for n in nodes]
        )
        if label_type:
            labels = {n: f'{agents[n].r:.1f}' if label_type == 'reputation' else str(n)
                      for n in nodes}
            nx.draw_networkx_labels(self.G, self.pos, labels, ax=self.network_ax, font_size=7)
        labels = ['Unaware', 'True forward', 'Fake forward', 'Verified fake', 'Discarded']
        handles = [Line2D([], [], marker='o', linestyle='', color=c, label=label)
                   for c, label in zip(colors.values(), labels)]
        handles.append(Line2D([], [], marker='o', linestyle='', markerfacecolor='white',
                              markeredgecolor='#f6a800', label='Top 5% degree (descriptive)'))
        self.network_ax.legend(handles=handles, loc='lower left', fontsize=8)
        metrics = self.social_network.network_metrics()
        self.network_ax.set_title(title +
            f"\nIsolates: {metrics['isolated_count']} · Largest component: "
            f"{metrics['largest_component_ratio']:.0%} · Degree CV: {metrics['degree_cv']:.2f}")
        self.network_ax.axis('off')
        self.network_fig.tight_layout()
        self.network_fig.canvas.draw()
        self.frames_drawn += 1
        if self.animate:
            plt.show(block=False)
            plt.pause(0.03)

    @staticmethod
    def rolling_action_rates(df, window):
        if not isinstance(window, int) or window < 1:
            raise ValueError('action_rolling_window must be a positive integer')
        counts = df[['share_action_count', 'verification_count', 'discarded_count']]
        exposures = df.exposed_count.rolling(window, min_periods=1).sum()
        return counts.rolling(window, min_periods=1).sum().div(
            exposures.where(exposures > 0), axis=0)

    @staticmethod
    def rolling_reach(df, window):
        if not isinstance(window, int) or window < 1:
            raise ValueError('rolling_window must be a positive integer')
        # Windows are measured in simulation rounds, not number of same-type cascades.
        return {truth: df.reach.where(df.message_is_true == truth).rolling(
                    window, min_periods=1).mean() for truth in (True, False)}

    def distribution_data(self):
        agents = self.social_network.social_agents
        effective = []
        for agent in agents:
            neighbors = list(self.G.neighbors(agent.node_id))
            effective.append(np.mean([agent.get_trust(j) for j in neighbors])
                             if neighbors else agent.default_trust)
        # Stored initial traits preserve the actual pre-learning state.
        # Isolates use default trust as an explicit fallback at both times.
        return {
            'Public reputation distribution': ([a.initial_reputation for a in agents],
                                                [a.r for a in agents]),
            'Effective neighbor trust distribution': ([a.default_trust for a in agents], effective),
            'Deception tendency distribution': ([a.initial_fake_tendency for a in agents],
                                                [a.fake_tendency for a in agents]),
        }

    def draw_heterogeneity(self):
        self.heterogeneity_fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
        bins = np.linspace(0, 1, 21)
        for ax, (title, (initial, final)) in zip(axes, self.distribution_data().items()):
            ax.hist(initial, bins=bins, alpha=.5, color='#4477aa', label='Initial')
            ax.hist(final, bins=bins, alpha=.5, color='#ee7733', label='Final')
            ax.set(title=title, xlabel='Value', ylabel='Agents', xlim=(0, 1))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.legend()
            ax.spines[['top', 'right']].set_visible(False)
        self.heterogeneity_fig.suptitle('Initial versus final agent distributions')
        self.heterogeneity_fig.canvas.draw()

    def analyze_history(self, df, rewire_prob, rolling_window=20, action_rolling_window=500):
        model = self.social_network
        action_rates = self.rolling_action_rates(df, action_rolling_window)
        reach = self.rolling_reach(df, rolling_window)
        self.panels = ['reach', 'actions', 'reputation', 'reach_over_time',
                       'trust', 'adaptation', 'network', 'degree']
        self.result_fig, axes = plt.subplots(3, 3, figsize=(15, 10),
                                             squeeze=False, layout='constrained')
        for ax, panel in zip(axes.flat, self.panels):
            if panel == 'reach':
                bins = np.linspace(0, 1, 16)
                for truth, label, color in [(True, 'True', '#22a884'), (False, 'Fake', '#e76f51')]:
                    subset = df.loc[df.message_is_true == truth, 'reach'].dropna()
                    if len(subset):
                        ax.hist(subset, bins=bins, alpha=.6, label=label, color=color)
                ax.set(title='Cascade reach distributions', xlabel='Exposed receivers / (n − 1)', ylabel='Cascades')
                if ax.patches:
                    ax.legend()
            elif panel == 'actions':
                for column, label, color in [('share_action_count', 'SHARE', '#22a884'),
                                             ('verification_count', 'VERIFY', '#6d5bd0'),
                                             ('discarded_count', 'DISCARD', '#555555')]:
                    ax.plot(df['round'], action_rates[column], label=label, color=color)
                ax.set(title='Receiver actions over time', xlabel=f'Round (window {action_rolling_window})',
                       ylabel='Fraction of exposures')
                ax.legend(fontsize=8)
            elif panel == 'reach_over_time':
                for truth, label, color in [(True, 'True', '#22a884'), (False, 'Fake', '#e76f51')]:
                    ax.plot(df['round'], reach[truth], label=label, color=color)
                ax.set(title='Cascade reach over time', xlabel=f'Round (window {rolling_window})',
                       ylabel='Exposed receivers / (n − 1)', ylim=(0, 1))
                ax.legend(fontsize=8)
            elif panel == 'trust':
                ax.plot(df['round'], df.mean_trust, label='Mean learned dyad trust')
                ax.fill_between(df['round'], df.mean_trust - df.trust_std,
                                df.mean_trust + df.trust_std, alpha=.2, label='±1 SD (not uncertainty)')
                ax.set(title='Private trust over time', xlabel='Round')
                ax.legend(fontsize=8)
            elif panel == 'network':
                ax.plot(df['round'], df.degree_cv, label='Degree coefficient of variation')
                ax.plot(df['round'], df.max_degree_ratio, label='Max degree / (n − 1)')
                ax.set(title='Connectivity concentration', xlabel='Round', ylabel='Ratio')
                ax.legend(fontsize=8)
            elif panel == 'degree':
                degrees, counts = np.unique([d for _, d in self.G.degree()], return_counts=True)
                ax.bar(degrees, counts, width=.8, color='#4477aa')
                initial = list(model.initial_degrees.values())
                label = 'Initial degree' if len(set(initial)) == 1 else 'Initial mean degree'
                ax.axvline(np.mean(initial), linestyle='--', color='#ee7733', label=label)
                ax.axvline(np.mean([d for _, d in self.G.degree()]), linestyle=':',
                           color='#222222', label='Final mean degree')
                ax.set(title='Final degree distribution', xlabel='Degree', ylabel='Nodes')
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                ax.legend(fontsize=8)
            else:
                column, title = ('mean_reputation', 'Mean public reputation') if panel == 'reputation' else (
                    'mean_fake_tendency', 'Mean originator deception tendency')
                ax.plot(df['round'], df[column])
                ax.set(title=title, xlabel='Round')
                if panel == 'adaptation':
                    ax.axhline(1 - np.mean([a.baseline_truthfulness for a in model.social_agents]),
                               linestyle=':', color='grey', label='Mean initial tendency')
                    ax.legend(fontsize=8)
            ax.spines[['top', 'right']].set_visible(False)
        self.result_fig.delaxes(axes.flat[-1])
        self.result_fig.suptitle('Misinformation ABM • measured cascades')
        self.result_fig.canvas.draw()
        self.draw_heterogeneity()
        for name, column, title, ylabel in (
            ('degree_gini_over_time', 'degree_gini', 'Degree inequality over time', 'Degree Gini'),
            ('isolates_over_time', 'isolated_count', 'Isolated agents over time', 'Number of isolates'),
        ):
            fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
            ax.axhline(0, color='grey', linestyle=':', alpha=.4)
            ax.plot(df['round'], df[column], color='#4477aa')
            ax.set(title=title, xlabel='Cascade', ylabel=ylabel)
            ax.spines[['top', 'right']].set_visible(False)
            if column == 'isolated_count':
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            self.structure_figures[name] = fig

    def save_figures(self, output_dir):
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        for name, fig in [('network', self.network_fig), ('results', self.result_fig),
                          ('heterogeneity', self.heterogeneity_fig), *self.structure_figures.items()]:
            if fig is not None:
                fig.savefig(output / f'{name}.png', dpi=140)

    def show_network_graph(self):
        plt.show()
