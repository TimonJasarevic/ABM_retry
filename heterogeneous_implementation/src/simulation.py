"""Run directly for animation and results, or use --headless for saved figures."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .network import SocialNetwork
    from .definitions import NetworkType
else:
    from network import SocialNetwork
    from definitions import NetworkType


class Simulation:
    def __init__(self, social_network, rounds=2000, rewire_prob=0,
                 max_steps=30, draw_every=50, burn_in=0, plot_network=True):
        if rounds < 1 or burn_in < 0:
            raise ValueError('rounds must be positive and burn_in nonnegative')
        if draw_every is not None and draw_every < 1:
            raise ValueError('draw_every must be positive or None')
        if not 0 <= rewire_prob <= 1:
            raise ValueError('rewire_prob must lie in [0, 1]')
        self.social_network = social_network
        self.rounds = rounds
        self.burn_in = burn_in
        self.rewire_prob = rewire_prob
        self.max_steps = max_steps
        self.draw_every = draw_every
        self.plot_network = plot_network
        self.plotter = None

    def _collect_metrics(self, round_number, rewired_edges):
        model = self.social_network
        agents = model.social_agents
        trust = [t for a in agents for t in a.trust.values()]
        experienced = [a for a in agents if a.direct_exposure_count > 0]
        return {
            **model.last_cascade, 'round': round_number,
            **model.baseline_metrics(),
            'mean_reputation': np.mean([a.r for a in agents]),
            'mean_fake_tendency': np.mean([a.fake_tendency for a in agents]),
            'mean_audience_response_score': np.mean([a.audience_response_score for a in agents]),
            **{f'mean_direct_{action}_rate': np.mean([getattr(a, f'direct_{action}_rate')
                for a in experienced]) if experienced else float('nan')
               for action in ('share', 'verification', 'discard')},
            'mean_trust': np.mean(trust) if trust else float('nan'),
            'trust_std': np.std(trust) if trust else float('nan'),
            'learned_trust_count': len(trust),
            **model.network_metrics(),
            'rewired_edges': rewired_edges,
            **model.last_rewiring,
        }

    def run(self, rolling_window=20, plot_results=True, save_csv=None,
            show=True, output_dir=None, action_rolling_window=500):
        if not isinstance(action_rolling_window, int) or action_rolling_window < 1:
            raise ValueError('action_rolling_window must be a positive integer')
        if not isinstance(rolling_window, int) or rolling_window < 1:
            raise ValueError('rolling_window must be a positive integer')
        if __package__:
            from .plotter import NetworkPlotter
        else:
            from plotter import NetworkPlotter
        self.plotter = NetworkPlotter(self.social_network, animate=show)
        history = []
        if self.plot_network:
            self.plotter.draw_network_graph('Initial network',  label_type='reputation')

        for round_number in range(1, self.rounds + self.burn_in + 1):
            self.social_network.simulation_step(max_steps=self.max_steps)
            rewired = self.social_network.rewire_network(self.rewire_prob)
            self.social_network.update_influencers()
            if round_number > self.burn_in:
                measured_round = round_number - self.burn_in
                history.append(self._collect_metrics(measured_round, rewired))
                if (self.plot_network and self.draw_every is not None
                        and measured_round % self.draw_every == 0):
                    truth = 'true' if self.social_network.last_message else 'fake'
                    self.plotter.draw_network_graph(f'Round {measured_round}: {truth} message', label_type='reputation')
        df = pd.DataFrame(history)
        if save_csv:
            df.to_csv(save_csv, index=False)
        summary = self._summary(df)
        print('\nMeasured simulation summary')
        print(summary.to_string(float_format=lambda x: f'{x:.4f}'))
        if plot_results:
            self.plotter.analyze_history(df, self.rewire_prob, rolling_window, action_rolling_window)
        if output_dir:
            self.plotter.save_figures(output_dir)
        if show:
            self.plotter.show_network_graph()
        return df, summary

    def _summary(self, df):
        fake = df[df.message_is_fake]
        exposure = df.exposed_count.sum()
        fake_exposure = fake.exposed_count.sum()
        values = {
            'measured_cascades': len(df), 'true_cascades': df.message_is_true.sum(),
            'fake_cascades': len(fake), 'mean_exposed_receivers': df.exposed_count.mean(),
            'mean_true_reach': df.loc[df.message_is_true, 'reach'].mean(),
            'mean_fake_reach': fake.reach.mean(),
            'verify_action_fraction': df.verification_count.sum() / exposure if exposure else np.nan,
            'share_action_fraction': df.share_action_count.sum() / exposure if exposure else np.nan,
            'discard_action_fraction': df.discarded_count.sum() / exposure if exposure else np.nan,
            'fake_propagation_rate': fake.fake_shares.sum() / fake_exposure if fake_exposure else np.nan,
            'fake_correction_rate': fake.corrected_fake_count.sum() / fake_exposure if fake_exposure else np.nan,
            'final_mean_trust': df.mean_trust.iloc[-1],
            'final_trust_std': df.trust_std.iloc[-1],
            'final_mean_fake_tendency': df.mean_fake_tendency.iloc[-1],
            'final_mean_reputation': df.mean_reputation.iloc[-1],
            'measured_rewired_edges': df.rewired_edges.sum(),
            **{f'measured_{name}': df[name].sum() for name in (
                'local_rewires', 'global_rewires', 'failed_rewire_attempts',
                'isolated_rewire_attempts')},
            'final_degree_std': df.degree_std.iloc[-1],
            'final_max_degree_ratio': df.max_degree_ratio.iloc[-1],
            'fraction_truncated': df.cascade_truncated.mean(),
            'fraction_frontier_censored': df.unexposed_frontier_at_cap.mean(),
            'final_min_degree': df.min_degree.iloc[-1],
            'final_isolates': df.isolated_count.iloc[-1],
            'final_components': df.component_count.iloc[-1],
            'final_largest_component_ratio': df.largest_component_ratio.iloc[-1],
            'final_degree_gini': df.degree_gini.iloc[-1],
        }
        return pd.Series(values, name='value')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless', action='store_true', help='Render without GUI windows')
    parser.add_argument('--output-dir', type=Path, help='Save network/results PNG and history CSV')
    parser.add_argument('--rounds', type=int, help='Default: 50000 cascades')
    parser.add_argument('--max-steps', type=int, default=20)
    parser.add_argument('--draw-every', type=int, default=1000)
    parser.add_argument('--rewire-prob', type=float, default=0.00025)
    parser.add_argument('--global-search-prob', type=float, default=0.20)
    parser.add_argument('--rewire-sensitivity', type=float, default=2.5)
    parser.add_argument('--action-rolling-window', type=int, default=500)
    parser.add_argument('--reach-rolling-window', type=int, default=500)
    args = parser.parse_args()
    rounds = args.rounds if args.rounds is not None else (50000 if args.headless else 50000)
    if args.headless:
        import matplotlib
        matplotlib.use('Agg')
    output = args.output_dir
    if args.headless and output is None:
        output = Path(__file__).resolve().parents[1] / 'results'
    if output:
        output.mkdir(parents=True, exist_ok=True)

    social_network = SocialNetwork(
        network_type=NetworkType.Regular, n=120, m=4, seed=42,
        global_search_prob=args.global_search_prob, rewire_sensitivity=args.rewire_sensitivity,
    )
    sim = Simulation(social_network, rounds=rounds, burn_in=0,
                     rewire_prob=args.rewire_prob, max_steps=args.max_steps,
                     draw_every=args.draw_every, plot_network=not args.headless)
    sim.run(show=not args.headless, output_dir=output,
            action_rolling_window=args.action_rolling_window, rolling_window=args.reach_rolling_window,
            save_csv=output / 'history.csv' if output else None)

if __name__ == '__main__':
    main()
