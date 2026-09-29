import mesa
import networkx as nx
import math
from mesa.space import NetworkGrid
if __package__:
    from .definitions import NetworkType, MessageState, Action
    from . import agents
else:
    from definitions import NetworkType, MessageState, Action
    import agents


# Model parameters:
#
# Network structure
# network_type: Type of social network used to initialize the graph.
# n: Number of agents/nodes.
# p: Connection probability for network types that use random edge formation.
# m: Degree/attachment parameter used by network generators where applicable.
# seed: Random seed for reproducibility.
#
# Agent heterogeneity
# mean_truthfulness: Mean baseline probability that an agent creates true information.
# mean_default_trust: Mean initial trust an agent assigns to an unknown sender.
# mean_initial_reputation: Mean initial public reputation of agents.
# baseline_heterogeneity_concentration: Controls variation around heterogeneous baseline means;
#                                      higher values make agents more similar.
#
# Receiver decision-making
# verify_cost: Cost paid by a receiver for verifying a message.
# rationality: Strength of preference for higher-utility actions in the softmax decision rule.
# risk_aversion: Penalty for uncertain/variable outcomes.
# loss_aversion: Extra weight assigned to negative outcomes relative to positive outcomes.
#
# Trust and reputation
# trust_learning_rate: Speed at which private sender-specific trust updates after verification.
# trust_weight: Weight of private trust in perceived message credibility.
# reputation_weight: Weight of public source reputation in perceived message credibility.
# reputation_learning_rate: Speed at which public reputation updates after verified information.
# reputation_noise: Noise when agents observe another agent's public reputation.
#
# Audience-response / fake-production adaptation
# audience_response_sensitivity: Log-odds response to direct audience feedback.
# audience_response_reference: Neutral audience-response reference.
# audience_response_learning_rate: Recency-weighted direct audience learning rate.
#
# Receiver payoffs
# share_reward: General benefit from sharing/participating in information propagation.
# receiver_true_reward: Additional reward for sharing information that is actually true.
# false_share_cost: Cost to a receiver for sharing information that is actually false.
# detect_fake_reward: Reward for successfully detecting fake information through verification.
#
# Sender/originator payoffs
# sender_forward_reward: Reward to a normal sender when their message is forwarded.
# sender_fake_penalty: Penalty to the immediate sender when fake information is verified.
# originator_true_reward: Reward to the original source when a true message is forwarded.
# originator_fake_reward: Reward to the original source when a fake message is forwarded.
# originator_true_cost: Cost of producing a true message.
# originator_fake_cost: Cost of producing a fake message.
#
# Network adaptation
# rewire_sensitivity: Strength of trust/reputation preferences during rewiring;
#                     low-trust ties are more likely to be dropped and high-reputation
#                     candidates are more likely to be selected.
# global_search_prob: Probability that rewiring searches globally rather than through
#                     local friends-of-friends.
class SocialNetwork(mesa.Model):
    def __init__(self, network_type=NetworkType.Regular, n=120, p=0.04, m=4,
                 mean_truthfulness=0.7, verify_cost=0.25, rationality=2.5, seed=42,
                 trust_learning_rate=0.2, trust_weight=0.5, mean_default_trust=0.5,
                 reputation_weight=0.5,
                 audience_response_sensitivity=5.0, risk_aversion=0.15, loss_aversion=1.75,
                 reputation_learning_rate=0.2, reputation_noise=0.0,
                 receiver_true_reward=0.5, false_share_cost=1.0,
                 detect_fake_reward=0, sender_forward_reward=0.5,
                 originator_true_reward=0.5, originator_fake_reward=1.0,
                 originator_true_cost=0.5, originator_fake_cost=0.1,
                 sender_fake_penalty=1.0, rewire_sensitivity=2.5,
                 share_reward=0.5,
                 audience_response_reference=0.0, audience_response_learning_rate=0.2,
                 mean_initial_reputation=0.5, baseline_heterogeneity_concentration=20.0,
                 global_search_prob=0.20):
        super().__init__(rng=seed)
        parameters = locals().copy()
        for name in ('trust_learning_rate', 'trust_weight',
                     'reputation_learning_rate',
                     'reputation_weight', 'p', 'audience_response_learning_rate',
                     'global_search_prob'):
            if not 0 <= parameters[name] <= 1:
                raise ValueError(f'{name} must lie in [0, 1]')
        for name in ('verify_cost', 'rationality', 'risk_aversion', 'reputation_noise',
                     'receiver_true_reward', 'false_share_cost', 'detect_fake_reward',
                     'sender_forward_reward', 'originator_true_reward', 'originator_fake_reward',
                     'originator_true_cost', 'originator_fake_cost',
                     'rewire_sensitivity', 'sender_fake_penalty', 'share_reward',
                     'audience_response_sensitivity'):
            if not math.isfinite(parameters[name]) or parameters[name] < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if not math.isfinite(loss_aversion) or loss_aversion < 1:
            raise ValueError('loss_aversion must be finite and >= 1')
        if not isinstance(n, int) or n < 1:
            raise ValueError('n must be a positive integer')
        if not math.isclose(trust_weight + reputation_weight,
                            1.0, rel_tol=0, abs_tol=1e-12):
            raise ValueError('belief weights must sum to 1')
        if verify_cost <= 0 or receiver_true_reward <= 0 or sender_forward_reward <= 0:
            raise ValueError('verification cost and true/ordinary forward rewards must be positive')
        if not 0 < originator_true_reward < originator_fake_reward:
            raise ValueError('require 0 < originator_true_reward < originator_fake_reward')
        if not 0 <= originator_fake_cost < originator_true_cost:
            raise ValueError('require 0 <= originator_fake_cost < originator_true_cost')
        if not -1 <= audience_response_reference <= 1:
            raise ValueError('audience_response_reference must lie in [-1, 1]')
        if (not math.isfinite(baseline_heterogeneity_concentration)
                or baseline_heterogeneity_concentration <= 0):
            raise ValueError('baseline_heterogeneity_concentration must be finite and positive')
        for name in ('mean_default_trust', 'mean_initial_reputation', 'mean_truthfulness'):
            mean = parameters[name]
            if not 0 < mean < 1:
                raise ValueError(f'{name} must lie strictly between 0 and 1 for Beta draws')
            if not (mean * baseline_heterogeneity_concentration > 0
                    and (1 - mean) * baseline_heterogeneity_concentration > 0):
                raise ValueError(f'{name} produces an unrepresentable Beta shape')
        # Store the simple model parameters without a separate configuration class.
        for name, value in parameters.items():
            if name not in ('self', '__class__'):
                setattr(self, name, value)
        self.last_rewiring = dict(local_rewires=0, global_rewires=0,
                                  failed_rewire_attempts=0, isolated_rewire_attempts=0)
        self.total_verification_cost_paid = 0.0
        self.last_message = None
        self.last_initiator_id = None
        self.last_cascade = None
        self.current_message_publicly_verified = False
        self._pending_public_verification = None
        self.social_agents = []
        self.influencers = []
        self.normal_users = []
        self.influencer_nodes = []
        self._generate_network(network_type)
        self.initial_degrees = dict(self.G.degree())
        self.grid = NetworkGrid(self.G)
        self._init_agents()
        self.update_influencers()

    def _check_for_influencers(self):
        top_fraction = 0.05
        number_of_influencers = max(1, int(top_fraction * self.G.number_of_nodes()))

        sorted_degree_nodes = sorted(
            self.G.nodes(),
            key=lambda node: self.G.degree[node],
            reverse=True
        )

        top_degree_nodes = sorted_degree_nodes[:number_of_influencers]

        return top_degree_nodes

    def update_influencers(self):
        old_influencer_nodes = set(self.influencer_nodes)
        updated_influencer_nodes = set(self._check_for_influencers())

        removed_influencer_nodes = old_influencer_nodes - updated_influencer_nodes
        new_influencer_nodes = updated_influencer_nodes - old_influencer_nodes

        # Remove influencer status
        for node_id in removed_influencer_nodes:
            self.social_agents[node_id].is_influencer = False

        # Add influencer status
        for node_id in new_influencer_nodes:
            self.social_agents[node_id].is_influencer = True

        # Store all current influencer node IDs
        self.influencer_nodes = sorted(updated_influencer_nodes)

        # Store all current influencer agent objects
        self.influencers = [
            self.social_agents[node_id]
            for node_id in self.influencer_nodes
        ]
        self.normal_users = list(set(self.social_agents).difference(self.influencers))



    def _generate_network(self, network_type):
        if network_type == NetworkType.Random:
            self.G = nx.erdos_renyi_graph(self.n, self.p, seed=self.random)
            return 1

        elif network_type == NetworkType.ScaleFree:
            self.G = nx.barabasi_albert_graph(self.n, self.m, seed=self.random)
            return 1
        elif network_type == NetworkType.Regular:
            degree = self.m

            if degree >= self.n:
                raise ValueError("Regular graph degree must be smaller than n")

            if (degree * self.n) % 2 != 0:
                raise ValueError("For a regular graph, degree * n must be even")

            self.G = nx.random_regular_graph(
                d=degree,
                n=self.n, seed=self.random
            )
            return 1

        else:
            raise ValueError("Unknown network type")

    def draw_baseline(self, mean):
        """One seeded Beta draw; no sample-mean correction or round-by-round redraw."""
        concentration = self.baseline_heterogeneity_concentration
        return self.random.betavariate(mean * concentration, (1 - mean) * concentration)

    def baseline_metrics(self):
        """Population SD and mean of persistent initial traits (not dynamic reputation)."""
        metrics = {}
        for label, attribute in (('baseline_trust', 'default_trust'),
                                 ('initial_reputation', 'initial_reputation'),
                                 ('baseline_truthfulness', 'baseline_truthfulness')):
            values = [getattr(a, attribute) for a in self.social_agents]
            mean = math.fsum(values) / len(values)
            metrics[f'mean_{label}'] = mean
            metrics[f'std_{label}'] = math.sqrt(math.fsum((v - mean)**2 for v in values) / len(values))
        return metrics

    def _init_agents(self):
        for node in sorted(self.G.nodes()):
            agent = None

            if node in self.influencer_nodes:
                agent = agents.SocialAgent(
                    self,
                    node_id=node,
                    is_influencer=True
                )
                self.influencers.append(agent)

            else:
                agent = agents.SocialAgent(
                    self,
                    node_id=node,
                    is_influencer=False
                )

            self.grid.place_agent(agent, node)
            self.social_agents.append(agent)
        self.normal_users = list(set(self.social_agents).difference(self.influencers))


    def simulation_step(self, max_steps=30):
        if not isinstance(max_steps, int) or max_steps < 0:
            raise ValueError('max_steps must be a nonnegative integer')
        self._reset_message_states()

        # choose who starts the message
        initiator_agent = self._choose_initiator()
        # initiator creates the message
        tendency_before = initiator_agent.fake_tendency
        message = initiator_agent.initiate_message()
        self.current_message_publicly_verified = False
        self._pending_public_verification = None
        production_cost = self.originator_true_cost if message else self.originator_fake_cost
        initiator_agent.pending_payoff -= production_cost
        originator_propagation_reward = 0.0
        originator_sender_penalties = 0.0
        ordinary_sender_reward = 0.0
        sender_penalties = 0.0
        originator_direct_exposures = 0
        originator_direct_verifications = 0
        originator_direct_shares = 0
        originator_direct_discards = 0

        # Store cascade-level information for output metrics.
        self.last_message = message
        self.last_initiator_id = initiator_agent.node_id

        active_agents = {initiator_agent}

        # active_sharers contains agent objects
        active_sharers = [initiator_agent]

        step = 0

        while active_sharers and step < max_steps:
            exposed_agents = self._collect_exposures(active_sharers)
            if not exposed_agents:
                break
            new_sharers = []
            active_agents.update(receiver for _, receiver in exposed_agents)


            # Canonical receiver order also assigns RNG draws consistently when
            # graph insertion or exposure-list order changes. Reputation stays frozen.
            for sender_agents, receiver_agent in sorted(exposed_agents, key=lambda pair: pair[1].node_id):
                sender_ids = tuple(a.node_id for a in sender_agents)
                agent_response, attributions = receiver_agent.receive_message(
                    message, sender_ids, initiator_agent.node_id)
                for sender_id, sender_payoff in attributions.items():
                    sender_agent = self.social_agents[sender_id]
                    sender_agent.pending_payoff += sender_payoff
                    sender_penalties += max(0.0, -sender_payoff)
                    if sender_agent is initiator_agent:
                        originator_propagation_reward += max(0.0, sender_payoff)
                        originator_sender_penalties += max(0.0, -sender_payoff)
                    else:
                        ordinary_sender_reward += max(0.0, sender_payoff)
                if initiator_agent.node_id in sender_ids:
                    originator_direct_exposures += 1
                    originator_direct_verifications += int(receiver_agent.last_action == Action.VERIFY)
                    originator_direct_shares += int(receiver_agent.last_action == Action.SHARE)
                    originator_direct_discards += int(receiver_agent.last_action == Action.DISCARD)
                if agent_response == 1:
                    new_sharers.append(receiver_agent)

            self.commit_public_verification()
            active_sharers = new_sharers
            step += 1

        # Learn from direct receiver actions once per cascade, independently of payoff.
        initiator_agent.observe_direct_audience(
            originator_direct_shares, originator_direct_verifications, originator_direct_discards)
        originator_payoff = -production_cost + originator_propagation_reward - originator_sender_penalties
        receivers = active_agents - {initiator_agent}
        exposed = len(receivers)
        verified = sum(a.last_action == Action.VERIFY for a in receivers)
        shares = sum(a.last_action == Action.SHARE for a in receivers)
        discarded = sum(a.last_action == Action.DISCARD for a in receivers)
        forwarders = shares + (verified if message else 0)
        self.last_cascade = {
            'message_is_true': message, 'message_is_fake': not message,
            'originator_id': initiator_agent.node_id,
            'originator_fake_tendency_before': tendency_before,
            'originator_fake_tendency_after': initiator_agent.fake_tendency,
            'originator_direct_exposures': originator_direct_exposures,
            'originator_direct_verifications': originator_direct_verifications,
            'originator_direct_shares': originator_direct_shares,
            'originator_direct_discards': originator_direct_discards,
            'originator_audience_response_score': initiator_agent.audience_response_score,
            'originator_multi_sender': originator_payoff,
            'originator_production_cost': production_cost,
            'originator_propagation_reward': originator_propagation_reward,
            'originator_sender_penalties': originator_sender_penalties,
            'ordinary_sender_reward': ordinary_sender_reward,
            'sender_penalties': sender_penalties,
            'receiver_multi_sender': sum(a.receiver_payoff for a in receivers),
            'public_reputation_updates': int(self.current_message_publicly_verified),
            'exposed_count': exposed, 'shared_count': forwarders,
            'forwarder_count': forwarders, 'share_action_count': shares,
            'verification_count': verified, 'discarded_count': discarded,
            'corrected_fake_count': verified if not message else 0,
            'true_shares': forwarders if message else 0,
            'fake_shares': forwarders if not message else 0,
            'cascade_depth': step,
            # Conservative cap flag, plus a stricter flag for omitted exposures.
            'cascade_truncated': bool(active_sharers and step == max_steps),
            'unexposed_frontier_at_cap': bool(step == max_steps and any(
                self.social_agents[j].message_state == MessageState.Unaware
                for a in active_sharers for j in self.G.neighbors(a.node_id))),
            'total_objective_multi_sender': sum(a.pending_payoff for a in self.social_agents),
            'verification_cost': verified * self.verify_cost,
            'reach': exposed / (self.n - 1) if self.n > 1 else float('nan'),
            'fake_reach': exposed / (self.n - 1) if self.n > 1 and not message else float('nan'),
            'fake_propagation_rate': forwarders / exposed if exposed and not message else float('nan'),
            'verification_rate': verified / exposed if exposed else float('nan'),
            'fake_correction_rate': verified / exposed if exposed and not message else float('nan'),
        }
        self.update_agents()
        return active_agents

    def queue_public_verification(self, originator_id, message):
        if not self.current_message_publicly_verified and self._pending_public_verification is None:
            self._pending_public_verification = (originator_id, float(message))

    def commit_public_verification(self):
        """One weak public signal per cascade, applied only at a wave boundary."""
        if self._pending_public_verification is not None:
            source, observation = self._pending_public_verification
            self.social_agents[source].update_reputation(observation)
            self.current_message_publicly_verified = True
            self._pending_public_verification = None

    def _collect_exposures(self, active_sharers):
        """All distinct transmitting neighbors, one event per unaware receiver."""
        candidates = {}
        for sender in active_sharers:
            for node in self.G.neighbors(sender.node_id):
                if self.social_agents[node].message_state == MessageState.Unaware:
                    candidates.setdefault(node, {})[sender.node_id] = sender
        return [(tuple(candidates[node][j] for j in sorted(candidates[node])), self.social_agents[node])
                for node in sorted(candidates)]

    def _choose_initiator(self):
        return self.random.choice(self.social_agents)

    def _reset_message_states(self):
        for agent in self.social_agents:
            agent.message_state = MessageState.Unaware
            agent.last_action = None

    def update_agents(self):
        for agent in self.social_agents:
            agent.update()

    def propagation_reward(self, sender_id, originator_id, message):
        """Reward the immediate sender only when its receiver forwards."""
        if sender_id == originator_id:
            return self.originator_true_reward if message else self.originator_fake_reward
        return self.sender_forward_reward

    def _rewire_candidates(self, agent_id, global_search=False):
        neighbors = set(self.G.neighbors(agent_id))
        pool = (set(self.G.nodes()) if global_search else
                {k for j in neighbors for k in self.G.neighbors(j)})
        return sorted(pool - neighbors - {agent_id})

    def _rewire_softmax(self, scores):
        if not scores:
            return []
        peak = max(scores)
        # Stabilize exponentials; retain positive weights even at extreme sensitivity.
        weights = [math.exp(max(-700.0, self.rewire_sensitivity * (s - peak)))
                   for s in scores]
        total = sum(weights)
        return [w / total for w in weights]

    def _rewire_drop_probabilities(self, agent, neighbors):
        return self._rewire_softmax([-agent.get_trust(j) for j in neighbors])

    def _rewire_probabilities(self, agent, candidates):
        return self._rewire_softmax([
            agent.observe_reputation(self.social_agents[k]) for k in candidates])

    def rewire_network(self, rewire_prob):
        """One attempt per node: trust-based dissolution and reputation-based formation."""
        if not 0 <= rewire_prob <= 1:
            raise ValueError('rewire_prob must lie in [0, 1]')
        self.last_rewiring = dict(local_rewires=0, global_rewires=0,
                                  failed_rewire_attempts=0, isolated_rewire_attempts=0)
        if rewire_prob == 0:
            return 0
        rewired_edges = 0
        for agent in self.social_agents:
            if self.random.random() >= rewire_prob:
                continue
            neighbors = sorted(self.G.neighbors(agent.node_id))
            if not neighbors:
                # No owned edge to replace; recovery occurs through incoming discovery.
                self.last_rewiring['isolated_rewire_attempts'] += 1
                continue
            dropped = self.random.choices(
                neighbors, weights=self._rewire_drop_probabilities(agent, neighbors), k=1)[0]
            global_search = self.random.random() < self.global_search_prob
            candidates = self._rewire_candidates(agent.node_id, global_search)
            if not candidates and not global_search:
                global_search = True
                candidates = self._rewire_candidates(agent.node_id, global_search=True)
            if not candidates:
                self.last_rewiring['failed_rewire_attempts'] += 1
                continue
            chosen = self.random.choices(candidates,
                weights=self._rewire_probabilities(agent, candidates), k=1)[0]
            self.G.remove_edge(agent.node_id, dropped)
            self.G.add_edge(agent.node_id, chosen)
            self.last_rewiring['global_rewires' if global_search else 'local_rewires'] += 1
            rewired_edges += 1
        return rewired_edges

    def network_metrics(self):
        """Connectivity and concentration, with population SD and degree Gini."""
        degrees = sorted(d for _, d in self.G.degree())
        mean = sum(degrees) / self.n
        sd = math.sqrt(sum((d - mean)**2 for d in degrees) / self.n)
        gini = (2 * sum(i*d for i, d in enumerate(degrees, 1)) /
                (self.n * sum(degrees)) - (self.n + 1) / self.n) if mean else 0.0
        components = list(nx.connected_components(self.G))
        return {'min_degree': min(degrees), 'isolated_count': degrees.count(0),
                'degree_one_count': degrees.count(1), 'component_count': len(components),
                'largest_component_ratio': max(map(len, components)) / self.n,
                'mean_degree': mean, 'degree_std': sd,
                'degree_cv': sd / mean if mean else 0.0,
                'max_degree': max(degrees), 'degree_gini': max(0.0, gini),
                'max_degree_ratio': max(degrees) / (self.n - 1) if self.n > 1 else 0.0}
