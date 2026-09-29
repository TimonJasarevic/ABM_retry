import math

import mesa
if __package__:
    from .definitions import MessageState, Action
else:
    from definitions import MessageState, Action


class SocialAgent(mesa.Agent):
    # node_id: networkx node id
    # r: public source reliability in [0, 1]

    def __init__(self, model, node_id, is_influencer=False):
        super().__init__(model)

        # Inits
        self.node_id = node_id
        # Independent initial draws: receiver trait, public state, producer trait.
        self.default_trust = model.draw_baseline(model.mean_default_trust)
        self.initial_reputation = model.draw_baseline(model.mean_initial_reputation)
        self.r = self.initial_reputation  # dynamic public reputation; never reverts to its initial value
        self.baseline_truthfulness = model.draw_baseline(model.mean_truthfulness)
        self.pending_payoff = 0.0  # realised objective payoff within this cascade
        self.receiver_payoff = 0.0
        self.is_influencer = is_influencer

        # States
        self.message_state = MessageState.Unaware
        self.last_action = None

        # Parameters
        # Shared SD of perceptual error on the log-odds scale of public reputation.
        self.reputation_noise = model.reputation_noise
        # Model values are population means; these seeded Gamma draws are persistent.
        k = model.baseline_heterogeneity_concentration
        self.risk_aversion = (model.random.gammavariate(k, model.risk_aversion / k)
                              if model.risk_aversion > 0 else 0.0)
        self.loss_aversion = (1.0 + model.random.gammavariate(k, (model.loss_aversion - 1) / k)
                              if model.loss_aversion > 1 else 1.0)

        # Sender-specific trust:
        # Each receiver can trust specific neighbors differently.
        self.trust = {}                     # sender_id -> trust level
        self.trust_learning_rate = self.model.trust_learning_rate

        # Persistent observations from direct receivers of this agent's own claims.
        self.audience_response_score = model.audience_response_reference
        self.initial_fake_tendency = self.fake_tendency  # Snapshot before any learning.
        self.direct_share_count = 0
        self.direct_discard_count = 0
        self.direct_verification_count = 0
        self.direct_exposure_count = 0

        # Counters
        self.n_verify = 0
        self.n_share = 0
        self.n_discard = 0

        self.receiver_true_reward = model.receiver_true_reward
        # Objective cost of blind false propagation, separate from explicit detection.
        self.false_share_cost = model.false_share_cost
        self.sender_fake_penalty = model.sender_fake_penalty
        self.detect_fake_reward = model.detect_fake_reward

    @staticmethod
    def _sender_ids(senders):
        """Accept a single ID or a simultaneous set; count each transmitter once."""
        ids = (senders,) if isinstance(senders, int) else tuple(sorted(set(senders)))
        if not ids:
            raise ValueError('An exposure needs at least one sender')
        return ids

    def receive_message(self, message, sender_id, originator_id):
        """One receiver outcome; return each sender's fractional attribution.

        Verification privately informs this receiver only. Public evidence is
        queued and committed by the model after the entire propagation wave.
        """
        if self.message_state != MessageState.Unaware:
            return (0, {})
        senders = self._sender_ids(sender_id)
        action = self.choose_action(senders, originator_id)
        self._record_action(action)
        if action == Action.DISCARD:
            self.message_state = MessageState.Discarded
            return (0, {})
        if action == Action.VERIFY:
            self.model.total_verification_cost_paid += self.model.verify_cost
            for sender in senders:
                self._update_trust(sender, float(message))
            self.model.queue_public_verification(originator_id, message)
            if message:
                payoff = self.model.share_reward + self.receiver_true_reward - self.model.verify_cost
                self.message_state = MessageState.TrueBeliever
            else:
                payoff = self.detect_fake_reward - self.model.verify_cost
                self.message_state = MessageState.Corrected
        else:
            payoff = self.model.share_reward + (self.receiver_true_reward if message else -self.false_share_cost)
            self.message_state = MessageState.TrueBeliever if message else MessageState.FalseBeliever
        self.receiver_payoff += payoff
        self.pending_payoff += payoff
        forward = action == Action.SHARE or message
        if forward:
            # Role-specific reward divided by k: total is the mean role reward,
            # not the sum of k full rewards. Originators get only their own share.
            attribution = {j: self.model.propagation_reward(j, originator_id, message) / len(senders)
                           for j in senders}
        else:
            attribution = {j: -self.sender_fake_penalty / len(senders) for j in senders}
        return (int(forward), attribution)

    def update_reputation(self, observation):
        """One public record per verification; no payoff settlement or decay."""
        self.r += self.model.reputation_learning_rate * (observation - self.r)

    def get_trust(self, sender_id):
        """
        Return the receiver's trust in a specific sender.

        If the receiver has no previous verified history with this sender,
        use default_trust.
        """
        return self.trust.get(sender_id, self.default_trust)

    def _subjective_truthfulness(self, sender_id, originator_id):
        """Truth belief combines immediate-sender trust and original-source reputation."""
        senders = self._sender_ids(sender_id)
        effective_trust = math.fsum(self.get_trust(j) for j in senders) / len(senders)
        return (self.model.trust_weight * effective_trust
                + self.model.reputation_weight * self.observe_reputation(
                    self.model.social_agents[originator_id]))

    def _update_trust(self, sender_id, observation):
        """
        Update trust in this specific sender after verification.

        observation = 1.0 means the sender sent true news.
        observation = 0.0 means the sender sent fake news.
        """
        if self.trust_learning_rate == 0:
            return
        current_trust = self.get_trust(sender_id)

        updated_trust = current_trust + self.trust_learning_rate * (
            observation - current_trust
        )

        self.trust[sender_id] = min(1.0, max(0.0, updated_trust))

    @property
    def direct_share_rate(self):
        return self.direct_share_count / self.direct_exposure_count if self.direct_exposure_count else float('nan')

    @property
    def direct_verification_rate(self):
        return self.direct_verification_count / self.direct_exposure_count if self.direct_exposure_count else float('nan')

    @property
    def direct_discard_rate(self):
        return self.direct_discard_count / self.direct_exposure_count if self.direct_exposure_count else float('nan')

    @property
    def fake_tendency(self):
        """Heterogeneous baseline shifted by recent direct audience response."""
        baseline = 1.0 - self.baseline_truthfulness
        if baseline in (0.0, 1.0):
            return baseline
        shift = self.model.audience_response_sensitivity * (
            self.audience_response_score - self.model.audience_response_reference)
        if shift == 0:
            return baseline
        logit = math.log(baseline) - math.log1p(-baseline) + shift
        if logit >= 0:
            return 1.0 / (1.0 + math.exp(-logit))
        value = math.exp(logit)
        return value / (1.0 + value)

    def observe_direct_audience(self, shares, verifications, discards):
        """Learn once from direct receiver actions, not transmitting-edge counts."""
        if any(not isinstance(n, int) or n < 0 for n in (shares, verifications, discards)):
            raise ValueError('direct action counts must be nonnegative integers')
        self.direct_share_count += shares
        self.direct_verification_count += verifications
        self.direct_discard_count += discards
        exposures = shares + verifications + discards
        self.direct_exposure_count += exposures
        if exposures:
            # One aggregate observation per originating cascade, independent of receiver order.
            response = (shares - verifications) / exposures
            self.audience_response_score += self.model.audience_response_learning_rate * (
                response - self.audience_response_score)

    def _outcome_utility(self, p_true, x_true, x_fake):
        # Loss weighting applies to NET outcomes before mean and variance.
        v_true = x_true if x_true >= 0 else self.loss_aversion * x_true
        v_fake = x_fake if x_fake >= 0 else self.loss_aversion * x_fake
        mean = p_true * v_true + (1 - p_true) * v_fake
        variance = p_true * (v_true - mean)**2 + (1 - p_true) * (v_fake - mean)**2
        return mean - self.risk_aversion * variance

    def action_utilities(self, sender_id, originator_id):
        # Hypothetical net outcomes are transformed for choice only, never settlement.
        p_true = self._subjective_truthfulness(sender_id, originator_id)
        return {
            Action.VERIFY: self._outcome_utility(
                p_true, self.model.share_reward + self.receiver_true_reward - self.model.verify_cost,
                self.detect_fake_reward - self.model.verify_cost),
            Action.SHARE: self._outcome_utility(
                p_true, self.model.share_reward + self.receiver_true_reward,
                self.model.share_reward - self.false_share_cost),
            Action.DISCARD: 0.0,
        }

    def action_probabilities(self, sender_id, originator_id):
        utilities = self.action_utilities(sender_id, originator_id)
        peak = max(utilities.values())
        weights = {action: math.exp(self.model.rationality * (u - peak))
                   for action, u in utilities.items()}
        total = sum(weights.values())
        return {action: weight / total for action, weight in weights.items()}

    def choose_action(self, sender_id, originator_id):
        probabilities = self.action_probabilities(sender_id, originator_id)
        return self.model.random.choices(
            list(probabilities), weights=list(probabilities.values()), k=1)[0]

    def _record_action(self, action):
        self.last_action = action

        if action == Action.VERIFY:
            self.n_verify += 1
        elif action == Action.SHARE:
            self.n_share += 1
        elif action == Action.DISCARD:
            self.n_discard += 1

    def initiate_message(self):
        # message is true with probability 1 - fake_tendency.
        # fake_tendency reflects the prior and recent direct-audience response.
        message_is_fake = self.model.random.random() < self.fake_tendency
        message = not message_is_fake

        if message:
            self.message_state = MessageState.TrueBeliever
        else:
            self.message_state = MessageState.FalseBeliever

        return message

    def update(self):
        """Clear temporary payoff accounting; reputation is evidence-driven."""
        self.pending_payoff = 0.0
        self.receiver_payoff = 0.0

    def observe_reputation(self, agent):
        """Gaussian perceptual error with shared SD on reputation's log-odds scale."""
        if self.reputation_noise == 0:
            return agent.r
        eps = 1e-12
        r = min(1.0 - eps, max(eps, agent.r))
        logit_r = math.log(r) - math.log1p(-r)
        perceived_logit = logit_r + self.model.random.gauss(0.0, self.reputation_noise)
        if perceived_logit >= 0:
            perceived = 1.0 / (1.0 + math.exp(-perceived_logit))
        else:
            exp_value = math.exp(perceived_logit)
            perceived = exp_value / (1.0 + exp_value)
        # The exact sigmoid is strictly interior. Only repair floating-point
        # saturation at extreme logits; no clipping of ordinary noisy scores.
        if perceived == 1.0:
            return math.nextafter(1.0, 0.0)
        if perceived == 0.0:
            return math.nextafter(0.0, 1.0)
        return perceived
