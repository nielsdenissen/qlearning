import pandas as pd


class Qlearning:
    _qmatrix = None
    _learn_rate = None
    _discount_factor = None

    def __init__(self,
                 possible_states,
                 possible_actions,
                 initial_reward,
                 learning_rate,
                 discount_factor):
        """
        Initialise the q learning class with an initial matrix and the parameters for learning.

        :param possible_states: list of states the agent can be in
        :param possible_actions: list of actions the agent can perform
        :param initial_reward: the initial Q-values to be used in the matrix
        :param learning_rate: the learning rate used for Q-learning
        :param discount_factor: the discount factor used for Q-learning
        """
        # Initialize the matrix with Q-values
        init_data = [[initial_reward for _ in possible_states]
                     for _ in possible_actions]
        self._qmatrix = pd.DataFrame(data=init_data,
                                     index=possible_actions,
                                     columns=possible_states)

        # Save the parameters
        self._learn_rate = learning_rate
        self._discount_factor = discount_factor

    def get_best_action(self, state):
        """
        Retrieve the action resulting in the highest Q-value for a given state.

        :param state: the state for which to determine the best action
        :return: the best action from the given state
        """
        # Return the action (index) with maximum Q-value
        return self._qmatrix[[state]].idxmax().iloc[0]

    def update_model(self, state, action, reward, next_state):
        """
        Update the Q-values for a given observation.

        :param state: The state the observation started in
        :param action: The action taken from that state
        :param reward: The reward retrieved from taking action from state
        :param next_state: The resulting next state of taking action from state
        """
        # Update q_value for a state-action pair Q(s,a):
        # Q(s,a) = Q(s,a) + α( r + γmaxa' Q(s',a') - Q(s,a) )
        q_sa = self._qmatrix[state][action]
        max_q_sa_next = self._qmatrix[next_state][self.get_best_action(next_state)]
        r = reward
        alpha = self._learn_rate
        gamma = self._discount_factor

        # Do the computation
        self._qmatrix[state][action] = q_sa + alpha * (r + gamma * max_q_sa_next - q_sa)
