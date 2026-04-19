'''
@qdd
LinUCB_Disjoint_Hetero.py: the features are different for each user and each arm
LinUCB_Disjoint_Homo.py: the features are different for each user

'''


import numpy as np
import matplotlib.pyplot as plt

class LinUCB:
    def __init__(self, n_arms, n_features, alpha):
        self.n_arms = n_arms
        self.n_features = n_features
        self.alpha = alpha
        self.A = [np.identity(n_features) for _ in range(n_arms)] # The inverse of A can be considered as the covariance of the thetas
        self.b = [np.zeros(n_features) for _ in range(n_arms)] # b is an intermediary object for you to resolve theta later. Please note that this b is the same as the b in Algorithm 1 on page 4 of the paper "A contextual-bandit approach to personalized news article recommendation". Please also note that in this paper, the right panel of page 3, there is a b_a in line 3 (which means the response vector corresponding to m users.)  These two b are different, as the one on page 4 has d dimensions (which is the number of feature dimensions), whereas the b on page 3 has m dimensions (which is the number of data points, i.e., number of visitors). Actually, the b_a on page 3 should be c_a, which represents the response vector, as shown in equation (3) on the same page.

    def update(self, arm, context, reward):
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context

    def recommend(self, context):
        scores = np.zeros(self.n_arms)
        thetas = [np.zeros(self.n_features) for _ in range(self.n_arms)]
        for arm in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[arm])
            theta = A_inv @ self.b[arm]
            p_arm = theta.T @ context[arm] + self.alpha * np.sqrt(context[arm].T @ A_inv @ context[arm])
            scores[arm] = p_arm
            thetas[arm] = theta
        return np.argmax(scores), thetas, scores


# The simulator is to generate some offline data so that I can test how well LinUCB is performing. In live enviroment (i.e., deploying linucb on some ecommerce website for recommendation), the data is not given ahead, but shows up lively.

class Simulator:
    def __init__(self, n_trials, n_arms, n_features, good_arms, good_bias):
        self.n_trials = n_trials # In the example of news recommendation, there is one user showing up in each trial
        self.n_arms = n_arms
        self.n_features = n_features
        self.good_arms = good_arms
        self.good_bias = good_bias
        self.X = [np.zeros(n_features) for _ in range(n_trials)]

    # X is a contextual matrix, with the size of n_trials * n_arms * n_features. This is to make sure that at a particular step, I can retrieve the feature vecors for the current user directly by indexing user_id. 
    def simulate_context_matrix(self, n_trials, n_arms, n_features):
        X = np.array([[np.random.uniform(low=0, high = 1, size=n_features) for _ in np.arange(n_arms)] for _ in np.arange(n_trials)])  ## If you use this line, it means that the user has different features when facing different arms. In this case, when you retrieve the context values, you should use both the user_id and the item_id to index. This means that the features can be user-dependent, or item-dependent, or user-item-dependent. 
        # X = np.array([np.random.uniform(low=0, high = 1, size=n_features) for _ in np.arange(n_trials)]) # Of course, you can allow each user to have the same features across different arms.
        self.X = X
        return X

    # true_theta are something you don't know in reality. But in simulations, we assume that we know ahead, and see if the bandit algorithm can estimate thetas well (because reward is the linear function of features expressed by theta values.)
    def simulate_theta(self, n_arms, n_features, good_arms, good_bias) :
        true_theta = np.array([np.random.normal(loc = 0, size = n_features, scale = 1/4) for _ in np.arange(n_arms)])
        if len(good_arms)>0:
            true_theta[good_arms] = true_theta[good_arms] + good_bias
        self.true_theta = true_theta
        return true_theta

    def initialization(self):
        self.simulate_context_matrix(self.n_trials, self.n_arms, self.n_features)
        self.simulate_theta(self.n_arms, self.n_features, self.good_arms,self.good_bias)


# This is to predict the reward of exposing a particular arm to a particular user with the feature values x. As you can see here x is lowercase, becase it only consists of n_features feature values for a particular user. Then the reward should be predicted as a linear sum of x, plus some error noise. 
def simulate_one_time_reward(arm, x, theta, scale_noise = 0.01):
    signal = theta @ x
    noise  = np.random.normal(scale = scale_noise) 
    return (signal + noise)

## Start running over 1000 trials
my_n_trials = 5000
my_n_arms = 4
my_n_features = 5
good_arms = [2]
good_bias = 1
my_alpha = 0.5

np.random.seed(4014)

results = dict()
my_simulation = Simulator(my_n_trials, my_n_arms, my_n_features, good_arms, good_bias)
my_simulation.initialization()

my_linucb = LinUCB(my_n_arms, my_n_features, my_alpha)

# theta_history is to let me know the estimations for theta values over time
theta_history  = np.empty(shape=(my_n_trials, my_n_arms, my_n_features))
# score_history is to let me know the estimations for the upper bound of each arm
score_history  = np.empty(shape=(my_n_trials, my_n_arms))
# arm_selection_history and r_payoff are to let me know the arms I have chosen and the rewards I have received over time
arm_selection_history, r_payoff = [np.empty(my_n_trials) for _ in range(2)]

for t in np.arange(my_n_trials):
    arm_selected,theta_history[t],score_history[t]  = my_linucb.recommend(my_simulation.X[t])
    # at each time step, when I select one arm, then I should observe some reward. In reality, this reward will be shown as whether user clicks or not. But in my simulations, I generated a reward (which is based on my true model used in simulations, i.e., true_theta)
    reward_observed = simulate_one_time_reward(arm=arm_selected, x=my_simulation.X[t][arm_selected], theta=my_simulation.true_theta[arm_selected])
    my_linucb.update(arm_selected,my_simulation.X[t][arm_selected],reward_observed)
    r_payoff[t] = reward_observed
    arm_selection_history[t] = arm_selected

results = dict(theta_history=theta_history, upper_bound_score_history=score_history, arm_selection_history=arm_selection_history, r_payoff_history=r_payoff)

## Hope all these comments help you to better understand LinUCB. I will leave the reward plots to you guys. I believe you can do it!!

