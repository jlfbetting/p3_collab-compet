# Report

## Implementation 1
### Learning algorithm.

I tried several algorithms. Since each agent gets its own, local state, and have a common goal of keeping the ball in game for as long as possible, I figured that I could use th
DDPG agent as I developed for Project 2 of this nanodegree. So, I copied my own code from [this repository](https://github.com/jlfbetting/p2_continuous-control) (details about my implementation can be found there) and edited the code in the
Jupyter Notebook in such a way that for each timestep, the DDPG agent first chooses an action for the first player, and then for the second player, after which the environment progresses. In other words, the agent plays the game with itself.
To my surprise, I did not even have to change the hyperparameters in order to reach the rubric goal of 0.5.

The architecture of the actor networks is as follows:
* input layer size = 33 (state space).
* 2 hidden layers of each 200 hidden units, with `ReLU` activation layers, and a batch normalization layer after the first hidden layer.
* output layer size = 4 (the action space), with a tanh activation layer.

The architecture of the critic networks is as follows:
* input layer size = 4 (action space).
* 2 hidden layers of each 200 hidden units, with `ReLU` activation layers, and a batch normalization layer after the first hidden layer.
* output layer size = 1 (no activation layer).

The use of batch normalization layers was also described in the paper by Lillicrap et al. (2016, section 3). Gradient clipping was added (as given as a hint on the Udacity benchmark description page). I also made sure that the local networks get the same weights as the corresponding target networks at initialization. Two hyperparameters, `learn_every` and `learn_times`, were added to the original code. After every `learn_every` steps, both networks are trained `learn_times` times, each time with a different memory sample. The hyperparameter values were kept the same as in by submission of P2:

* `buffer_size` = 1e5       (replay buffer size)
* `batch_size` = 256        (minibatch size)
* `gamma` = 0.95            (discount factor)
* `tau` = 5e-2              (for soft update of target parameters)
* `lr_actor` = 1e-4         (learning rate of the actor)
* `lr_critic` = 1e-4        (learning rate of the critic)
* `weight_decay` = 0        (L2 weight decay)
* `learn_every` = 100       (learn after this number of steps)
* `learn_times` = 40        (number of sampling and training per cycle)

I also used a `random_seed` value of 10 in the notebook, which turned out to work well.

### Plot of rewards
 The plot below shows how the score changes as more episodes are simulated. On average, the score goes up. The environment was solved in 1261 episodes, as can be seen in the notebook. The actor_local network gave an average score over 100 episodes of 0.504
 ![Episode-score plot](https://github.com/jlfbetting/p3_collab-compet/blob/main/imp1.png)
 
## Implementation 2

### Learning algorithm.
I wanted to see if I could improve performance by flattening the state and action spaces and feeding that to the agent. Whereas in implementation 1, the agent alternatingly
plays player 1 and player 2, this implementation would be analogous to one agent playing both sides at the same time. I gave the critic network an extra parameter `reward_units` to accommodate multiple rewards.
The advantage of such an implementation could be that the two players can adapt to each other's behaviour easily (because the actions are determined at the same time). The disadvantage would be that the state and action spaces become twice larger,
so learning might be tougher. Furthermore, because the position of the ball is described in the states of both players, there is some redundancy. I changed the implementation of the critic class to allow two reward signals (since we have two players).

### Plot of rewards
 The plot below shows how the score changes as more episodes are simulated. On average, the score goes up. The environment was solved in 2467 episodes, as can be seen in the notebook. The actor_local network gave an average score over 100 episodes of 0.505
 ![Episode-score plot](https://github.com/jlfbetting/p3_collab-compet/blob/main/imp2.png)
  
 ### Ideas for future work
* We could experiment with adding more elements to the neural networks, such as dropout layers.
* We could change implementation 2 further by converting the concatenated localized (subjective) states into an objective grid, with the absolute positions of both players and the ball.
* We could use Prioritized Experience Replay (as introduced in the lesson on Deep Q-Networks, and in [this paper](https://arxiv.org/abs/1511.05952)). Instead of sampling the experiences linearly, we could prioritize the more important experiences (i.e. those experiences from which the network can learn the most) so that training goes more efficiently.


