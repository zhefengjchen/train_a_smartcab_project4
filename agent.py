import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import namedtuple

# create namedtuple to conert inputs dict to namedtuple which is hashable
LearningAgentState = namedtuple('LearningAgentState', ['light', 'oncoming', 'left', 'right', 'next_waypoint'])


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        # initialize the next waypoint by randomly assigning a valid action to it
        self.next_waypoint = random.choice(Environment.valid_actions[1:])

        #initialize Q-Learning table
        self.Q_table = {}     # Q value table
        self.gamma = 0.1        # discount factor of maxQ(s',a')
        self.alpha = 0.9        # learning rate
        self.epsilon = 0      # probability of doing random move for epsilon-greedy algorithm

        self.success = 0

        #statistics
        self.dest = 0         # number of times that agent get to the destination
        self.moves = 0        # total number of moves at each run
        self.positive = 0     # total number of correct moves
        self.negative = 0     # total number of incorrect moves

        self.steps = 0.0           # For each run, number of moves to get to the destination
        self.penalty_per_run = 0.0    # For each run, number of wrong moves

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        if self.steps > 0:
            print "At each run, penalty ratio is: {}/{}={}%".format(self.penalty_per_run,self.steps,(round(self.penalty_per_run/self.steps,4))*100)
            self.steps = 0.0
            self.penalty_per_run = 0.0

    def update(self, t):
        #number of moves
        self.moves += 1
        #number of moves each run
        self.steps += 1

        # initialize gamma and alpha
        self.gamma = 0.1
        self.alpha = 0.9
        self.epsilon = 0.6/float(t+1)

        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        deadline = self.env.get_deadline(self)
        inputs = self.env.sense(self)

        # TODO: Update state
        self.state = LearningAgentState(light = inputs['light'], oncoming = inputs['oncoming'], left = inputs['left'], right = inputs['right'],next_waypoint=self.next_waypoint)


        # TODO: Select action according to your policy
        if self.state not in self.Q_table.keys():
            self.Q_table[self.state] = {None:0.0, 'forward':0.0, 'left':0.0, 'right':0.0}
            action = random.choice(Environment.valid_actions[:])
        else:
            action = None
            Q_value = 0
            for k, v in self.Q_table[self.state].iteritems():
                if v > Q_value:
                    Q_value = v
                    action = k
            if random.random() < self.epsilon:
                # print "---------------------------------------------------"
                action = random.choice(Environment.valid_actions[:])

        # TODO: Learn policy based on state, action, reward
        # Execute action and get reward
        reward = self.env.act(self, action)

        if reward >=10:
            self.dest += 1
            self.positive += 1
        else:
            if reward >= 0:
                self.positive += 1
            else:
                self.negative += 1
                self.penalty_per_run += 1


        #sense the new state environment
        new_inputs = self.env.sense(self)
        self.next_waypoint = self.planner.next_waypoint()
        new_state = LearningAgentState(light=new_inputs['light'], oncoming=new_inputs['oncoming'], left=new_inputs['left'],right=new_inputs['right'],next_waypoint=self.next_waypoint)

        #Find the optimal Q_hat(s',a')
        if new_state not in self.Q_table.keys():
            Q_hat_s_prime = 0

        else:
            Q_hat_s_prime = max(self.Q_table[new_state].values())

        #Calcaule Q_hat(s,a) and Q(s,a)
        Q_hat_s = reward + self.gamma * Q_hat_s_prime
        self.Q_table[self.state][action] = (1 - self.alpha) * self.Q_table[self.state][action] + self.alpha * Q_hat_s

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    #statistics
    num_sim = 10
    total_destination = 0.0
    total_positive = 0.0
    total_negative = 0.0
    total_move = 0.0


    for x in xrange(num_sim):
        # Set up environment and agent
        e = Environment()  # create environment (also adds some dummy traffic)
        a = e.create_agent(LearningAgent)  # create agent
        e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
        # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

        # Now simulate it
        sim = Simulator(e, update_delay=0,display=False)  # create simulator (uses pygame when display=True, if available)
        # NOTE: To speed up simulation, reduce update_delay and/or set display=False

        sim.run(n_trials=100)  # run for a specified number of trials
        # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
        total_destination += a.dest
        total_move += a.moves
        total_positive += a.positive
        total_negative += a.negative

    print "average successful rate: ", float(total_destination/num_sim)
    print "average penalty ratio: {}/{}={}%".format(total_negative,total_move,(round(total_negative/total_move,4))*100)

if __name__ == '__main__':
    run()
