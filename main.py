import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from time import sleep

class GridWorld:
    """
    Custom Grid World Environment
    """
    def __init__(self, width=5, height=5, obstacles=None, goal=None, start=None):
        self.width = width
        self.height = height
        
        # Define grid elements
        self.obstacles = obstacles if obstacles is not None else [(1, 1), (2, 2), (3, 3)]
        self.goal = goal if goal is not None else (4, 4)
        self.start = start if start is not None else (0, 0)
        
        # Define actions: 0=up, 1=right, 2=down, 3=left
        self.actions = [0, 1, 2, 3]
        self.action_names = ['Up', 'Right', 'Down', 'Left']
        
        # Define rewards
        self.goal_reward = 10
        self.obstacle_reward = -10
        self.step_reward = -1
        
        # Initialize agent position
        self.agent_pos = list(self.start)
        
    def reset(self):
        """Reset the environment to the start state"""
        self.agent_pos = list(self.start)
        return tuple(self.agent_pos)
    
    def step(self, action):
        """
        Take a step in the environment
        Returns: next_state, reward, done
        """
        # Store current position
        prev_pos = self.agent_pos.copy()
        
        # Move agent based on action
        if action == 0:  # Up
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 1:  # Right
            self.agent_pos[0] = min(self.width - 1, self.agent_pos[0] + 1)
        elif action == 2:  # Down
            self.agent_pos[1] = min(self.height - 1, self.agent_pos[1] + 1)
        elif action == 3:  # Left
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        
        # Check if agent hit an obstacle
        if tuple(self.agent_pos) in self.obstacles:
            reward = self.obstacle_reward
            done = True
            self.agent_pos = prev_pos  # Stay in previous position
        # Check if agent reached the goal
        elif tuple(self.agent_pos) == self.goal:
            reward = self.goal_reward
            done = True
        else:
            reward = self.step_reward
            done = False
        
        return tuple(self.agent_pos), reward, done
    
    def render(self):
        """Visualize the grid world"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Create grid
        for x in range(self.width + 1):
            ax.axvline(x, color='black', linewidth=1)
        for y in range(self.height + 1):
            ax.axhline(y, color='black', linewidth=1)
        
        # Set limits and aspect ratio
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        
        # Add obstacles
        for obs in self.obstacles:
            rect = patches.Rectangle(obs, 1, 1, linewidth=1, edgecolor='black', 
                                   facecolor='red', alpha=0.7)
            ax.add_patch(rect)
        
        # Add goal
        goal_rect = patches.Rectangle(self.goal, 1, 1, linewidth=1, edgecolor='black', 
                                    facecolor='green', alpha=0.7)
        ax.add_patch(goal_rect)
        
        # Add agent
        agent_circle = plt.Circle((self.agent_pos[0] + 0.5, self.agent_pos[1] + 0.5), 
                                0.3, color='blue')
        ax.add_patch(agent_circle)
        
        # Add start position
        start_rect = patches.Rectangle(self.start, 1, 1, linewidth=1, edgecolor='black', 
                                     facecolor='yellow', alpha=0.3)
        ax.add_patch(start_rect)
        
        # Add labels
        ax.text(self.agent_pos[0] + 0.5, self.agent_pos[1] + 0.5, 'A', 
               ha='center', va='center', fontsize=12, fontweight='bold')
        ax.text(self.goal[0] + 0.5, self.goal[1] + 0.5, 'G', 
               ha='center', va='center', fontsize=12, fontweight='bold')
        
        ax.set_xticks(np.arange(0, self.width + 1, 1))
        ax.set_yticks(np.arange(0, self.height + 1, 1))
        ax.set_title('Grid World Environment')
        ax.grid(True)
        
        plt.gca().invert_yaxis()  # To have (0,0) at top-left
        plt.show()
    
    def get_state_index(self, state):
        """Convert state coordinates to a unique index"""
        return state[0] * self.height + state[1]
    
    def get_state_from_index(self, index):
        """Convert index back to state coordinates"""
        x = index // self.height
        y = index % self.height
        return (x, y)

class QLearningAgent:
    """
    Q-Learning Agent
    """
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, 
                 exploration_decay=0.995, min_exploration_rate=0.01):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        
        # Initialize Q-table with zeros
        self.num_states = env.width * env.height
        self.num_actions = len(env.actions)
        self.q_table = np.zeros((self.num_states, self.num_actions))
    
    def choose_action(self, state):
        """Choose an action using epsilon-greedy policy"""
        state_idx = self.env.get_state_index(state)
        
        # Exploration: choose random action
        if np.random.uniform(0, 1) < self.exploration_rate:
            return random.choice(self.env.actions)
        # Exploitation: choose best action
        else:
            return np.argmax(self.q_table[state_idx])
    
    def update_q_table(self, state, action, reward, next_state):
        """Update Q-table using Q-learning update rule"""
        state_idx = self.env.get_state_index(state)
        next_state_idx = self.env.get_state_index(next_state)
        action_idx = action
        
        # Q-learning update
        best_next_action = np.max(self.q_table[next_state_idx])
        td_target = reward + self.discount_factor * best_next_action
        td_error = td_target - self.q_table[state_idx, action_idx]
        self.q_table[state_idx, action_idx] += self.learning_rate * td_error
    
    def decay_exploration(self):
        """Decay exploration rate"""
        self.exploration_rate = max(self.min_exploration_rate, 
                                   self.exploration_rate * self.exploration_decay)

def train_agent(env, agent, episodes=1000, max_steps=100, verbose=True):
    """
    Train the Q-learning agent
    """
    episode_rewards = []
    episode_steps = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(max_steps):
            # Choose and take action
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            
            # Update Q-table
            agent.update_q_table(state, action, reward, next_state)
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        # Decay exploration rate
        agent.decay_exploration()
        
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        
        if verbose and (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}: Steps = {steps}, Reward = {total_reward}, "
                  f"Exploration = {agent.exploration_rate:.3f}")
    
    return episode_rewards, episode_steps

def visualize_learning(episode_rewards, episode_steps):
    """Plot learning progress"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot rewards
    ax1.plot(episode_rewards)
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True)
    
    # Plot steps
    ax2.plot(episode_steps)
    ax2.set_title('Steps per Episode')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def get_optimal_path(env, agent, max_steps=20):
    """Find the optimal path using the learned Q-table"""
    state = env.reset()
    path = [state]
    total_reward = 0
    
    for step in range(max_steps):
        state_idx = env.get_state_index(state)
        action = np.argmax(agent.q_table[state_idx])  # Greedy action
        next_state, reward, done = env.step(action)
        
        path.append(next_state)
        total_reward += reward
        state = next_state
        
        if done:
            break
    
    return path, total_reward

def visualize_optimal_path(env, agent):
    """Visualize the optimal path found by the agent"""
    path, total_reward = get_optimal_path(env, agent)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create grid
    for x in range(env.width + 1):
        ax.axvline(x, color='black', linewidth=1)
    for y in range(env.height + 1):
        ax.axhline(y, color='black', linewidth=1)
    
    # Set limits and aspect ratio
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect('equal')
    
    # Add obstacles
    for obs in env.obstacles:
        rect = patches.Rectangle(obs, 1, 1, linewidth=1, edgecolor='black', 
                               facecolor='red', alpha=0.7)
        ax.add_patch(rect)
    
    # Add goal
    goal_rect = patches.Rectangle(env.goal, 1, 1, linewidth=1, edgecolor='black', 
                                facecolor='green', alpha=0.7)
    ax.add_patch(goal_rect)
    
    # Add start position
    start_rect = patches.Rectangle(env.start, 1, 1, linewidth=1, edgecolor='black', 
                                 facecolor='yellow', alpha=0.3)
    ax.add_patch(start_rect)
    
    # Draw path
    for i in range(len(path) - 1):
        start = (path[i][0] + 0.5, path[i][1] + 0.5)
        end = (path[i+1][0] + 0.5, path[i+1][1] + 0.5)
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    
    # Add labels
    ax.text(env.start[0] + 0.5, env.start[1] + 0.5, 'S', 
           ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(env.goal[0] + 0.5, env.goal[1] + 0.5, 'G', 
           ha='center', va='center', fontsize=12, fontweight='bold')
    
    ax.set_xticks(np.arange(0, env.width + 1, 1))
    ax.set_yticks(np.arange(0, env.height + 1, 1))
    ax.set_title(f'Optimal Path (Total Reward: {total_reward})')
    ax.grid(True)
    
    plt.gca().invert_yaxis()  # To have (0,0) at top-left
    plt.show()
    
    return path

def print_q_table(env, agent):
    """Print the learned Q-table in a readable format"""
    print("\nLearned Q-Table:")
    print("State\t\tUp\t\tRight\t\tDown\t\tLeft")
    print("-" * 70)
    
    for state_idx in range(agent.num_states):
        state = env.get_state_from_index(state_idx)
        q_values = agent.q_table[state_idx]
        
        print(f"{state}", end="\t\t")
        for action in range(agent.num_actions):
            print(f"{q_values[action]:.2f}", end="\t\t")
        print()

# Main execution
if __name__ == "__main__":
    # Create environment
    print("Creating Grid World Environment...")
    env = GridWorld(width=5, height=5, 
                   obstacles=[(1, 1), (2, 2), (3, 1), (1, 3)], 
                   goal=(4, 4), start=(0, 0))
    
    # Display initial environment
    print("Initial Grid World:")
    env.render()
    
    # Create Q-learning agent
    print("\nInitializing Q-Learning Agent...")
    agent = QLearningAgent(env, learning_rate=0.1, discount_factor=0.9, 
                          exploration_rate=1.0, exploration_decay=0.995)
    
    # Train the agent
    print("\nTraining Agent...")
    episode_rewards, episode_steps = train_agent(env, agent, episodes=1000, max_steps=50)
    
    # Visualize learning progress
    print("\nVisualizing Learning Progress...")
    visualize_learning(episode_rewards, episode_steps)
    
    # Display learned Q-table
    print_q_table(env, agent)
    
    # Show optimal path
    print("\nVisualizing Optimal Path...")
    optimal_path = visualize_optimal_path(env, agent)
    
    print(f"\nOptimal Path: {optimal_path}")
    print(f"Path Length: {len(optimal_path)} steps")
    
    # Test the trained agent
    print("\nTesting Trained Agent...")
    test_path, test_reward = get_optimal_path(env, agent)
    print(f"Test Path: {test_path}")
    print(f"Test Reward: {test_reward}")