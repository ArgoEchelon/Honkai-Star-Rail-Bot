import torch
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from datetime import datetime

# Import our custom environment and agent
from Ikuso import HSREnvironment
from DQN import DQNAgent

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def train(env, agent, num_episodes=10, max_steps=1500, render=False, 
          save_freq=3, log_freq=1, save_dir="models"):
    """
    Train the DQN agent in the HSR environment.
    
    Args:
        env: The HSR environment
        agent: The DQN agent
        num_episodes: Maximum number of episodes to train
        max_steps: Maximum steps per episode
        render: Whether to render the environment
        save_freq: How often to save the model (in episodes)
        log_freq: How often to log results (in episodes)
        save_dir: Directory to save models
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Lists to track progress
    episode_rewards = []
    episode_lengths = []
    losses = []
    
    # Track best model
    best_reward = -float('inf')
    
    print("Starting training...")
    start_time = time.time()
    
    for episode in range(num_episodes):
        print("Test1")
        # Reset environment
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = []
        
        for step in range(max_steps):
            # Select action
            action = agent.select_action(state)
            
            # Take action
            next_state, reward, done, _, _ = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Learn
            if len(agent.memory) > agent.batch_size:
                loss = agent.learn()
                if loss is not None:
                    episode_loss.append(loss)
            
            # Update state and accumulated reward
            state = next_state
            episode_reward += reward
            
            # Render if requested
            if render:
                env.render()
            
            # Break if episode is done
            if done:
                break
        
        # Episode complete
        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
        
        # Calculate average loss for this episode
        avg_loss = np.mean(episode_loss) if episode_loss else 0
        losses.append(avg_loss)
        
        # Decay exploration rate
        agent.decay_epsilon()
        
        # Update target network periodically
        if episode % agent.update_target_freq == 0:
            agent.update_target_network()
        
        # Save model periodically
        if episode % save_freq == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(save_dir, f"hsr_dqn_episode_{episode}_{timestamp}.pt")
            agent.save_model(filename)
            
            # Plot training progress
            plot_progress(episode_rewards, episode_lengths, losses, 
                          os.path.join(save_dir, f"training_progress_{timestamp}.png"))
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save_model(os.path.join(save_dir, "best_model.pt"))
        
        # Log progress
        if episode % log_freq == 0:
            elapsed = time.time() - start_time
            print(f"Episode {episode}/{num_episodes} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Length: {step+1} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"Elapsed: {elapsed:.2f}s")
    
    # Training complete
    print(f"Training completed. Total time: {time.time() - start_time:.2f}s")
    
    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_dir, f"hsr_dqn_final_{timestamp}.pt")
    agent.save_model(filename)
    
    # Plot final progress
    plot_progress(episode_rewards, episode_lengths, losses, 
                  os.path.join(save_dir, f"final_progress_{timestamp}.png"))
    
    return episode_rewards, episode_lengths, losses

def plot_progress(rewards, lengths, losses, filename):
    """Plot and save training progress."""
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    
    # Plot rewards
    axs[0].plot(rewards)
    axs[0].set_title('Episode Rewards')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Reward')
    
    # Plot episode lengths
    axs[1].plot(lengths)
    axs[1].set_title('Episode Lengths')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Steps')
    
    # Plot losses
    axs[2].plot(losses)
    axs[2].set_title('Average Loss per Episode')
    axs[2].set_xlabel('Episode')
    axs[2].set_ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def evaluate(env, agent, num_episodes=10, max_steps=1000, render=True):
    """
    Evaluate the trained agent.
    
    Args:
        env: The HSR environment
        agent: The trained DQN agent
        num_episodes: Number of episodes to evaluate
        max_steps: Maximum steps per episode
        render: Whether to render the environment
    """
    rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # Select action (no exploration during evaluation)
            action = agent.select_action(state, evaluation=True)
            
            # Take action
            next_state, reward, done, _, _ = env.step(action)
            
            # Update state and accumulated reward
            state = next_state
            episode_reward += reward
            
            # Render if requested
            if render:
                env.render()
                time.sleep(0.1)  # Slow down rendering for better visualization
            
            # Break if episode is done
            if done:
                break
        
        rewards.append(episode_reward)
        print(f"Evaluation Episode {episode+1}/{num_episodes} | Reward: {episode_reward:.2f}")
    
    print(f"Average evaluation reward: {np.mean(rewards):.2f}")
    return rewards

def main():
    """Main function to set up and run the training."""
    # Create the environment
    env = HSREnvironment(monitor_number=1, debug=True)
    
    # Get state and action dimensions from the environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print(f"State size: {state_size}, Action size: {action_size}")
    
    # Create the agent
    agent = DQNAgent(state_size, action_size, device=device)
    
    # Training
    rewards, lengths, losses = train(
        env=env,
        agent=agent,
        num_episodes=10,  # Adjust based on your needs
        max_steps=1500,
        render=True,  # Set to False for faster training
        save_freq=1,
        log_freq=1,
        save_dir="models"
    )
    
    # Evaluate the trained agent
    eval_rewards = evaluate(
        env=env,
        agent=agent,
        num_episodes=5,
        max_steps=1000,
        render=True
    )
    
    # Close the environment
    env.close()
    
    return agent

if __name__ == "__main__":
    main()