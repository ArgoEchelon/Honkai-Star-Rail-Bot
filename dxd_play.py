import torch
import time
import argparse

from Ikuso import HSREnvironment
from DQN import DQNAgent

def play(model_path, render=True, num_episodes=10, max_steps=1000, action_delay=0.1):
    """
    Play the game using a trained model.
    
    Args:
        model_path: Path to the trained model
        render: Whether to render the environment
        num_episodes: Number of episodes to play
        max_steps: Maximum steps per episode
        action_delay: Delay between actions in seconds
    """
    # Create the environment
    env = HSREnvironment()
    
    # Get state and action dimensions
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create agent
    agent = DQNAgent(state_size, action_size, device=device)
    
    # Load trained model
    print(f"Loading model from {model_path}")
    agent.load_model(model_path)
    print(f"Model loaded. Epsilon: {agent.epsilon}")
    
    # Set epsilon to minimum for deterministic actions
    # Comment this out if you want some exploration
    agent.epsilon = agent.epsilon_min
    
    for episode in range(num_episodes):
        print(f"Starting episode {episode+1}/{num_episodes}")
        
        # Reset environment
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # Select action (deterministic policy)
            action = agent.select_action(state, evaluation=True)
            action_name = env.action_map[action]
            
            print(f"Step {step+1}: Taking action {action} ({action_name})")
            
            # Take action
            next_state, reward, done, _, info = env.step(action)
            
            # Update state and accumulated reward
            state = next_state
            episode_reward += reward
            
            # Render if requested
            if render:
                env.render()
            
            # Add delay between actions for better visualization
            time.sleep(action_delay)
            
            # Print reward
            print(f"  Reward: {reward:.2f}, Cumulative: {episode_reward:.2f}")
            
            # Break if episode is done
            if done:
                print(f"Episode finished after {step+1} steps with reward {episode_reward:.2f}")
                break
        
        if step == max_steps - 1:
            print(f"Episode reached maximum steps ({max_steps}) with reward {episode_reward:.2f}")
    
    # Close the environment
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play HSR with a trained RL agent")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to play")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum steps per episode")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between actions in seconds")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    
    args = parser.parse_args()
    
    play(
        model_path=args.model,
        render=not args.no_render,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        action_delay=args.delay
    )