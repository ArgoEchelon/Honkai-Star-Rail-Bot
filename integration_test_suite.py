import unittest
import numpy as np
import torch
import os
import time
import tempfile
from unittest.mock import MagicMock, patch
import matplotlib.pyplot as plt

# Import your modules
from DQN import DQNAgent, ReplayBuffer, DQN
from Ikuso import HSREnvironment

class TestTrainingScript(unittest.TestCase):
    """Tests for the training script functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Patch dependencies to avoid actual screen interaction
        self.patcher_mss = patch('mss.mss')
        self.patcher_pyautogui = patch('pyautogui.press')
        self.patcher_pyautogui_click = patch('pyautogui.click')
        self.patcher_time = patch('time.sleep')
        self.patcher_cv2_imread = patch('cv2.imread')
        self.patcher_pytesseract = patch('pytesseract.image_to_string')
        self.patcher_plt_savefig = patch('matplotlib.pyplot.savefig')
        self.patcher_plt_close = patch('matplotlib.pyplot.close')
        
        # Start patches
        self.mock_mss = self.patcher_mss.start()
        self.mock_pyautogui = self.patcher_pyautogui.start()
        self.mock_pyautogui_click = self.patcher_pyautogui_click.start()
        self.mock_time = self.patcher_time.start()
        self.mock_cv2_imread = self.patcher_cv2_imread.start()
        self.mock_pytesseract = self.patcher_pytesseract.start()
        self.mock_plt_savefig = self.patcher_plt_savefig.start()
        self.mock_plt_close = self.patcher_plt_close.start()
        
        # Configure mocks
        self.mock_mss.return_value.monitors = [None, {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}]
        self.mock_mss.return_value.grab.return_value = np.zeros((1080, 1920, 4), dtype=np.uint8)
        self.mock_cv2_imread.return_value = np.zeros((50, 50), dtype=np.uint8)
        self.mock_pytesseract.return_value = "0"
        
        # Create environment
        self.env = HSREnvironment(monitor_number=1, debug=False)
        
        # Create mock state and info for reset
        mock_state = np.zeros(self.env.observation_space.shape, dtype=np.float32)
        mock_info = {
            'raw_templates': {template['name']: False for template in self.env.templates},
            'raw_text': {'remaining_action_value': '1000', 'skill_points': '3', 'lingsha_stacks': '2'},
            'numeric_values': {'remaining_actions': 1000, 'skill_points': 3, 'lingsha_stacks': 2}
        }
        
        # Important change: Mock the entire reset method, not just components
        self.env.reset = MagicMock(return_value=(mock_state, mock_info))
        self.env.step = MagicMock(return_value=(mock_state, 0.5, False, False, mock_info))
        
        # Create agent
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.agent = DQNAgent(self.state_size, self.action_size, device="cpu")
        
        # Import training function
        from xdding import train, plot_progress, evaluate
        self.train_func = train
        self.plot_progress_func = plot_progress
        self.evaluate_func = evaluate
        
    def tearDown(self):
        """Clean up after tests."""
        self.patcher_mss.stop()
        self.patcher_pyautogui.stop()
        self.patcher_pyautogui_click.stop()
        self.patcher_time.stop()
        self.patcher_cv2_imread.stop()
        self.patcher_pytesseract.stop()
        self.patcher_plt_savefig.stop()
        self.patcher_plt_close.stop()
        
    def test_train_function(self):
        """Test the train function."""
        # Create a temporary directory for saving models
        with tempfile.TemporaryDirectory() as temp_dir:
            # Patch agent methods to avoid actual learning
            with patch.object(self.agent, 'learn', return_value=0.1) as mock_learn:
                with patch.object(self.agent, 'select_action', return_value=0) as mock_select:
                    with patch.object(self.agent, 'save_model') as mock_save:
                        with patch.object(self.agent, 'update_target_network') as mock_update:
                            # Run training for 3 episodes with 5 steps max per episode
                            rewards, lengths, losses = self.train_func(
                                env=self.env,
                                agent=self.agent,
                                num_episodes=3,
                                max_steps=5,
                                render=False,
                                save_freq=1,
                                log_freq=1,
                                save_dir=temp_dir
                            )
                            
                            # Check that methods were called correctly
                            self.assertEqual(self.env.reset.call_count, 3)  # 3 episodes
                            
                            # Each episode has 5 steps, so 3*5=15 steps total
                            # But done is always False in our mock, so we'll hit max_steps each time
                            self.assertEqual(mock_select.call_count, 15)
                            
                            # Save model should be called at least 3 times (once per episode)
                            self.assertGreaterEqual(mock_save.call_count, 3)
                            
                            # Update target network at least once
                            self.assertGreaterEqual(mock_update.call_count, 1)
                            
                            # Check returned data
                            self.assertEqual(len(rewards), 3)
                            self.assertEqual(len(lengths), 3)
                            self.assertEqual(len(losses), 3)
                            
    def test_evaluate_function(self):
        """Test the evaluate function."""
        # Patch agent methods
        with patch.object(self.agent, 'select_action', return_value=0) as mock_select:
            # Run evaluation for 2 episodes with 5 steps max per episode
            rewards = self.evaluate_func(
                env=self.env,
                agent=self.agent,
                num_episodes=2,
                max_steps=5,
                render=False
            )
            
            # Check that methods were called correctly
            self.assertEqual(self.env.reset.call_count, 2)  # 2 episodes
            
            # select_action should be called 5 times per episode with evaluation=True
            self.assertEqual(mock_select.call_count, 10)
            
            # Check returned data
            self.assertEqual(len(rewards), 2)
            
    def test_plot_progress_function(self):
        """Test the plot_progress function."""
        # Create test data
        rewards = [1.0, 2.0, 3.0]
        lengths = [10, 20, 30]
        losses = [0.5, 0.3, 0.1]
        
        # Call plot_progress
        with tempfile.NamedTemporaryFile(suffix='.png') as tmp:
            self.plot_progress_func(rewards, lengths, losses, tmp.name)
            
            # Check that savefig was called
            self.mock_plt_savefig.assert_called_once_with(tmp.name)
            
            # Check that close was called
            self.mock_plt_close.assert_called_once()


class TestMemoryManagement(unittest.TestCase):
    """Tests for memory management and performance aspects."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a replay buffer
        self.buffer_size = 1000
        self.buffer = ReplayBuffer(self.buffer_size)
        
        # Create a state and action space for testing
        self.state_size = 20
        self.action_size = 8
        
    def test_buffer_memory_usage(self):
        """Test memory usage of the replay buffer."""
        # Add a bunch of experiences
        for i in range(500):
            state = np.random.rand(self.state_size).astype(np.float32)
            action = np.random.randint(0, self.action_size)
            reward = np.random.rand()
            next_state = np.random.rand(self.state_size).astype(np.float32)
            done = bool(np.random.randint(0, 2))
            
            self.buffer.add(state, action, reward, next_state, done)
            
        # Check buffer size
        self.assertEqual(len(self.buffer), 500)
        
        # Check that sampling works efficiently
        start_time = time.time()
        for _ in range(10):
            states, actions, rewards, next_states, dones = self.buffer.sample(64)
        sampling_time = time.time() - start_time
        
        # Sampling 10 batches should be quick (less than 1 second)
        self.assertLess(sampling_time, 1.0)
        
    def test_memory_leak(self):
        """Test for memory leaks in the buffer."""
        # Fill the buffer to capacity
        for i in range(self.buffer_size * 2):
            state = np.random.rand(self.state_size).astype(np.float32)
            action = np.random.randint(0, self.action_size)
            reward = np.random.rand()
            next_state = np.random.rand(self.state_size).astype(np.float32)
            done = bool(np.random.randint(0, 2))
            
            self.buffer.add(state, action, reward, next_state, done)
            
        # Buffer should respect its capacity limit
        self.assertEqual(len(self.buffer), self.buffer_size)
        
        # Sample from the buffer multiple times
        for _ in range(10):
            states, actions, rewards, next_states, dones = self.buffer.sample(64)
            
        # Add more experiences to test overwriting
        for i in range(100):
            state = np.random.rand(self.state_size).astype(np.float32)
            action = np.random.randint(0, self.action_size)
            reward = np.random.rand()
            next_state = np.random.rand(self.state_size).astype(np.float32)
            done = bool(np.random.randint(0, 2))
            
            self.buffer.add(state, action, reward, next_state, done)
            
        # Buffer size should still be at capacity
        self.assertEqual(len(self.buffer), self.buffer_size)


class TestAgentPerformance(unittest.TestCase):
    """Tests for agent learning performance."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple environment with predictable rewards
        self.state_size = 10
        self.action_size = 4
        
        # Create a DQN agent
        self.agent = DQNAgent(self.state_size, self.action_size, device="cpu")
        
        # Modify some hyperparameters for faster learning
        self.agent.learning_rate = 0.01
        self.agent.epsilon = 0.1
        self.agent.batch_size = 32
        
        # Create a simple buffer with predictable patterns
        self.buffer = self.agent.memory
        
    def test_learning_improvement(self):
        """Test that the agent improves its predictions over training."""
        # Create a simple pattern where action 2 is always best
        best_action = 2
        
        # Add experiences where the best action gives reward 1, others 0
        for i in range(1000):
            state = np.random.rand(self.state_size).astype(np.float32)
            for action in range(self.action_size):
                reward = 1.0 if action == best_action else 0.0
                next_state = np.random.rand(self.state_size).astype(np.float32)
                done = False
                
                self.buffer.add(state, action, reward, next_state, done)
        
        # Initial prediction
        test_state = np.random.rand(self.state_size).astype(np.float32)
        test_state_tensor = torch.FloatTensor(test_state).unsqueeze(0)
        
        initial_q_values = self.agent.policy_net(test_state_tensor).detach().numpy()
        
        # Train for some iterations
        for _ in range(100):
            self.agent.learn()
            
        # Final prediction
        final_q_values = self.agent.policy_net(test_state_tensor).detach().numpy()
        
        # Check that the best action's Q-value has increased more than others
        initial_best_q = initial_q_values[0][best_action]
        final_best_q = final_q_values[0][best_action]
        
        # Compute average change in other actions
        other_actions = [a for a in range(self.action_size) if a != best_action]
        initial_other_q = [initial_q_values[0][a] for a in other_actions]
        final_other_q = [final_q_values[0][a] for a in other_actions]
        
        avg_other_change = sum(f - i for f, i in zip(final_other_q, initial_other_q)) / len(other_actions)
        best_action_change = final_best_q - initial_best_q
        
        # Best action's Q-value should increase more than others
        self.assertGreater(best_action_change, avg_other_change)
        
        # Check if best action is now predicted
        final_best_action = np.argmax(final_q_values)
        self.assertEqual(final_best_action, best_action)
        
    def test_target_network_stability(self):
        """Test that the target network provides stability during learning."""
        # Fill buffer with some experiences
        for i in range(500):
            state = np.random.rand(self.state_size).astype(np.float32)
            action = np.random.randint(0, self.action_size)
            reward = np.random.rand()
            next_state = np.random.rand(self.state_size).astype(np.float32)
            done = False
            
            self.buffer.add(state, action, reward, next_state, done)
        
        # Get initial target network predictions
        test_state = np.random.rand(self.state_size).astype(np.float32)
        test_state_tensor = torch.FloatTensor(test_state).unsqueeze(0)
        
        initial_target_q = self.agent.target_net(test_state_tensor).detach().numpy()
        
        # Train policy network for several iterations
        for _ in range(50):
            self.agent.learn()
            
        # Target network should still have the same predictions
        current_target_q = self.agent.target_net(test_state_tensor).detach().numpy()
        np.testing.assert_array_equal(initial_target_q, current_target_q)
        
        # Update target network
        self.agent.update_target_network()
        
        # Now target network should match policy network
        updated_target_q = self.agent.target_net(test_state_tensor).detach().numpy()
        policy_q = self.agent.policy_net(test_state_tensor).detach().numpy()
        
        np.testing.assert_array_equal(updated_target_q, policy_q)


if __name__ == '__main__':
    unittest.main()