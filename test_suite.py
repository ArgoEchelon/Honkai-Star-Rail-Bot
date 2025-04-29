import unittest
import numpy as np
import torch
import os
import tempfile
from unittest.mock import MagicMock, patch

# Import your modules
from DQN import DQNAgent, ReplayBuffer, DQN
from Ikuso import HSREnvironment

class TestReplayBuffer(unittest.TestCase):
    """Tests for the ReplayBuffer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.capacity = 100
        self.buffer = ReplayBuffer(self.capacity)
        
    def test_initialization(self):
        """Test that the buffer initializes correctly."""
        self.assertEqual(len(self.buffer), 0)
        self.assertEqual(self.buffer.buffer.maxlen, self.capacity)
        
    def test_add_experience(self):
        """Test adding experiences to the buffer."""
        # Add a single experience
        state = np.array([1, 2, 3, 4])
        action = 1
        reward = 0.5
        next_state = np.array([2, 3, 4, 5])
        done = False
        
        self.buffer.add(state, action, reward, next_state, done)
        self.assertEqual(len(self.buffer), 1)
        
        # Check that the experience was added correctly
        added_exp = self.buffer.buffer[0]
        np.testing.assert_array_equal(added_exp[0], state)
        self.assertEqual(added_exp[1], action)
        self.assertEqual(added_exp[2], reward)
        np.testing.assert_array_equal(added_exp[3], next_state)
        self.assertEqual(added_exp[4], done)
        
    def test_sample_batch(self):
        """Test sampling a batch of experiences."""
        # Add multiple experiences
        for i in range(10):
            state = np.array([i, i+1, i+2, i+3])
            action = i % 4
            reward = i * 0.1
            next_state = np.array([i+1, i+2, i+3, i+4])
            done = i % 3 == 0
            
            self.buffer.add(state, action, reward, next_state, done)
            
        # Sample a batch
        batch_size = 5
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        
        # Check shapes
        self.assertEqual(states.shape, (batch_size, 4))
        self.assertEqual(actions.shape, (batch_size,))
        self.assertEqual(rewards.shape, (batch_size,))
        self.assertEqual(next_states.shape, (batch_size, 4))
        self.assertEqual(dones.shape, (batch_size,))
        
        # Check types
        self.assertTrue(isinstance(states, torch.FloatTensor))
        self.assertTrue(isinstance(actions, torch.LongTensor))
        self.assertTrue(isinstance(rewards, torch.FloatTensor))
        self.assertTrue(isinstance(next_states, torch.FloatTensor))
        self.assertTrue(isinstance(dones, torch.FloatTensor))
        
    def test_capacity_limit(self):
        """Test that the buffer respects its capacity limit."""
        # Fill the buffer beyond capacity
        for i in range(self.capacity + 50):
            state = np.array([i, i+1, i+2, i+3])
            action = i % 4
            reward = i * 0.1
            next_state = np.array([i+1, i+2, i+3, i+4])
            done = i % 3 == 0
            
            self.buffer.add(state, action, reward, next_state, done)
            
        # Check that the buffer length is capped at capacity
        self.assertEqual(len(self.buffer), self.capacity)


class TestDQN(unittest.TestCase):
    """Tests for the DQN neural network."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.state_size = 10
        self.action_size = 4
        self.hidden_size = 64
        self.network = DQN(self.state_size, self.action_size, self.hidden_size)
        
    def test_initialization(self):
        """Test that the network initializes with correct shapes."""
        # Check input layer
        self.assertEqual(
            self.network.network[0].in_features, 
            self.state_size
        )
        self.assertEqual(
            self.network.network[0].out_features, 
            self.hidden_size
        )
        
        # Check hidden layer
        self.assertEqual(
            self.network.network[2].in_features, 
            self.hidden_size
        )
        self.assertEqual(
            self.network.network[2].out_features, 
            self.hidden_size
        )
        
        # Check output layer
        self.assertEqual(
            self.network.network[4].in_features, 
            self.hidden_size
        )
        self.assertEqual(
            self.network.network[4].out_features, 
            self.action_size
        )
        
    def test_forward_pass(self):
        """Test forward pass through the network."""
        batch_size = 5
        x = torch.randn(batch_size, self.state_size)
        
        # Forward pass
        output = self.network(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, self.action_size))


class TestDQNAgent(unittest.TestCase):
    """Tests for the DQNAgent class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.state_size = 20
        self.action_size = 8
        self.device = "cpu"
        self.agent = DQNAgent(self.state_size, self.action_size, self.device)
        
    def test_initialization(self):
        """Test that the agent initializes correctly."""
        # Check basic attributes
        self.assertEqual(self.agent.state_size, self.state_size)
        self.assertEqual(self.agent.action_size, self.action_size)
        self.assertEqual(self.agent.device, self.device)
        
        # Check networks
        self.assertTrue(isinstance(self.agent.policy_net, DQN))
        self.assertTrue(isinstance(self.agent.target_net, DQN))
        
        # Check optimizer
        self.assertTrue(isinstance(self.agent.optimizer, torch.optim.Adam))
        
        # Check replay buffer
        self.assertTrue(isinstance(self.agent.memory, ReplayBuffer))
        self.assertEqual(self.agent.memory.buffer.maxlen, self.agent.buffer_size)
        
    def test_select_action_exploration(self):
        """Test action selection during exploration."""
        # Force exploration by setting epsilon to 1
        self.agent.epsilon = 1.0
        
        # Mock the random choice to ensure it's called
        with patch('random.randrange', return_value=3) as mock_random:
            state = np.zeros(self.state_size)
            action = self.agent.select_action(state)
            
            # Check that random.randrange was called with action_size
            mock_random.assert_called_once_with(self.action_size)
            self.assertEqual(action, 3)
            
    def test_select_action_exploitation(self):
        """Test action selection during exploitation."""
        # Force exploitation by setting epsilon to 0
        self.agent.epsilon = 0.0
        
        # Create a state that would result in a predictable action
        state = np.zeros(self.state_size)
        
        # Mock the policy_net to return a predictable Q-value
        mock_q_values = torch.tensor([[-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0]])
        with patch.object(self.agent.policy_net, 'forward', return_value=mock_q_values):
            action = self.agent.select_action(state)
            
            # The agent should choose the action with the highest Q-value (index 7)
            self.assertEqual(action, 7)
            
    def test_select_action_evaluation(self):
        """Test action selection during evaluation (no exploration)."""
        # Even with high epsilon, evaluation mode should disable exploration
        self.agent.epsilon = 1.0
        
        # Create a state that would result in a predictable action
        state = np.zeros(self.state_size)
        
        # Mock the policy_net to return a predictable Q-value
        mock_q_values = torch.tensor([[-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0]])
        with patch.object(self.agent.policy_net, 'forward', return_value=mock_q_values):
            action = self.agent.select_action(state, evaluation=True)
            
            # The agent should choose the action with the highest Q-value (index 7)
            self.assertEqual(action, 7)
            
    def test_learn(self):
        """Test the learning process."""
        # Fill the replay buffer with enough samples
        for i in range(self.agent.batch_size):
            state = np.random.rand(self.state_size)
            next_state = np.random.rand(self.state_size)
            action = np.random.randint(0, self.action_size)
            reward = np.random.rand()
            done = bool(np.random.randint(0, 2))
            
            self.agent.store_transition(state, action, reward, next_state, done)
            
        # Mock the optimizer to check if step is called
        with patch.object(self.agent.optimizer, 'step') as mock_step:
            loss = self.agent.learn()
            
            # Check that optimizer.step was called
            mock_step.assert_called_once()
            
            # Check that loss is a float
            self.assertTrue(isinstance(loss, float))
            
    def test_update_target_network(self):
        """Test updating the target network."""
        # Modify the policy network weights
        for param in self.agent.policy_net.parameters():
            param.data = torch.randn_like(param.data)
            
        # Check that the networks have different parameters
        for policy_param, target_param in zip(self.agent.policy_net.parameters(), 
                                              self.agent.target_net.parameters()):
            self.assertFalse(torch.all(policy_param.data == target_param.data))
            
        # Update target network
        self.agent.update_target_network()
        
        # Check that the networks now have the same parameters
        for policy_param, target_param in zip(self.agent.policy_net.parameters(), 
                                              self.agent.target_net.parameters()):
            self.assertTrue(torch.all(policy_param.data == target_param.data))
            
    def test_decay_epsilon(self):
        """Test epsilon decay."""
        initial_epsilon = self.agent.epsilon
        
        # Decay epsilon
        self.agent.decay_epsilon()
        
        # Check that epsilon decreased
        self.assertTrue(self.agent.epsilon < initial_epsilon)
        
        # Check that epsilon doesn't go below minimum
        self.agent.epsilon = self.agent.epsilon_min
        self.agent.decay_epsilon()
        self.assertEqual(self.agent.epsilon, self.agent.epsilon_min)
        
    def test_save_load_model(self):
        """Test saving and loading the model."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            filename = tmp.name
            
        try:
            # Save the model
            self.agent.save_model(filename)
            
            # Create a new agent
            new_agent = DQNAgent(self.state_size, self.action_size, self.device)
            
            # Verify that the agents have different parameters
            for p1, p2 in zip(self.agent.policy_net.parameters(), 
                              new_agent.policy_net.parameters()):
                self.assertFalse(torch.all(p1.data == p2.data))
            
            # Load the model
            new_agent.load_model(filename)
            
            # Verify that the agents now have the same parameters
            for p1, p2 in zip(self.agent.policy_net.parameters(), 
                              new_agent.policy_net.parameters()):
                self.assertTrue(torch.all(p1.data == p2.data))
                
        finally:
            # Clean up
            os.remove(filename)


class TestHSREnvironment(unittest.TestCase):
    """Tests for the HSREnvironment class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock environment since we can't actually do screen capture in tests
        with patch('mss.mss'):
            with patch('pytesseract.image_to_string'):
                self.env = HSREnvironment(debug=False)
                
        # Mock the screen capture method
        self.env.screen_cap = MagicMock(return_value=np.zeros((1080, 1920, 3), dtype=np.uint8))
        
        # Mock template matching
        self.env.template_match = MagicMock(return_value=(
            {template['name']: False for template in self.env.templates},
            np.zeros((1080, 1920, 3), dtype=np.uint8)
        ))
        
        # Mock text recognition
        self.env.get_text_from_region = MagicMock(return_value="0")
        
    def test_initialization(self):
        """Test that the environment initializes correctly."""
        # Check action space
        self.assertEqual(self.env.action_space.n, 8)
        
        # Check observation space
        expected_features = len(self.env.templates) + self.env.num_text_features
        self.assertEqual(self.env.observation_space.shape, (expected_features,))
        
    def test_reset(self):
        """Test environment reset."""
        # Mock _execute_reset_sequence
        self.env._execute_reset_sequence = MagicMock()
        
        # Mock _get_state
        mock_state = np.zeros(self.env.observation_space.shape, dtype=np.float32)
        mock_info = {"test": "info"}
        self.env._get_state = MagicMock(return_value=(mock_state, mock_info))
        
        # Reset the environment
        state, info = self.env.reset()
        
        # Check that the reset sequence was executed
        self.env._execute_reset_sequence.assert_called_once()
        
        # Check that _get_state was called
        self.env._get_state.assert_called_once()
        
        # Check return values
        np.testing.assert_array_equal(state, mock_state)
        self.assertEqual(info, mock_info)
        
    def test_step(self):
        """Test environment step."""
        # Mock _execute_action
        self.env._execute_action = MagicMock()
        
        # Mock _get_state
        mock_state = np.zeros(self.env.observation_space.shape, dtype=np.float32)
        mock_info = {"test": "info"}
        self.env._get_state = MagicMock(return_value=(mock_state, mock_info))
        
        # Mock _calculate_reward
        mock_reward = 1.0
        self.env._calculate_reward = MagicMock(return_value=mock_reward)
        
        # Mock _is_done
        mock_done = False
        self.env._is_done = MagicMock(return_value=mock_done)
        
        # Initialize previous state
        self.env.state = np.zeros(self.env.observation_space.shape, dtype=np.float32)
        
        # Take a step
        action = 0
        next_state, reward, done, truncated, info = self.env.step(action)
        
        # Check that _execute_action was called with the correct key
        self.env._execute_action.assert_called_once_with(self.env.action_map[action])
        
        # Check that _get_state was called
        self.env._get_state.assert_called_once()
        
        # Check that _calculate_reward was called with the correct arguments
        self.env._calculate_reward.assert_called_once_with(mock_state, mock_info)
        
        # Check that _is_done was called with the correct arguments
        self.env._is_done.assert_called_once_with(mock_state, mock_info)
        
        # Check return values
        np.testing.assert_array_equal(next_state, mock_state)
        self.assertEqual(reward, mock_reward)
        self.assertEqual(done, mock_done)
        self.assertFalse(truncated)
        self.assertEqual(info, mock_info)
        
    def test_get_state(self):
        """Test getting the current state."""
        # Mock check_state
        mock_results = {
            'templates': {template['name']: False for template in self.env.templates},
            'text': {
                'remaining_action_value': '1000',
                'skill_points': '3',
                'lingsha_stacks': '2'
            }
        }
        self.env.check_state = MagicMock(return_value=mock_results)
        
        # Get state
        state, info = self.env._get_state()
        
        # Check that check_state was called
        self.env.check_state.assert_called_once()
        
        # Check state shape
        self.assertEqual(state.shape, self.env.observation_space.shape)
        
        # Check state values (all templates are False, so their values should be 0)
        template_features = state[:self.env.num_template_features]
        np.testing.assert_array_equal(template_features, np.zeros(self.env.num_template_features))
        
        # Check text features
        text_features = state[self.env.num_template_features:]
        expected_text_features = np.array([
            1000 / 2000.0,  # remaining_action_value normalized
            0.0,  # skill_points normalized (but capped at 0.0 as per your code)
            0.0   # lingsha_stacks normalized (but capped at 0.0 as per your code)
        ], dtype=np.float32)
        np.testing.assert_array_almost_equal(text_features, expected_text_features)
        
        # Check info
        self.assertEqual(info['raw_templates'], mock_results['templates'])
        self.assertEqual(info['raw_text'], mock_results['text'])
        self.assertEqual(info['numeric_values']['remaining_actions'], 1000)
        self.assertEqual(info['numeric_values']['skill_points'], 3)
        self.assertEqual(info['numeric_values']['lingsha_stacks'], 2)

    def test_is_done(self):
        """Test the done condition."""
        # Create mock state and info
        mock_state = np.zeros(self.env.observation_space.shape, dtype=np.float32)
        
        # Test when restart_button is not detected
        mock_info = {
            'raw_templates': {'restart_button': False}
        }
        self.assertFalse(self.env._is_done(mock_state, mock_info))
        
        # Test when restart_button is detected
        mock_info['raw_templates']['restart_button'] = True
        self.assertTrue(self.env._is_done(mock_state, mock_info))


if __name__ == '__main__':
    unittest.main()