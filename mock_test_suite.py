import unittest
import numpy as np
import cv2
import torch
from unittest.mock import MagicMock, patch, mock_open
import tempfile
import os

from Environment import HSREnvironment

class TestHSREnvironmentMocking(unittest.TestCase):
    """Tests for the HSREnvironment class with mocked image processing."""
    
    def setUp(self):
        """Set up test fixtures with mocked dependencies."""
        # Create patches for external dependencies
        self.mss_patcher = patch('mss.mss')
        self.pytesseract_patcher = patch('pytesseract.image_to_string')
        self.pyautogui_patcher = patch('pyautogui.press')
        self.pyautogui_click_patcher = patch('pyautogui.click')
        self.time_patcher = patch('time.sleep')
        self.cv2_imread_patcher = patch('cv2.imread')
        self.cv2_matchTemplate_patcher = patch('cv2.matchTemplate')
        self.cv2_minMaxLoc_patcher = patch('cv2.minMaxLoc')
        
        # Start patches
        self.mock_mss = self.mss_patcher.start()
        self.mock_pytesseract = self.pytesseract_patcher.start()
        self.mock_pyautogui_press = self.pyautogui_patcher.start()
        self.mock_pyautogui_click = self.pyautogui_click_patcher.start()
        self.mock_time_sleep = self.time_patcher.start()
        self.mock_cv2_imread = self.cv2_imread_patcher.start()
        self.mock_cv2_matchTemplate = self.cv2_matchTemplate_patcher.start()
        self.mock_cv2_minMaxLoc = self.cv2_minMaxLoc_patcher.start()
        
        # Configure mocks
        self.mock_mss.return_value.monitors = [None, {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}]
        self.mock_mss.return_value.grab.return_value = np.zeros((1080, 1920, 4), dtype=np.uint8)
        self.mock_pytesseract.return_value = "0"
        self.mock_cv2_imread.return_value = np.zeros((50, 50), dtype=np.uint8)
        self.mock_cv2_matchTemplate.return_value = np.zeros((50, 50), dtype=np.float32)
        self.mock_cv2_minMaxLoc.return_value = (0, 0.5, (0, 0), (25, 25))
        
        # Create environment with dependency injection
        self.env = HSREnvironment(monitor_number=1, debug=False)
        
    def tearDown(self):
        """Clean up after tests."""
        # Stop all patches
        self.mss_patcher.stop()
        self.pytesseract_patcher.stop()
        self.pyautogui_patcher.stop()
        self.pyautogui_click_patcher.stop()
        self.time_patcher.stop()
        self.cv2_imread_patcher.stop()
        self.cv2_matchTemplate_patcher.stop()
        self.cv2_minMaxLoc_patcher.stop()
        
    def test_template_match_with_mocks(self):
        """Test template matching with mocked OpenCV functions."""
        # Set up mock behavior for a match (above threshold)
        self.mock_cv2_minMaxLoc.return_value = (0, 0.9, (0, 0), (25, 25))
        
        # Call template_match
        template_config = [{'name': 'test_template', 'template_path': 'test.png', 'threshold': 0.8}]
        results, _ = self.env.template_match(template_config)
        
        # Check that template was detected
        self.assertTrue(results['test_template'])
        
        # Set up mock behavior for no match (below threshold)
        self.mock_cv2_minMaxLoc.return_value = (0, 0.7, (0, 0), (25, 25))
        
        # Call template_match again
        results, _ = self.env.template_match(template_config)
        
        # Check that template was not detected
        self.assertFalse(results['test_template'])
        
    def test_get_text_from_region_with_mocks(self):
        """Test text extraction with mocked pytesseract."""
        # Set up mock return value
        self.mock_pytesseract.return_value = "1234"
        
        # Call get_text_from_region
        region = (100, 100, 200, 50)
        text = self.env.get_text_from_region(region)
        
        # Check result
        self.assertEqual(text, "1234")
        
        # Verify pytesseract was called
        self.mock_pytesseract.assert_called_once()
        
    def test_execute_action_simple(self):
        """Test executing simple actions."""
        # Test pressing a single key
        self.env._execute_action('1')
        self.mock_pyautogui_press.assert_called_with('1')
        
        # Reset mock
        self.mock_pyautogui_press.reset_mock()
        
        # Test pressing multiple keys
        self.env._execute_action(['e', ('e', 1)])
        
        # Check that press was called twice
        self.assertEqual(self.mock_pyautogui_press.call_count, 2)
        
        # Check that sleep was called with correct delay
        self.mock_time_sleep.assert_called_with(1)
        
    def test_execute_reset_sequence(self):
        """Test the reset sequence."""
        # Execute reset sequence
        self.env._execute_reset_sequence()
        
        # Check that click was called once
        self.mock_pyautogui_click.assert_called_once()
        
        # Check that press was called multiple times
        self.assertTrue(self.mock_pyautogui_press.call_count > 0)
        
        # Check that sleep was called multiple times
        self.assertTrue(self.mock_time_sleep.call_count > 0)
        
    def test_calculate_reward_with_spacebar(self):
        """Test reward calculation for spacebar action."""
        # Mock state and info
        self.env.state = np.zeros(self.env.observation_space.shape)
        self.env.previous_state = np.zeros(self.env.observation_space.shape)
        self.env.current_action = 4  # Spacebar action
        
        # Mock template results
        mock_info = {
            'raw_templates': {'spacebar': True},
            'numeric_values': {
                'remaining_actions': 1500,
                'skill_points': 3,
                'lingsha_stacks': 2
            }
        }
        
        # Set up previous values
        self.env.prev_templates = {'spacebar': True}
        self.env.prev_numeric_values = {'remaining_actions': 2000}
        
        # Calculate reward
        reward = self.env._calculate_reward(self.env.state, mock_info)
        
        # Check reward is positive for pressing spacebar when needed
        self.assertTrue(reward > 0)
        
    def test_calculate_reward_with_character_ults(self):
        """Test reward calculation for character ultimates."""
        # Test Firefly ult
        self.env.state = np.zeros(self.env.observation_space.shape)
        self.env.previous_state = np.zeros(self.env.observation_space.shape)
        self.env.current_action = 0  # Firefly ult action
        
        # Mock templates with Firefly ult available
        mock_info = {
            'raw_templates': {'firefly_ult': False, 'fugue_0_stacks': False},
            'numeric_values': {
                'remaining_actions': 1500,
                'skill_points': 3,
                'lingsha_stacks': 2
            }
        }
        
        # Set up previous values with ult available
        self.env.prev_templates = {'firefly_ult': True, 'fugue_0_stacks': False}
        self.env.prev_numeric_values = {'remaining_actions': 2000}
        
        # Calculate reward
        reward = self.env._calculate_reward(self.env.state, mock_info)
        
        # Check reward is positive for using ult when available
        self.assertTrue(reward > 0)
        
        # Test using ult when unavailable
        self.env.prev_templates = {'firefly_ult': False}
        reward = self.env._calculate_reward(self.env.state, mock_info)
        
        # Check reward is negative for using ult when unavailable
        self.assertTrue(reward < 0)
        
    def test_is_done_detection(self):
        """Test detecting when the episode is done."""
        # Create a mock state
        mock_state = np.zeros(self.env.observation_space.shape)
        
        # Test with restart button not present
        mock_info = {'raw_templates': {'restart_button': False}}
        self.assertFalse(self.env._is_done(mock_state, mock_info))
        
        # Test with restart button present
        mock_info = {'raw_templates': {'restart_button': True}}
        self.assertTrue(self.env._is_done(mock_state, mock_info))


class TestEnvironmentEndToEnd(unittest.TestCase):
    """Integration tests for the HSR environment with mocked I/O."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create patches
        self.mss_patcher = patch('mss.mss')
        self.pyautogui_patcher = patch('pyautogui.press')
        self.pytesseract_patcher = patch('pytesseract.image_to_string')
        self.time_patcher = patch('time.sleep')
        
        # Start patches
        self.mock_mss = self.mss_patcher.start()
        self.mock_pyautogui = self.pyautogui_patcher.start()
        self.mock_pytesseract = self.pytesseract_patcher.start()
        self.mock_time = self.time_patcher.start()
        
        # Configure mock to return test values
        self.mock_mss.return_value.monitors = [None, {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}]
        self.mock_mss.return_value.grab.return_value = np.zeros((1080, 1920, 4), dtype=np.uint8)
        self.mock_pytesseract.return_value = "0"
        
        # Create a patched version of template matching
        mock_templates = [
            {'name': 'bug1'},
            {'name': 'bug2'},
            {'name': 'fugue_ult'},
            {'name': 'firefly_ult'},
            {'name': 'ruanmei_ult'},
            {'name': 'lingsha_ult'},
            {'name': 'fugue_0_stacks'},
            {'name': 'ruanmei_0_stacks'},
            {'name': 'firefly_e2'},
            {'name': 'fuyuan'},
            {'name': 'firefly_turn'},
            {'name': 'fugue_turn'},
            {'name': 'lingsha_turn'},
            {'name': 'ruanmei_turn'},
            {'name': 'restart_button'},
            {'name': 'firefly_lock'},
            {'name': 'spacebar'}
        ]

        self.patcher_template_match = patch.object(
            HSREnvironment, 'template_match', 
            return_value=({template['name']: False for template in mock_templates}, np.zeros((1080, 1920, 3)))
        )
        self.patcher_get_text = patch.object(
            HSREnvironment, 'get_text_from_region',
            return_value="0"
        )
        
        # Start the patchers
        self.mock_template_match = self.patcher_template_match.start()
        self.mock_get_text = self.patcher_get_text.start()
        
        # Create environment
        self.env = HSREnvironment(monitor_number=1, debug=False)
        
    def tearDown(self):
        """Clean up after tests."""
        self.mss_patcher.stop()
        self.pyautogui_patcher.stop()
        self.pytesseract_patcher.stop()
        self.time_patcher.stop()
        self.patcher_template_match.stop()
        self.patcher_get_text.stop()
        
    def test_reset_and_step_sequence(self):
        """Test a sequence of reset and steps."""
        # Mock the reset sequence and get state method
        with patch.object(self.env, '_execute_reset_sequence') as mock_reset:
            with patch.object(self.env, '_get_state') as mock_get_state:
                # Configure _get_state to return mock values
                mock_state = np.zeros(self.env.observation_space.shape, dtype=np.float32)
                mock_info = {
                    'raw_templates': {template['name']: False for template in self.env.templates},
                    'raw_text': {'remaining_action_value': '1000', 'skill_points': '3', 'lingsha_stacks': '2'},
                    'numeric_values': {'remaining_actions': 1000, 'skill_points': 3, 'lingsha_stacks': 2}
                }
                mock_get_state.return_value = (mock_state, mock_info)
                
                # Reset the environment
                state, info = self.env.reset()
                
                # Check that reset sequence was called
                mock_reset.assert_called_once()
                
                # Check that get_state was called
                mock_get_state.assert_called_once()
                
                # Check returned state and info
                np.testing.assert_array_equal(state, mock_state)
                self.assertEqual(info, mock_info)
                
                # Reset mock_get_state for step testing
                mock_get_state.reset_mock()
                
                # Configure _calculate_reward and _is_done
                with patch.object(self.env, '_calculate_reward', return_value=1.0) as mock_reward:
                    with patch.object(self.env, '_is_done', return_value=False) as mock_done:
                        with patch.object(self.env, '_execute_action') as mock_execute:
                            # Take a step
                            action = 0  # Firefly ult
                            next_state, reward, done, truncated, info = self.env.step(action)
                            
                            # Check that execute_action was called with correct key
                            mock_execute.assert_called_once_with(self.env.action_map[action])
                            
                            # Check that get_state was called again
                            mock_get_state.assert_called_once()
                            
                            # Check that calculate_reward and is_done were called
                            mock_reward.assert_called_once()
                            mock_done.assert_called_once()
                            
                            # Check returned values
                            np.testing.assert_array_equal(next_state, mock_state)
                            self.assertEqual(reward, 1.0)
                            self.assertFalse(done)
                            self.assertFalse(truncated)
                            self.assertEqual(info, mock_info)


class TestIntegrationWithDQN(unittest.TestCase):
    """Integration tests connecting the HSR environment with the DQN agent."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Patch dependencies to avoid actual screen interaction
        self.patcher_mss = patch('mss.mss')
        self.patcher_pyautogui = patch('pyautogui.press')
        self.patcher_pyautogui_click = patch('pyautogui.click')
        self.patcher_time = patch('time.sleep')
        self.patcher_cv2_imread = patch('cv2.imread')
        self.patcher_pytesseract = patch('pytesseract.image_to_string')
        
        # Start patches
        self.mock_mss = self.patcher_mss.start()
        self.mock_pyautogui = self.patcher_pyautogui.start()
        self.mock_pyautogui_click = self.patcher_pyautogui_click.start()
        self.mock_time = self.patcher_time.start()
        self.mock_cv2_imread = self.patcher_cv2_imread.start()
        self.mock_pytesseract = self.patcher_pytesseract.start()
        
        # Configure mocks
        self.mock_mss.return_value.monitors = [None, {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}]
        self.mock_mss.return_value.grab.return_value = np.zeros((1080, 1920, 4), dtype=np.uint8)
        self.mock_cv2_imread.return_value = np.zeros((50, 50), dtype=np.uint8)
        self.mock_pytesseract.return_value = "0"
        
        # Create mock environment methods
        self.env = HSREnvironment(monitor_number=1, debug=False)
        self.env.check_state = MagicMock(return_value={
            'templates': {template['name']: False for template in self.env.templates},
            'text': {
                'remaining_action_value': '1000',
                'skill_points': '3',
                'lingsha_stacks': '2'
            }
        })
        self.env._execute_reset_sequence = MagicMock()
        self.env._execute_action = MagicMock()
        self.env._calculate_reward = MagicMock(return_value=0.5)
        self.env._is_done = MagicMock(side_effect=[False, False, False, True])  # Done after 4 steps
        
        # Import DQNAgent here to avoid circular import in test setup
        from DQN import DQNAgent
        
        # Create agent
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.agent = DQNAgent(self.state_size, self.action_size, device="cpu")
        
    def tearDown(self):
        """Clean up after tests."""
        self.patcher_mss.stop()
        self.patcher_pyautogui.stop()
        self.patcher_pyautogui_click.stop()
        self.patcher_time.stop()
        self.patcher_cv2_imread.stop()
        self.patcher_pytesseract.stop()
        
    def test_training_loop(self):
        """Test a short training loop with the agent and environment."""
        # Patch the learn method to avoid actual learning
        with patch.object(self.agent, 'learn', return_value=0.1) as mock_learn:
            # Set up mock for select_action
            with patch.object(self.agent, 'select_action', side_effect=[0, 1, 2, 3]) as mock_select:
                # Run a short training loop (4 steps)
                state, _ = self.env.reset()
                total_reward = 0
                
                for _ in range(4):
                    action = self.agent.select_action(state)
                    next_state, reward, done, _, _ = self.env.step(action)
                    
                    # Store experience
                    self.agent.store_transition(state, action, reward, next_state, done)
                    
                    # Learn
                    if len(self.agent.memory) > self.agent.batch_size:
                        self.agent.learn()
                    
                    # Update tracking variables
                    state = next_state
                    total_reward += reward
                    
                    if done:
                        break
                
                # Check that methods were called correctly
                self.assertEqual(mock_select.call_count, 4)
                # Learn won't be called early in training since memory buffer isn't filled yet
                self.assertEqual(mock_learn.call_count, 0)
                
                # Check that step was called 4 times
                self.assertEqual(self.env._execute_action.call_count, 4)
                
                # Check that calculate_reward was called 4 times
                self.assertEqual(self.env._calculate_reward.call_count, 4)
                
                # Check that is_done was called 4 times
                self.assertEqual(self.env._is_done.call_count, 4)


if __name__ == '__main__':
    unittest.main()