import pyautogui
import time
import pytesseract
import cv2
import numpy as np
import mss
import gymnasium as gym
from gymnasium import spaces
import logging

class HSREnvironment(gym.Env):
    """
    A Gymnasium environment wrapper for your game with template matching and text recognition.
    """
    def __init__(self, monitor_number=1, debug=False, logger=None):
        super(HSREnvironment, self).__init__()
        
        # Set up logger
        self.logger = logger or logging.getLogger(__name__)
        # Screen capture setup
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[monitor_number]
        self.debug = debug

        self.first_reset = True
        
        # Define action and observation spaces
        # Actions: 1, 2, 3, 4, q, e, a, d (8 actions)
        self.action_space = spaces.Discrete(8)
        self.action_map = {
            0: '1', # Firefly Ult
            1: '2', # Fugue Ult
            2: '3', # Ruan Mei Ult
            3: '4', # Lingsha Ult
            4: 'space', # Basic attack 
            5: ['e', ('e', 1)], # Skill
            6: 'a', # Change to target to the Left
            7: 'd', # Change to target to the Right
        }
        
        # Observation space will be a flattened vector of:
        # - Template matching results (booleans converted to float)
        # - Text features extracted from specific regions
        self.templates = [
            {
                'name': 'bug1',
                'template_path': 'assets/Bug1.png',
                'region': (1812, 13, 400, 120),
                'threshold': 0.8
            },
            {
                'name': 'bug2',
                'template_path': 'assets/Bug2.png',
                'region': (1812, 13, 400, 120),
                'threshold': 0.8
            },
            {
                'name': 'fugue_ult',
                'template_path': 'assets/Fugue Ult.png',
                'region': (477, 1063, 300, 330),
                'threshold': 0.7
            },
            {
                'name': 'firefly_ult',
                'template_path': 'assets/Firefly Ult.png',
                'region': (265, 1087, 300, 330),
                'threshold': 0.8
            },
            {
                'name': 'ruanmei_ult',
                'template_path': 'assets/RM Ult.png',
                'region': (900, 1080, 300, 330),
                'threshold': 0.8
            },
            {
                'name': 'lingsha_ult',
                'template_path': 'assets/Lingsha Ult.png',
                'region': (1200, 1080, 300, 330),
                'threshold': 0.8
            },
            {
                'name': 'fugue_0_stacks',
                'template_path': 'assets/Fugue 0 Stacks.png',
                'region': (490, 1230, 220, 70),
                'threshold': 0.8
            },
            {
                'name': 'ruanmei_0_stacks',
                'template_path': 'assets/RM 0 Stacks.png',
                'region': (790, 1226, 220, 70),
                'threshold': 0.8
            },
            {
                'name': 'firefly_e2',
                'template_path': 'assets/Firefly E2.png',
                'region': (193, 1226, 100, 80),
                'threshold': 0.8
            },
            {
                'name': 'fuyuan',
                'template_path': 'assets/Fuyuan.png',
                'region': (59, 54, 200, 720),
                'threshold': 0.8
            },
            {
                'name': 'firefly_turn',
                'template_path': 'assets/Firefly Act.png',
                'region': (90, 45, 200, 150),
                'threshold': 0.8
            },
            {
                'name': 'fugue_turn',
                'template_path': 'assets/Fugue Act.png',
                'region': (90, 45, 200, 150),
                'threshold': 0.8
            },
            {
                'name': 'lingsha_turn',
                'template_path': 'assets/Lingsha Act.png',
                'region': (90, 45, 200, 150),
                'threshold': 0.8
            },
            {
                'name': 'ruanmei_turn',
                'template_path': 'assets/RM Act.png',
                'region': (90, 45, 200, 150),
                'threshold': 0.8
            },
            {
                'name': 'restart_button',
                'template_path': 'assets/Restart Button.png',
                'region': (676, 1217, 600, 130),
                'threshold': 0.8
            },
            {
                'name': 'firefly_lock',
                'template_path': 'assets/Firefly Lock.png',
                'region': (300, 1125, 200, 150),
                'threshold': 0.8
            },
            {
                'name': 'spacebar',
                'template_path': 'assets/Space.png',
                'region': (2320, 1040, 160, 80),
                'threshold': 0.8
            },
        ]
        
        self.text_regions = {
            'remaining_action_value': (2395, 387, 160, 70),
            'skill_points': (1860, 1264, 45, 65),
            'lingsha_stacks': (1158, 1247, 25, 30),
        }
        
        # Calculate number of features in our observation space
        self.num_template_features = len(self.templates)
        self.num_text_features = 3
        total_features = self.num_template_features + self.num_text_features
        
        # Define observation space as a Box with values between 0 and 1
        self.observation_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(total_features,), 
            dtype=np.float32
        )
        
        # Initialize state
        self.state = None
        self.previous_state = None
        self.previous_remaining_actions = None
        
    def reset(self, **kwargs):
        """Reset the environment and return the initial state."""
        self.previous_state = None
        self.previous_remaining_actions = None
        self._execute_reset_sequence()
        # Get initial state
        state, info = self._get_state()
        
        return state, info
    
    def step(self, action):
        """
        Take an action in the environment.
        
        Args:
            action: Integer from action space (0-8)
            
        Returns:
            next_state, reward, done, truncated, info
        """
        # Store previous state for reward calculation
        self.previous_state = self.state
        
        # Store the action for reward calculation
        self.current_action = action
        self.logger.debug(action)

        # Execute action (press key)
        key = self.action_map[action]
        self._execute_action(key)
        
        # Give the game time to respond
        time.sleep(0.1)
        
        # Get new state
        state, info = self._get_state()
        
        # Calculate reward
        reward = self._calculate_reward(state, info)
        
        # Check if episode is done (e.g., battle over)
        done = self._is_done(state, info)

        self.last_action = action
        
        # Truncated is False by default (not used in this environment)
        truncated = False
        
        return state, reward, done, truncated, info
    
    def _execute_reset_sequence(self):
        """Execute a sequence of key presses to reset the game."""
        self.logger.info("Executing reset sequence...")
        pyautogui.click(1117, 1262, duration=0.25)
        time.sleep(2)
        pyautogui.press('4')
        time.sleep(1)
        pyautogui.press('e')
        time.sleep(1)
        pyautogui.press('3')
        time.sleep(1)
        pyautogui.press('e')
        time.sleep(1)
        pyautogui.press('2')
        time.sleep(1)
        pyautogui.press('e')
        time.sleep(1)
        pyautogui.press('1')
        time.sleep(1)
        pyautogui.press('e')
        time.sleep(1)
        pyautogui.press('e')
        time.sleep(8) # Wait for game to run through actions
    
    def _execute_action(self, action_key):
        """Execute keyboard action, possibly involving multiple key presses."""
        if isinstance(action_key, str):
            # Simple single key action
            pyautogui.press(action_key)
        elif isinstance(action_key, list) or isinstance(action_key, tuple):
            # Sequence of keys with timing
            for i, key_step in enumerate(action_key):
                if isinstance(key_step, str):
                    # Simple key press
                    pyautogui.press(key_step)
                    
                    # Add delay if there's a next key press
                    if i < len(action_key) - 1:
                        # Check if next item is a tuple with delay information
                        next_step = action_key[i+1]
                        if isinstance(next_step, tuple) and len(next_step) == 2:
                            # Extract delay from the next tuple
                            delay = next_step[1]
                            time.sleep(delay)
                            
                            # Press the next key
                            pyautogui.press(next_step[0])
                            
                            # Skip the next item since we've handled it here
                            i += 1
    
    def _get_state(self):
        """Get the current state as a normalized vector."""
        results = self.check_state()
        
        # Extract template matching results
        template_features = np.zeros(self.num_template_features, dtype=np.float32)
        for i, template in enumerate(self.templates):
            template_name = template['name']
            template_features[i] = float(results['templates'].get(template_name, False))
        
        # Extract and normalize text features
        text_features = np.zeros(self.num_text_features, dtype=np.float32)

        # Process text
        remaining_actions_text = results['text'].get('remaining_action_value', '')
        try:
            # Extract numeric value from text
            remaining_actions = float(''.join(filter(str.isdigit, remaining_actions_text)))
            # Normalize to [0, 1] 
            text_features[0] = min(1.0, remaining_actions / 2000.0)
            
            # Store for reward calculation
            self.previous_remaining_actions = remaining_actions
        except (ValueError, TypeError):
            # Handle case where text couldn't be converted to number
            text_features[0] = 0.0
        
        skill_points_text = results['text'].get('skill_points', '')
        try:
            # Extract numeric value from text
            skill_points = float(''.join(filter(str.isdigit, skill_points_text)))
            # Normalize to [0, 1]
            text_features[1] = min(0.0, skill_points / 5.0)
            # Store for reward calculation
            self.previous_skill_points = skill_points
        except (ValueError, TypeError):
            # Handle case where text couldn't be converted to number
            text_features[1] = 0.0

        lingsha_stacks_text = results['text'].get('lingsha_stacks', '')
        try:
            lingsha_stacks = float(''.join(filter(str.isdigit,  lingsha_stacks_text)))
            text_features[2] = min(0.0, lingsha_stacks / 5.0)
            self.previous_lingsha_stacks = lingsha_stacks
        except (ValueError, TypeError):
            text_features[2] = 0.0

        # Combine features
        state = np.concatenate([template_features, text_features], dtype=np.float32)
        self.state = state
        
        # Additional info
        info = {
            'raw_templates': results['templates'],
            'raw_text': results['text'],
            'numeric_values': {
                'remaining_actions': remaining_actions if 'remaining_actions' in locals() else 0,
                'skill_points': skill_points if 'skill_points' in locals() else 0,
                'lingsha_stacks': lingsha_stacks if 'lingsha_stacks' in locals() else 0
            }
        }
        
        return state, info
    
    def _calculate_reward(self, state, info):
        """
        Calculate reward based on state changes.
        """
        reward = 0
        
        if self.previous_state is None or not hasattr(self, 'current_action'):
            return reward
        
        # Get template matching results
        templates = info['raw_templates']
        current_turn = None
        numeric_values = info.get('numeric_values', {})
        current_skill_points = numeric_values.get('skill_points', 0)
        remaining_action_value = numeric_values.get('remaining_actions', 0)
        lingsha_stacks = numeric_values.get('lingsha_stacks', 0)

        # Get the previous states
        previous_turn = getattr(self, 'previous_turn', None)
        prev_numeric_values = getattr(self, 'prev_numeric_values', {})
        prev_templates = getattr(self, 'prev_templates', {})
        prev_action_value = prev_numeric_values.get('remaining_actions', 0)

        # Check if both values are valid readings (not OCR errors). Any non 0 value is most likely not an error.
        is_action_value_valid = remaining_action_value > 0
        is_prev_action_value_valid = prev_action_value > 0
        if is_action_value_valid and is_prev_action_value_valid:
            action_value_change = prev_action_value - remaining_action_value 

            # Check if action value decreased 
            if action_value_change > 0:
                reward += 0.1
                self.logger.info("Reward: +0.1 for actions moving")


            # Try to get it to click the spacebar to advance the ults and prevent misinputs 
            if templates.get('spacebar', False) and self.current_action != 4:
                reward += -0.5 
                self.logger.info("Penalty: -0.5 Spacebar needs to be pressed to advance action first")

            if templates.get('spacebar', False) and self.current_action == 4 and prev_templates.get('spacebar', False):
                reward += 3 
                self.logger.info("Reward: +3 Spacebar pressed to advance action")
            elif self.current_action == 4 and prev_templates.get('spacebar', False):
                reward += 3 
                self.logger.info("Reward: +3 Spacebar pressed to advance action (2)")
                

                
        # Check whose turn it is
        for template_name, is_detected in templates.items():
            if "_turn" in template_name and is_detected:
                current_turn = template_name.replace("_turn", "")

        # ---------------------------------- Firefly-------------------------------------------

        # If Firefly is in ult give a reward for using skill as it has no cost thanks to E1 (Woo Eidolons!)
        if current_turn == "firefly" and self.current_action == 5 and templates.get('firefly_lock', False):
            reward += 4.5
            self.logger.info("Reward: +4.5 for using skill during Firefly's turn")

        # If it's Firefly's turn and E2 was available and is now used
        if current_turn == "firefly" and prev_templates.get("firefly_e2", False) and not templates.get("firefly_e2", False):
            reward += 1
            self.logger.info("Reward: +1 for triggering Firefly's E2")

        # If it's Firefly's turn and basic is used, and skill points is above 0
        if current_turn == "firefly" and self.current_action == 4 and current_skill_points > 0:
            reward += -3
            self.logger.info("Penalty: -3 for not using skill on Firefly's turn with skill points available")

        # Check if Firefly ult was used as an action, and if ult was available
        if self.current_action == 0 and prev_templates.get('firefly_ult', False):
            if templates.get('fugue_0_stacks', False):
                reward += 3
                self.logger.info("Reward: +3 for using Firefly Ult when available, but Fugue stacks are not active")
            else:
                reward += 6 
                self.logger.info("Reward: +6 for using Firefly Ult when available")

        # Check if Firefly ult was used as an action, and if ult was unavailable
        if self.current_action == 0 and not templates.get('firefly_ult', False) and not prev_templates.get('firefly_ult', False):
            reward += -3
            self.logger.info("Penalty: -3 for attempting to use Firefly Ult when unavailable")

        # ---------------------------------- Fugue -----------------------------------------------

        # Check if current turn is Fugue, Fugue has 0 stacks detected, and skill points > 0
        if (current_turn == "fugue" and templates.get('fugue_0_stacks', False) and self.current_action == 4 and current_skill_points > 0):
            # Apply negative reward
            reward += -1
            self.logger.info(f"Penalty: -1 for having unused skill points on Fugue's turn and she has no active stacks")

        # Check if Fugue ult was used as an action, and if ult was available
        if self.current_action == 1 and prev_templates.get('fugue_ult', False):
            reward += 2 
            self.logger.info("Reward: +2 for using Fugue Ult when available") # Though Fugue has a few situations where you could hold her ult, I've seen the situation maybe 3 times in 5 months, I think it's fine xdd

        # Check if Fugue ult was used as an action, and if ult was unavailable
        if self.current_action == 1 and not templates.get('fugue_ult', False) and not prev_templates.get('fugue_ult', False):
            reward += -2
            self.logger.info("Penalty: -2 for attempting to use Fugue Ult when unavailable")

        # Check if Fugue is the current character, skill is used, and her stacks are not empty
        if (current_turn == "fugue" and self.current_action == 5 and not templates.get('fugue_0_stacks', False)):
            reward -= 1
            self.logger.info("Penalty: -1 for using skill when Fugue's stacks are still up")
        # --------------------------------- Ruan Mei ------------------------------------------

        # Check if last turn was Ruan Mei, Ruan Mei has 0 stacks detected, and skill points > 0
        if current_turn == "ruanmei" and self.current_action == 4 and templates.get('ruanmei_0_stacks', False) and current_skill_points > 0:
            # Apply negative reward
            reward += -2
            self.logger.info(f"Penalty: -2 for having unused skill points after Ruan Mei's turn and she has no active stacks")

        # Check if Ruan Mei's ult was used as an action, and if ult was available
        if self.current_action == 2 and not templates.get('ruanmei_ult', False) and prev_templates.get('ruanmei_ult', False):
            reward += 1 # reward for using it as soon as it's gotten, Ruan mei struggles to get a two turn ult anyway, so it should definitely just ult when it get's it
            self.logger.info("Reward: +1 for using Ruan Mei Ult when available")

        # Check if Ruan Mei's ult was used as an action, and if ult was unavailable
        if self.current_action == 2 and not templates.get('ruanmei_ult', False) and not prev_templates.get('ruanmei_ult', False):
            reward += -2 
            self.logger.info("Penalty: -2 for attempting to use Ruan Mei Ult when unavailable")

        # Check if Ruan Mei is the current character, skill is used, and her stacks are not empty
        if (current_turn == "ruanmei" and self.current_action == 5 and not templates.get('ruanmei_0_stacks', False)):
            reward -= 1
            self.logger.info("Penalty: -1 for using skill when Ruan Mei's stacks are still up")

        # ---------------------------------- Lingsha ----------------------------------------------

        # Check if Lingsha's ult was used, and Fuyuan is not summoned
        if self.current_action == 3 and lingsha_stacks == 0 and prev_templates.get('lingsha_ult', False):
            reward += -3 
            self.logger.info("Penalty: -3 for using Lingsha Ult without Fuyuan summoned")

        # Check if Lingsha's ult was used, and Fuyuan is summoned
        if self.current_action == 3 and lingsha_stacks > 0 and not templates.get('lingsha_ult', False) and prev_templates.get('lingsha_ult', False):
            reward += +2 
            self.logger.info("Reward: +2 for using Lingsha Ult with Fuyuan summoned")

        # Check if last turn was Lingsha, the agent is using basic, Lingsha has no stacks, and skill points > 0
        if current_turn == "lingsha" and self.current_action == 4 and lingsha_stacks == 0 and current_skill_points > 0 and not prev_templates.get('spacebar', False):
            # Apply negative reward
            reward += -1
            self.logger.info(f"Penalty: -1 for having unused skill points after Lingsha's turn and Fuyuan is not on the field")

        # Check if Lingsha is the current character, skill is used, and the skill point count is above 2
        if (current_turn == "lingsha" and self.current_action == 5 and current_skill_points > 2):
            reward += 1
            self.logger.info("Reward: +1.0 for using Lingsha's skill with extra skill points available")

        # Check if Ruan Mei's ult was used as an action, and if ult was unavailable
        if self.current_action == 3 and not templates.get('lingsha_ult', False) and not prev_templates.get('lingsha_ult', False):
            reward += -2 
            self.logger.info("Penalty: -2 for attempting to use Lingsha Ult when unavailable")

        # Store current info for next comparison
        self.previous_turn = current_turn
        self.prev_templates = templates.copy()
        self.prev_numeric_values = numeric_values.copy()
        return reward
    
    def _is_done(self, state, info):
        """Check if the episode is done (e.g., battle over)."""
        # Checks specfically for the 'Restart Battle Button' which only appears on the end screen xdd hope this doesn't break
        if 'restart_button' in info['raw_templates'] and info['raw_templates']['restart_button']:
            return True
        
        # Add other conditions for episode completion
        return False
    
    def screen_cap(self):
        """Capture screenshot."""
        time.sleep(2) # Wait for the environment to stabilise with each picture
        screenshot = np.array(self.sct.grab(self.monitor))
        bgr_screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
        return bgr_screenshot
    
    def template_match(self, regions_templates, screenshot=None):
        """Check multiple templates in specific regions of the screen."""
        if screenshot is None:
            screenshot = self.screen_cap()
        results = {}

        grey_screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

        for config in regions_templates:
            name = config['name']
            template_path = config['template_path']
            region = config.get('region', None)
            threshold = config.get('threshold', 0.8)

            template = cv2.imread(template_path, 0)

            if template is None:
                self.logger.error(f"Warning: Could not load template {template_path}")
                results[name] = False
                continue
            
            h, w = template.shape

            if region is not None:
                x, y, width, height = region
                roi = grey_screenshot[y:y+height, x:x+width]
            else:
                roi = grey_screenshot
            
            if roi.shape[0] < h or roi.shape[1] < w:
                self.logger.error(f"Warning: Region too small for template {name}")
                results[name] = False
                continue

            try:
                result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                results[name] = max_val >= threshold

                if self.debug and results[name]:
                    img_debug = screenshot.copy()
                    if region is not None:
                        top_left = (max_loc[0] + x, max_loc[1]+ y)
                    else:
                        top_left = max_loc
                    
                    bottom_right = (top_left[0] + w, top_left[1] + h)
                    cv2.rectangle(img_debug, top_left, bottom_right, (0, 255, 0), 2)
                    cv2.putText(img_debug, f"{name}: {max_val:.2f}",
                                (top_left[0], top_left[1] - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    cv2.imshow('Template Match Debug', img_debug)
                    cv2.waitKey(100)

            except Exception as e:
                self.logger.error(f"Error matching template {name}: {e}")
                results[name] = False

        if self.debug:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return results, screenshot
    
    def get_text_from_region(self, region=None, screenshot=None):
        """Get text from a specific region using pytesseract."""
        if screenshot is None:
            screenshot = self.screen_cap()

        if region is not None:
            x, y, width, height = region
            region_img = screenshot[y:y+height, x:x+width]
        else:
            region_img = screenshot
        
        rgb_screenshot = cv2.cvtColor(region_img, cv2.COLOR_BGR2RGB) # Convert to RGB for pytesseract
        text = pytesseract.image_to_string(rgb_screenshot, config='--psm 8') # Using psm 8 to treat image as a single word
        return text
    
    def screen_check(self, template=None, text=None):
        """Check for templates and text in the current screen."""
        screenshot = self.screen_cap()
        result = {
            'templates': {},
            'text': {}
        }

        if template:
            template_results, _ = self.template_match(template, screenshot=screenshot)
            result['templates'] = template_results

        if text:
            for region_name, region_coords in text.items():
                result['text'][region_name] = self.get_text_from_region(
                    region=region_coords, 
                    screenshot=screenshot
                )
        
        return result

    def check_state(self):
        """Check game state by looking for all templates and text regions."""
        return self.screen_check(template=self.templates, text=self.text_regions)
    
    def render(self):
        """Render the environment for visualization. (Debug)"""
        if self.debug:
            screenshot = self.screen_cap()
            cv2.imshow('Game State', screenshot)
            cv2.waitKey(1)
    
    def close(self):
        """Close the environment."""
        if self.debug:
            cv2.destroyAllWindows()