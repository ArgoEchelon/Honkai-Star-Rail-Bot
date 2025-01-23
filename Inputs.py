import pyautogui
import time
import numpy as np
import cv2
import mss
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

stdDelay = 0.9
def wait():
    time.sleep(stdDelay)

class HSREnvironment:
    def __init__(self, monitor_number=1):
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[monitor_number]

    def take_action(self, action): # 0 - Basic Attack, 1 - Skill, 2 - Char 1 Ult, 3 - Char 2 Ult, 4 - Char 3 Ult, 5 - Char 4 Ult, 6 - Select left, 7 - Select right
        action_map = {
            0: 'q', 1: 'e', 2: '1', 3: '2', 4: '3', 5: '4', 
            6: 'a', 7: 'd', 8: 'space'
        }
        if action in action_map:
            pyautogui.press(action_map[action])
            time.sleep(stdDelay)

    def check_pixel_color(self, x, y, target_color, threshold=60):
        # Check if pixel at (x,y) matches target_color within threshold
        screenshot = np.array(self.sct.grab(self.monitor))
        pixel_color = screenshot[y][x][:3]  # Get RGB values
        
        # Check if color matches within threshold
        matches = all(
            abs(pc - tc) <= threshold 
            for pc, tc in zip(pixel_color, target_color)
        )
        return matches

    def get_screen_state(self):
        # Get current screen state including pixel color checks
        screenshot = np.array(self.sct.grab(self.monitor))

        state = {
            'ult_1_ready': self.check_pixel_color(375, 1141, (132, 166, 255)),  # Firefly Ult Ready
            'ult_2_ready': self.check_pixel_color(673, 1181, (255, 255, 255)),  # Fugue Ult Ready
            'ult_3_ready': self.check_pixel_color(1013, 1231, (129, 248, 255)),  # TB Ult Ready
            'ult_4_ready': self.check_pixel_color(1314, 1231, (255, 244, 117)),  # Ruan Mei Ult Ready
            'ally_1_selected': self.check_pixel_color(441, 1297, (255, 255, 255)), # Ally Selected 
            'ally_2_selected': self.check_pixel_color(743, 1297, (255, 255, 255)),
            'ally_3_selected': self.check_pixel_color(1044, 1297, (255, 255, 255)),
            'ally_4_selected': self.check_pixel_color(1343, 1297, (255, 255, 255)),
            'superbreak_stacks': self.check_pixel_color(860, 1266, (214, 253, 254)), # TB stacks
            'rm_stacks': self.check_pixel_color(1163, 1265, (255, 250, 220)), # RM stacks
            'fugue_stacks': self.check_pixel_color(559, 1266, (228, 225, 255)), # Fugue stacks
            'skill_points_5': self.check_pixel_color(2029, 1292, (255, 255, 255)),
            'skill_points_4': self.check_pixel_color(2002, 1293, (255, 255, 255)),
            'skill_points_3': self.check_pixel_color(1976, 1291, (255, 255, 255)),
            'skill_points_2': self.check_pixel_color(1955, 1292, (255, 255, 255)),
            'skill_points_1': self.check_pixel_color(1930, 1291, (255, 255, 255)),
            'firefly_turn': self.check_pixel_color(440, 1118, (91, 130, 151)),
            'fugue_turn': self.check_pixel_color(744, 1121, (85, 123, 142)),
            'tb_turn': self.check_pixel_color(1046, 1129, (7, 10, 12)),
            'rm_turn': self.check_pixel_color(1346, 1122, (74, 106, 124)),
            'player_turn': self.check_pixel_color(80, 113, (255, 255, 0)),
            'ff_enhanced': self.check_pixel_color(361, 1187, (186, 193, 97)),
        }
        return state
    def check_coordinates(self, x, y):
        # Print RGB values at given coordinates
        screenshot = np.array(self.sct.grab(self.monitor))
        pixel_color = screenshot[y][x][:3]  # Get RGB values
        print(f"Coordinates ({x}, {y}): RGB{tuple(pixel_color)}")
        return pixel_color

    def act(self):
        state = self.get_screen_state()
        if state['ult_1_ready'] and state['superbreak_stacks'] and state['rm_stacks']:
            self.take_action(2)
            wait()
            self.take_action(8)

        if state['ult_4_ready']:
            self.take_action(5)
            wait()
            self.take_action(8)
            time.sleep(2)

        if state['ult_2_ready'] and state['superbreak_stacks']:
            self.take_action(3)
            wait()
            self.take_action(8)
            time.sleep(2)

        if state['ult_3_ready'] and not state['superbreak_stacks']:
            self.take_action(4)
            wait()
            self.take_action(8)
            time.sleep(2)

        if state['firefly_turn'] and state['ff_enhanced']:
            self.take_action(1)
            wait()
            self.take_action(1)
            return

        if not state['fugue_stacks'] and state['skill_points_1']:
            self.take_action(1)
            wait()
            self.take_action(1)
            return
        
        if state['firefly_turn'] and not state['ff_enhanced'] and state['skill_points_1']:
            self.take_action(1)
            wait()
            self.take_action(1)
            return

        if state['tb_turn'] and state['ult_3_ready'] and not state['skill_points_5']:
            self.take_action(4)
            wait()
            self.take_action(8)
            time.sleep(5)
            if not state['skill_points_3']:
                self.take_action(0)
                return
            else:
                self.take_action(1)
                wait()
                self.take_action(1)
                return
        
        if state['tb_turn'] and not state['skill_points_3']:
            self.take_action(1)
            wait()
            self.take_action(1)
            return

        if state['rm_turn'] and not state['rm_stacks']:
            self.take_action(1)
            wait()
            self.take_action(1)
            return
        elif state['rm_turn']:
            self.take_action(0)
            return

        if state['fugue_turn'] and not state['fugue_stacks']:
            self.take_action(1)
            wait()
            if state['ally_1_selected']:
                self.take_action(1)
                return
        elif state['fugue_turn']:
            self.take_action(0)
            return
                
        if state['player_turn']:
            self.take_action(8)
            return

def main():
    env = HSREnvironment()
    try:
        while True:
            # env.check_coordinates(1039, 1117)  
            # env.check_coordinates(744, 1121) 
            x = env.get_screen_state()
            state = env.act()
            print("Current state:", state)
            print("Current states", x)
            time.sleep(4)
    except KeyboardInterrupt:
        print("Monitoring stopped")

if __name__ == "__main__":
    #time.sleep(6)
    #pyautogui.press('e')
    main()




#if check_pixel_color(80, 113, (255, 255, 0)):
    #time.sleep(stdDelay)
'''
i = 4

for x in range(i):
    time.sleep(2.5)
    pyautogui.write(['e']) 
    time.sleep(stdDelay)'''