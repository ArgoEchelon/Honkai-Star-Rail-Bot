import pyautogui
import time
import pytesseract
import cv2
import numpy as np
import mss

# Test File to check detections outside of the RL agent being run
class HSREnvironment:
    def __init__(self, monitor_number=1, debug=False):
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[monitor_number]
        self.debug = debug
    
    def screen_cap(self):
        # Capture Screenshot
        screenshot = np.array(self.sct.grab(self.monitor))

        # Convert from BGRA to BGR
        bgr_screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
        return bgr_screenshot
    

    def template_match(self, regions_templates, screenshot=None):
        """
        Check multiple templates in specific regions of the screen
        
        Args:
            regions_templates: List of dictionaries with format:
                {
                    'name': 'template_name',
                    'template_path': 'path/to/template.png',
                    'region': (x, y, width, height),  # Optional, if None checks full screen
                    'threshold': 0.8  # Optional, default 0.8
                }
        
        Returns:
            Dictionary with template names as keys and boolean values indicating presence
        """

        # Catch just in case a screenshot isn't inputted
        if screenshot is None:
            screenshot = self.screen_cap()
        results = {}

        # Convert screenshot to greyscale for template matching
        grey_screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

        for config in regions_templates:
            name = config['name']
            template_path = config['template_path']
            region = config.get('region', None)
            threshold = config.get('threshold', 0.8)

            # Load template
            template = cv2.imread(template_path, 0) # 0 is greyscale

            if template is None:
                print(f"Warning: Could not load template {template_path}")
                results[name] = False
                continue
            
            h, w = template.shape

            # Crop screenshot to region if specified
            if region is not None:
                x, y, width, height = region
                roi = grey_screenshot[y:y+height, x:x+width]
            else:
                roi = grey_screenshot
            
            if roi.shape[0] < h or roi.shape[1] < w:
                print(f"Warning: Region too small for template {name}")
                results[name] = False
                continue

            # Perform template matching
            try:
                result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                # Check if template was found based on threshold
                results[name] = max_val >= threshold

                # For debugging - visualise matches
                if self.debug and results[name]:
                    img_debug = screenshot.copy()

                    # Adjust coordinates if using region 
                    if region is not None:
                        top_left = (max_loc[0] + x, max_loc[1]+ y)
                    else:
                        top_left = max_loc
                    
                    bottom_right = (top_left[0] + w, top_left[1] + y)
                    cv2.rectangle(img_debug, top_left, bottom_right, (0, 255, 0), 2)
                    cv2.putText(img_debug, f"{name}: {max_val:.2f}",
                                (top_left[0], top_left[1] - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    cv2.imshow('Template Match Debug', img_debug)
                    cv2.waitKey(100) # Brief display

            except Exception as e:
                print(f"Error matching template {name}: {e}")
                results[name] = False

        if self.debug:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return results, screenshot
    
    def get_text_from_region(self, region=None, screenshot=None):
        """
        Get text from a specific region using pytesseract
        
        Args:
            region: Tuple (x, y, width, height) or None for full screen
            
        Returns:
            Extracted text as string
        """
        if screenshot is None:
            screenshot = self.screen_cap()

        if region is not None:
            x, y, width, height = region
            region_img = screenshot[y:y+height, x:x+width]
        else:
            region_img = screenshot
        
        # Convert from BGR to RGB so pytesseract can read it
        rgb_screenshot = cv2.cvtColor(region_img, cv2.COLOR_BGR2RGB)
        text = pytesseract.image_to_string(rgb_screenshot, config='--psm 8')
        return text
    
    def screen_check(self, template=None, text=None):
        screenshot = self.screen_cap()
        result = {
            'templates': {},
            'text': {}
        }

        if template:
            template_results, i = self.template_match(template, screenshot=screenshot)
            result['templates'] = template_results

        if text:
            for region_name, region_coords in text.items():
                result['text'][region_name] = self.get_text_from_region(
                    region=region_coords, 
                    screenshot=screenshot
                )
        
        return result

    def check_state(self):
        templates = [
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

        text_regions = {
            'remaining_action_value': (2395, 387, 160, 70),
            'skill_points': (1860, 1264, 45, 65),
            'lingsha_stacks': (1158, 1247, 25, 30),
        }
        
        results = self.screen_check(template=templates, text=text_regions)

        return results

def main():
    #time.sleep(2)
    env = HSREnvironment()
    test = env.check_state()
    print(test)
    #text = env.screen_cap_text()
    #print(text)

if __name__ == "__main__":
    main()