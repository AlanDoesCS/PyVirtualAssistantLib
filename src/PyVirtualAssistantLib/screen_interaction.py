import pyautogui
import numpy as np
import easyocr
import cv2


class ScreenInteractor:
    def __init__(self, lang: str) -> None:
        self.reader = easyocr.Reader([lang])
        self.screen_width, self.screen_height = pyautogui.size()

    @staticmethod
    def screenshot() -> np.ndarray:
        """Take a screenshot of the screen."""
        return np.array(pyautogui.screenshot())

    def img_to_text(self, img: np.ndarray) -> str:
        """Extract text from the image using OCR"""
        return " ".join(self.reader.readtext(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), detail=0))

    @staticmethod
    def click(x: int, y: int) -> None:
        """Click on the screen at the given coordinates. (x, y)"""
        pyautogui.click(x, y)

    @staticmethod
    def move_cursor(x: int, y: int) -> None:
        """Move the cursor to the given coordinates. (x, y)"""
        pyautogui.moveTo(x, y)

    @staticmethod
    def type_text(text: str) -> None:
        """Type the given text on the screen."""
        pyautogui.typewrite(text)

    def click_text(self, search_text: str) -> bool: # returns success status
        """Click on the screen at the location of the given text."""
        img = self.screenshot()
        text_with_boxes = self.img_to_text(img)

        for text, box in text_with_boxes:
            if search_text.lower() in text.lower():
                # Calculate the center of the bounding box
                top_left = box[0]
                bottom_right = box[2]
                center_x = int((top_left[0] + bottom_right[0]) / 2)
                center_y = int((top_left[1] + bottom_right[1]) / 2)

                print(f"Clicking on text '{search_text}' at ({center_x}, {center_y})")
                self.click(center_x, center_y)
                return True

        print(f"Text '{search_text}' not found on the screen.")
        return False
