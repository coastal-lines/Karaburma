import copy
import time
import pyautogui


def click(*args):
    time.sleep(0)
    pyautogui.moveTo(x=args[0][0], y=args[0][1])
    pyautogui.click(x=args[0][0], y=args[0][1])

def click_and_return_difference(element, *args):
    pyautogui.moveTo(x=args[0][0], y=args[0][1])

    element.get_roi_element().update_element_roi_area_by_screenshot()
    before = copy.deepcopy(element.get_roi_element().get_roi())

    pyautogui.click(x=args[0][0], y=args[0][1])
    time.sleep(1)

    element.get_roi_element().update_element_roi_area_by_screenshot()
    after = element.get_roi_element().get_roi()

    return before, after