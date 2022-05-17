# from pynput.keyboard import Key, Controller
# import time
# keyboard = Controller()
# import pyautogui

# # Press and release space
# keyboard.press(Key.space)
# keyboard.release(Key.space)
# time.sleep(4)

# # Type a lower case A; this will work even if no key on the
# # physical keyboard is labelled 'A'
# while True:
#     # keyboard.press('a')
#     # keyboard.release('a')
#     # time.sleep(0.1)
#     pyautogui.keyDown('a')
# from ahk import AHK

# ahk = AHK()

# ahk.type('hello, world!')  # Send keys, as if typed (performs ahk string escapes)
# ahk.send_input('Hello`, World{!}')  # Like AHK SendInput, must escape strings yourself!
# ahk.key_state('Control')  # Return True or False based on whether Control key is pressed down
# ahk.key_state('CapsLock', mode='T')  # Check toggle state of a key (like for NumLock, CapsLock, etc)
# ahk.key_press('a')  # Press and release a key
# ahk.key_down('Control')  # Press down (but do not release) Control key
# ahk.key_up('Control')  # Release the key
# ahk.key_wait('a', timeout=3)  # Wait up to 3 seconds for the "a" key to be pressed. NOTE: This throws 
#                               # a TimeoutError if the key isn't pressed within the timeout window
# ahk.set_capslock_state("on")  # Turn CapsLock on
import keyboard

keyboard.press_and_release('shift+s, space')

keyboard.write('The quick brown fox jumps over the lazy dog.')

keyboard.add_hotkey('ctrl+shift+a', print, args=('triggered', 'hotkey'))

# Press PAGE UP then PAGE DOWN to type "foobar".
keyboard.add_hotkey('page up, page down', lambda: keyboard.write('foobar'))

# Blocks until you press esc.
keyboard.wait('esc')
keyboard.press("a")