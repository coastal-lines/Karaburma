#### L E N A | Machine Learning and Computer Vision based solution for searching typical GUI elements on the screen    
##### Luminous Elements Navigation Assistant

![image](https://github.com/coastal-lines/Lena/assets/70205794/44016028-823f-4b6e-b85f-0060d753a11e)


#### Reason of creating:
I had only screenshots and couldn't use tools like Selenium.

This project was started as a framework for Assessment Delivery functional visual testing.


#### Current features:
- detect the most popular WPF elements
- scrolling and reading the full text from a ListBox element
- scrolling through the Table element in both directions and reading all table cells
- flexible image search method (search using Affine and Pyramid operations)
- return coordinates of all elements in JSON format
- supports a variety of computer vision methods for extensions

#### Can be used as:
- package for python
- api service for any language

#### How to install:
- download this project
- navigate into folder with 'setup.py' file from the project
- run command 'python setup.py install'

#### How to import:
- package: 
```
from lena.main import Lena
```
- api service:
```
from lena.api.main import LenaApiService
```

#### How to use:
- find all elements, get the first button and click on the centre element:
```
lena = Lena(config_path="config.json", source_mode="screenshot", detection_mode="default", logging=False)

json_elements = lena.find_all_elements()

button1 = [element for element in json_elements.get('elements', []) if element.get('label') == 'button'][0]

pyautogui.click(x=button1['centre'][0], y=button1['centre'][1])
```
