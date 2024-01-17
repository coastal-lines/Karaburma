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
- find all elements, get the first button and click on the center element:
  <details>
  
  <summary>example: </summary>
  
  ```
  lena = Lena(config_path="config.json", source_mode="screenshot", detection_mode="default", logging=False)
  
  json_elements = lena.find_all_elements()
  
  button1 = [element for element in json_elements.get('elements', []) if element.get('label') == 'button'][0]
  
  pyautogui.click(x=button1['center'][0], y=button1['center'][1])
  ```
  
  </details>

- find listbox and read full text by scrolling:
  <details>
  
  <summary>example: </summary>
  
  ```
  lena = Lena(config_path="config.json", source_mode="screenshot", detection_mode="default", logging=False)
  
  json_elements = lena.find_listbox_and_expand_and_get_text()

  listbox = [element for element in json_elements.get('elements', []) if element.get('label') == 'listbox'][0]

  listbox_full_text = listbox['text']
  ```

  ![Screenshot_1_1](https://github.com/coastal-lines/Lena/assets/70205794/9a11e143-50cd-4054-be20-cd89c0acce97)

  text output:
  > Item 1, Item 2, Item 3, item 4, item 5, item 6, item 7, item 8, item 9, item 10, Item 11, item 12, item 13, Item 14, Item 15

  </details>

- find text from the latest table cell by scrolling in the both directions:
  <details>
  
  <summary>example: </summary>
  
  ```
  lena = Lena(config_path="config.json", source_mode="screenshot", detection_mode="default", logging=False)

  json_elements = lena.find_table_and_expand_and_read_text()

  full_expanded_table = [element for element in json.get('elements', []) if element.get('label') == 'table'][0]

  text = table['cells'][-1]['text']
  ```

  text output:

  ![Screenshot_6](https://github.com/coastal-lines/Lena/assets/70205794/8d7b7cb6-70a5-47b2-9a1e-29c174bf8faa)

  </details>


