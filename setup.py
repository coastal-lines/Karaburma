from setuptools import setup, find_packages
from glob import glob


setup(
    name='karaburma',
    version='0.1b',
    packages=find_packages(),
    install_requires=[
        'numpy==1.26.4',
        'loguru==0.7.2',
        'fastapi==0.104.1',
        'opencv-python==4.8.0.76',
        'scikit-image',
        'scikit-learn==1.3.2',
        'PyAutoGUI==0.9.54',
        'PyVirtualDisplay==0.2.5',
        'python-xlib==0.27',
        'python-configuration==0.9.1',
        'imutils==0.5.4',
        'pytesseract==0.3.10',
        'pandas==2.1.1',
        'uvicorn==0.23.2',
        'matplotlib==3.8.0',
        'requests==2.31.0',
        'mlxtend==0.23.0',
        'pytest==8.0.0',
        'pytest-asyncio==0.23.5',
        'starlette==0.27.0',
        'python-multipart==0.0.9'
    ],
    entry_points={
        'console_scripts': [
            'karaburma-cli = karaburma:main',
        ],
    }
)