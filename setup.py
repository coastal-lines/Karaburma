from setuptools import setup, find_packages

setup(
    name='karaburma',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'loguru==0.7.2',
        'fastapi==0.104.1',
        'opencv-python==4.8.0.76',
        'scikit-image==0.21.0',
        'scikit-learn==1.3.0',
        'PyAutoGUI==0.9.54',
        'python-configuration==0.9.1',
        'imutils==0.5.4',
        'pytesseract==0.3.10',
        'pandas==2.1.0',
        'uvicorn==0.23.2',
        'matplotlib==3.8.0'
    ],
    entry_points={
        'console_scripts': [
            'karaburma-cli = karaburma:main',
        ],
    }
)