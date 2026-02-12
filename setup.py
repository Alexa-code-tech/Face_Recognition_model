from setuptools import setup, find_packages

setup(
    name='Face_Recognition_model',
    version='0.1.0',
    author='Alexa-code-tech',
    author_email='youremail@example.com',
    description='A package for face recognition using machine learning.',
    packages=find_packages(),
    install_requires=[
        'opencv-python',
        'numpy',
        'tensorflow',
        'keras',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)