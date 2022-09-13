from setuptools import setup

requirements = [
    'torch',
    'scipy'
]

setup(
    name='fnet-pytorch',
    version='0.1.0',
    install_requires=requirements,
    description='PyTorch implementation of Google\'s FNet',
    url='https://github.com/erksch/fnet-pytorch',
    license='MIT License',
    py_modules=['fnet']
)
