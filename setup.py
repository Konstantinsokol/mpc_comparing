from setuptools import setup, find_packages

setup(
    name='simple-mpc',
    version='0.1.0',
    description='Simple MPC implementation for robot control',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=[
        'numpy',
    ],
    python_requires='>=3.6',
)