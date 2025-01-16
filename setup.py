from setuptools import setup, find_packages

setup(
    name='pcsis',
    version='0.1',
    python_requires='>=3.8',
    packages=find_packages(),
    install_requires=[
        'gurobipy',
        'matplotlib==3.7.5',
        'numpy==1.24.4',
        'scikit-learn==1.3.2',
        'sympy==1.12',
    ],
)