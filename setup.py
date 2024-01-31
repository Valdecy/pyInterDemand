from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / 'README.md').read_text()

setup(
    name='pyInterDemand',
    version='1.4.3',
    license='GNU',
    author='Valdecy Pereira',
    author_email='valdecy.pereira@gmail.com',
    url='https://github.com/Valdecy/pyInterDemand',
    packages=find_packages(),
    install_requires=[
        'matplotlib',
        'numpy',
		'pandas'
    ],
    description='pyInterDemand a Python Library for Intermittent Demand Methods',
    long_description=long_description,
    long_description_content_type='text/markdown',
)
