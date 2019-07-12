from setuptools import setup

setup(
    name='spectools',
    version='0.1.4',
    description='Helpers for 1D astrophysical spectroscopy',
    url='https://github.com/thriveth/spectools/',
    author='T. E. Rivera-Thorsen',
    author_email='trive@astro.su.se',
    packages=['spectools'],
    package_data={'spectools':['data/*.csv', 'data/*.ecsv', 'data/*.txt']},
    license='GPL3',
)
