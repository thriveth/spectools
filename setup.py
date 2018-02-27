from setuptools import setup

setup(
    name='spectools',
    version='0.1.1',
    description='Helpers for 1D astrophysical spectroscopy',
    url='https://github.com/thriveth/spectools/',
    author='T. E. Rivera-Thorsen',
    author_email='eriveth@uio.no',
    packages=['spectools'],
    package_data={'spectools':['data/*.csv', 'data/*.ecsv', 'data/*.txt']},
    license='GPL3',
)
