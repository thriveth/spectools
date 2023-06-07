from setuptools import setup

setup(
    name='spectools',
    version='0.2',
    description='Helpers for 1D astrophysical spectroscopy',
    url='https://github.com/thriveth/spectools/',
    author='T. Emil Rivera-Thorsen',
    author_email='trive@astro.su.se',
    packages=['spectools'],
    package_data={
        'spectools':
            ['data/*.csv', 'data/*.ecsv', 'data/*.txt',
             'data/UVES_sky_tables/*.dat'
             ]
            },
    license='GPL3',
)
