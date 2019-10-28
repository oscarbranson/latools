from setuptools import setup, find_packages

# bcause TravisCI was being a jerK
try:
  from latools import __version__
except:
  __version__ = "version_missing"

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='latools',
      version=__version__,
      description='Tools for LA-ICPMS data analysis.',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/oscarbranson/latools',
      author='Oscar Branson',
      author_email='oscarbranson@gmail.com',
      license='MIT',
      packages=find_packages(),
      classifiers=['Development Status :: 4 - Beta',
                   'Intended Audience :: Science/Research',
                   'Topic :: Scientific/Engineering',
                   'Programming Language :: Python :: 2',
                   'Programming Language :: Python :: 3',
                   ],
      python_requires='>3.6',
      install_requires=['numpy',
                        'pandas',
                        'matplotlib',
                        'uncertainties',
                        'sklearn',
                        'scipy',
                        'Ipython',
                        'configparser',
                        'tqdm'
                        ],
      package_data={
        'latools': ['latools.cfg',
                    'resources/*',
                    'resources/data_formats/*',
                    'resources/test_data/*'],
      },
      zip_safe=False)
