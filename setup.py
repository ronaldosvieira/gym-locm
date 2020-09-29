from setuptools import setup

setup(name='gym_locm',
      version='0.0.1',
      install_requires=['gym', 'numpy', 'prettytable', 'pexpect', 'sty'],
      extras_require={
            'experiments': ['numpy', 'scipy', 'stable_baselines', 'hyperopt',
                            'mplcursors', 'pandas', 'matplotlib', 'scikit-learn'],
      },
      entry_points={
            'console_scripts': [
                  'locm-runner=gym_locm.toolbox.runner:run'
            ]
      })
