from setuptools import setup

setup(name='gym_locm',
      version='0.0.1',
      install_requires=['gym', 'numpy', 'prettytable', 'pexpect'],
      entry_points={
            'console_scripts': [
                  'locm-runner=gym_locm.scripts.runner:run'
            ]
      })
