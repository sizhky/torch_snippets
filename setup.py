try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

VERSION = '0.238'
setup(
  name = 'torch_snippets',         # How you named your package folder (MyLib)
  packages = ['torch_snippets'],   # Chose the same as "name"
  version = VERSION,      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'utilites for torch metrics',   # Give a short description about your library
  author = 'Yeshwanth',                   # Type in your name
  author_email = '1992chinna@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/sizhky/torch_snippets',   # Provide either the link to your github or to your website
  download_url = f'https://github.com/sizhky/torch_snippets/archive/{VERSION}.tar.gz',    # I explain this later on
  keywords = ['plot', 'torch'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'loguru',
          'numpy',
          'pandas',
          'dill',
          'tqdm',
          'matplotlib',
          'pandas',
          'opencv-python-headless',
          'Pillow',
      ],
  classifiers=[
    'Development Status :: 4 - Beta',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
)

# python setup.py sdist bdist_wheel && python -m twine upload dist/*
