from setuptools import setup

setup(
    name='stockit',
    url='https://github.com/BenCaunt8300/stockit',
    author='Ben Caunt',
    author_email='bdcaunt@gmail.com',
    packages=['stockit'],
    install_requires=['pandas','numpy','matplotlib','tqdm','sklearn'],
    version='1.0',
    license='MIT',
    description='python library for easy stock analysis and future price estimations. FOR EDUCATIONAL PURPOSES ONLY, NOT INVESTMENT ADVICE',
    classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'Topic :: Software Development :: Build Tools',
      'License :: OSI Approved :: MIT License',
      'Programming Language :: Python :: 3',
      'Programming Language :: Python :: 3.4',
      'Programming Language :: Python :: 3.5',
      'Programming Language :: Python :: 3.6',
      'Programming Language :: Python :: 3.7',
      'Programming Language :: Python :: 3.8',
    ],

)

