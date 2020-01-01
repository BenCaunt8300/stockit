from setuptools import setup

setup(
    name='stockit',
    url='https://github.com/BenCaunt8300/stockit',
    download_url = 'https://github.com/BenCaunt8300/stockit/archive/v1.0.7.2.tar.gz',
    author='Ben Caunt',
    author_email='bdcaunt@gmail.com',
    packages=['stockit'],
    install_requires=['pandas','numpy','matplotlib','tqdm','sklearn','pandas_datareader'],
    version='v1.0.7.2',
    license='Apache 2.0',
    description='Python module containing a bundle of algorithms for financial analysis and future stock price estimations.',
    classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'Topic :: Software Development :: Build Tools',
      'License :: OSI Approved :: Apache Software License',
      'Programming Language :: Python :: 3',
      'Programming Language :: Python :: 3.4',
      'Programming Language :: Python :: 3.5',
      'Programming Language :: Python :: 3.6',
      'Programming Language :: Python :: 3.7',
      'Programming Language :: Python :: 3.8',
    ],

)
