from setuptools import setup, find_packages

setup( name = "markovmodel",
	version = "1.02",
	author = 'Chris Ward',
	author_email = 'chrisward@email.com',
	packages=find_packages(include=['markovmodel','markovmodel.*']),
	install_requires = ['numpy>=1.26.0','pandas'])