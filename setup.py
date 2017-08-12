from setuptools import setup, find_packages

import pgm

NAMESPACE_PACKAGES = []
INSTALL_REQUIRES = []
DISTNAME = 'pgm'
DOWNLOAD_URL = ''
VERSION = pgm.__version__
AUTHOR = 'matt johnson'
DESCRIPTION = 'Inference for PGMs'


def setup_package():
    metadata = dict(name=DISTNAME,
                    packages=find_packages(),
                    author=AUTHOR,
                    description=DESCRIPTION,
                    version=VERSION,
                    install_requires=INSTALL_REQUIRES,
                    namespace_packages=NAMESPACE_PACKAGES)
    setup(**metadata)
if __name__ == '__main__':
    setup_package()
