import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name = 'multilang-summarizer',
    packages = ['multilang_summarizer'],
    version = 'v1.02-beta',
    license='GPLv3',
    description = 'Multilanguage summarizer, intended to improve text readability',
    long_description = README,
    long_description_content_type = "text/markdown",
    author = 'Arturo Curiel',
    author_email = 'me@arturocuriel.com',
    url = 'http://www.arturocuriel.com',
    download_url = 'https://github.com/elmugrearturo/multilang_summarizer/archive/v1.02-beta.tar.gz',
    include_package_data=True,
    keywords = ['SUMMARIZATION', 'MULTILANGUAGE', 'RULE-BASED'],
    install_requires=[
            'nltk',
            'pyphen',
            'textstat',
            'sentence-splitter',
            'numpy',
        ],
    classifiers=[
      'Development Status :: 4 - Beta',
      'Intended Audience :: Developers',
      'Topic :: Software Development :: Build Tools',
      'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
      'Programming Language :: Python :: 3',
      'Programming Language :: Python :: 3.4',
      'Programming Language :: Python :: 3.5',
      'Programming Language :: Python :: 3.6',
      'Programming Language :: Python :: 3.7',
    ],
)
