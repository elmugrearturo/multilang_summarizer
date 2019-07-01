from distutils.core import setup

setup(
    name = 'multilang-summarization',
    packages = ['multilang_summarization'],
    version = 'v1.0-beta',
    license='GPLv3',
    description = 'Multilanguage summarizer, intended to improve text readability',
    author = 'Arturo Curiel',
    author_email = 'me@arturocuriel.com',
    url = 'arturocuriel.com',
    download_url = 'https://github.com/elmugrearturo/multilang_summarizer/archive/v1.0-beta.tar.gz',
    keywords = ['SUMMARIZATION', 'MULTILANGUAGE', 'RULE-BASED'],
    install_requires=[
            'nltk',
            'pyphen',
            'textstat',
            'sentence-splitter',
        ],
    classifiers=[
      'Development Status :: 4 - Beta',
      'Intended Audience :: Developers',
      'Topic :: Software Development :: Build Tools',
      'OSI Approved :: GNU General Public License v3 (GPLv3)'
      'Programming Language :: Python :: 3',
      'Programming Language :: Python :: 3.4',
      'Programming Language :: Python :: 3.5',
      'Programming Language :: Python :: 3.6',
      'Programming Language :: Python :: 3.7',
    ],
)
