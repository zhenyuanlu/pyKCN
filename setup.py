from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='pyKCN',
    version='0.1.0',
    # This will automatically discover and include all packages
    packages=find_packages(),
    install_requires=[
        # List dependencies here:
        'biopython==1.83',
        'nltk==3.8.1',
        'numpy==1.26.4',
        'pandas==2.1.1',
        'rapidfuzz==3.6.1',
        'xlrd==2.0.1',
        'pyarrow==15.0.0'
    ],
    author='Zhenyuan Lu',
    author_email='lu.zhenyua@northeastern.edu',
    description='A Python Tool for Bridging Scientific Knowledge through Keyword Analysis',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/zhenyuanlu/pyKCN',
    classifiers=[
        'Programming Language :: Python :: 3.12',
        'License :: OSI Approved :: MIT License',
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research'
        # Add more classifiers
    ],
)
