from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(name='multitask_negation_target',
      version='0.0.1',
      description='Multi-Task negation and targeted sentiment',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/jbarnesspain/multitask_negation_for_targeted_sentiment',
      author='Jeremy Barnes, Andrew Moore',
      install_requires=[
          'allennlp==0.9.0'
      ],
      python_requires='>=3.6.1',
      packages=find_packages(),
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Natural Language :: English',
          'Programming Language :: Python :: 3.6',
          'Topic :: Text Processing',
          'Topic :: Text Processing :: Linguistic',
      ])