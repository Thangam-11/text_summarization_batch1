from setuptools import setup, find_packages
from pathlib import Path

def get_requirements(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file if line.strip() and not line.startswith('#')]

setup(
    name='text_summarization_project',
    version='0.1.0',
    author='thangarasu',
    author_email='thangamani1128@gmail.com',
    description='A text summarization project using Flask and NLP techniques',
    long_description=Path('README.md').read_text(),
    long_description_content_type='text/markdown',
    packages=find_packages(),  # Use 'packages' instead of an additional find_packages
    install_requires=get_requirements('requirements.txt'),
    python_requires='>=3.10',
)
