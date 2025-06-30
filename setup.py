from setuptools import setup, find_packages

setup(
    name='BindingRMSD',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        # Core dependencies with updated secure versions
        'numpy>=1.24.0',
        'pandas>=2.0.0',
        'torch>=2.4.0',
        'dgl>=2.4.0',
        'rdkit>=2024.3.5',
        
        # HTTP and networking - updated for security
        'urllib3>=2.2.3',  # Fixed recent vulnerabilities
        'requests>=2.32.3',  # Fixed CVE-2024-35195
        'certifi>=2024.8.30',
        
        # Template and web security
        'jinja2>=3.1.4',  # Note: Has known SSTI issues, but latest available
        'markupsafe>=2.1.5',
        
        # File processing and parsing
        'pillow>=10.4.0',  # Fixed CVE-2024-28219
        'pyyaml>=6.0.2',  # Fixed arbitrary code execution
        
        # Utilities and support
        'click>=8.1.7',
        'tqdm>=4.66.5',
        'packaging>=24.1',
        'filelock>=3.16.1',
        'fsspec>=2024.9.0',
        'six>=1.16.0',
        
        # Scientific computing
        'scipy>=1.14.1',
        'networkx>=3.3',
        'sympy>=1.13.3',
        'mpmath>=1.3.0',
        
        # Date and time
        'python-dateutil>=2.9.0',
        'pytz>=2024.2',
        'tzdata>=2024.2',
        
        # Development and validation
        'pydantic>=2.9.2',
        'pydantic-core>=2.23.4',
        'typing-extensions>=4.12.2',
        
        # System utilities
        'psutil>=6.0.0',
        'charset-normalizer>=3.3.2',
        'idna>=3.10',
        
        # GPU acceleration (optional)
        'triton>=3.0.0; platform_machine != "aarch64"',  # Conditional install
        
        # Specialized chemistry tools
        'meeko>=0.5.0',  # For AutoDock file processing
    ],
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'isort>=5.12.0',
            'mypy>=1.5.0',
            'safety>=3.0.0',
        ],
        'docs': [
            'sphinx>=7.0.0',
            'sphinx-rtd-theme>=1.3.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'bindingrmsd-inference=bindingrmsd.inference:main',
        ],
    },
    author='Jaemin Sim',
    author_email='sjm0775@snu.ac.kr',
    description='Protein-ligand Binding RMSD prediction method using Graph Neural Networks',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/eightmm/BindingRMSD',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Operating System :: OS Independent',
    ],
    keywords='bioinformatics, drug discovery, molecular docking, graph neural networks, RMSD prediction',
    python_requires='>=3.10',
    project_urls={
        'Bug Reports': 'https://github.com/eightmm/BindingRMSD/issues',
        'Source': 'https://github.com/eightmm/BindingRMSD',
        'Documentation': 'https://github.com/eightmm/BindingRMSD#readme',
    },
)

