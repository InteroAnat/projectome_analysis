from setuptools import setup

setup(
    name='generate-tree',
    version='1.0',
    py_modules=['generate_tree'],
    entry_points={
        'console_scripts': [
            'gentree=generate_tree:main',
        ],
    },
)
