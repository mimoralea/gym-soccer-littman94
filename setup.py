from setuptools import setup, find_packages

setup(
    name='gym_soccer',
    version='0.0.1',
    description='Gym soccer environment - useful to replicate soccer experiments from Littman 94',
    url='https://github.com/mimoralea/gym-soccer-littman94',
    author='Miguel Morales',
    author_email='mimoralea@gmail.com',
    packages=find_packages(),  # Automatically find and include packages in the directory
    license='MIT License',
    install_requires=[
        'numpy==1.26.4',
        'gym>=0.26.2'
    ],
)
