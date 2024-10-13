from setuptools import setup

setup(
    name='gym_soccer',
    version='0.0.1',
    description='Gym soccer environment - useful to replicate soccer experiments from Littman 94',
    url='https://github.com/mimoralea/gym-soccer-littman94',
    author='Miguel Morales',
    author_email='mimoralea@gmail.com',
    packages=['gym_soccer', 'gym_soccer.envs'],
    license='MIT License',
    install_requires=['gymnasium', 'pettingzoo'],
)
