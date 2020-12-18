# Welcome to stochastics
![Version](https://img.shields.io/badge/version-0.3.0-blue.svg?cacheSeconds=2592000)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A simple library to help solving stochastic processes and finding resulting probilities of common applications

## Installation / How to use

To use this library is pretty easy. First you will need to install the library from PyPI. Make sure you are running Python version 3.8 or above:
```sh
pip install stochastics
```

The you can use the library in your own python code, like this:
```python
from stochastics.models import MarkovChain

# define some initial values for a markov chain
initial_probabilities = [0.3, 0.4]
transition_matrix = [
    [0.1, 0.9],
    [0.7, 0.3],
]

# create the markov chain with the parameters
mc = MarkovChain(initial_probabilities, trans_matrix=transition_matrix)
# find the probability of ending up in a state given a sequence of steps
p = mc.get_probability_from_sequence(state_sequence=[0, 1, 0])
print(p)
```

More utilization examples and API documentation can be seem in the **docs**.

## Development Setup

So you want to contribute to the development, that is very welcome! To develop this project there're only two initial requirements:

- Python 3.8+
- Having pip or pipenv installed

After making sure those requirements are fullfilled, you can simple run this command with `pip`:
```sh
pip install -r requirements-dev.txt
```

If you like `pipenv`, just do:
```sh
pipenv install --dev
```

That is already all you need. Just make sure to follow the Contribution Guidelines and you are all set.

After doing your work and getting ready to make your PR, I recommend you to use `pre-commit` to fix some standards and make sure nothing really can get in the way of your contribution.

## Run tests

There're tests in this project, and the best way to see how things work under the hood and check if you changes are ok, simply run the tests with `pytest`:
```sh
pytest
```

You can also see if the current test coverage of the project is good by running:
```sh
pytest --cov
```

## Need help? Found an bug? Have a good idea?

I encourage you to look around in the [issues](https://github.com/deadpyxel/stochastics/issues) section, see if someone else is working on a fix for your problem, there's already a report for the bug, or some new feature being dicussed that you are interested. If you found nothing, feel free to create an issue using one of the provided templates.

## Author

**Robson Cruz**

* Website: https://deadpyxel.github.io/
* Github: [@deadpyxel](https://github.com/deadpyxel)

## Show your support

Give a ⭐️ if this project helped you!


***
_This README was generated with ❤️ by [readme-md-generator](https://github.com/kefranabg/readme-md-generator)_
