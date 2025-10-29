# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/pylhc/turn_by_turn/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                           |    Stmts |     Miss |   Cover |   Missing |
|------------------------------- | -------: | -------: | ------: | --------: |
| turn\_by\_turn/\_\_init\_\_.py |       13 |        0 |    100% |           |
| turn\_by\_turn/ascii.py        |       95 |       11 |     88% |59, 64, 158-159, 196-198, 205-208 |
| turn\_by\_turn/constants.py    |       13 |        0 |    100% |           |
| turn\_by\_turn/doros.py        |       91 |        6 |     93% |109-110, 115-116, 175-176 |
| turn\_by\_turn/errors.py       |        9 |        0 |    100% |           |
| turn\_by\_turn/io.py           |       46 |        3 |     93% |171, 209-210 |
| turn\_by\_turn/iota.py         |       80 |        2 |     98% |     52-53 |
| turn\_by\_turn/lhc.py          |       46 |        0 |    100% |           |
| turn\_by\_turn/madng.py        |       86 |        7 |     92% |70-71, 120-121, 127, 191-192 |
| turn\_by\_turn/ptc.py          |      138 |        7 |     95% |141, 160-161, 193-194, 225, 230 |
| turn\_by\_turn/sps.py          |       64 |        3 |     95% |59, 125-126 |
| turn\_by\_turn/structures.py   |       45 |        3 |     93% |46, 87, 124 |
| turn\_by\_turn/trackone.py     |       58 |        4 |     93% | 82, 86-88 |
| turn\_by\_turn/utils.py        |       50 |        0 |    100% |           |
| turn\_by\_turn/xtrack\_line.py |       39 |        7 |     82% |82-83, 88, 98, 113, 121, 165 |
|                      **TOTAL** |  **873** |   **53** | **94%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/pylhc/turn_by_turn/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/pylhc/turn_by_turn/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pylhc/turn_by_turn/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/pylhc/turn_by_turn/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fpylhc%2Fturn_by_turn%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/pylhc/turn_by_turn/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.