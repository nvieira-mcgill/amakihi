# Contributing

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

You can contribute in many ways:

## Types of Contributions

### Report Bugs

Report bugs at <https://github.com/nvieira-mcgill/amakihi/issues>.

If you are reporting a bug, please include:

- Your operating system name and version.
- Any details about your local setup that might be helpful in troubleshooting.
- Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with \"bug\"
and \"help wanted\" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with
\"enhancement\" and \"help wanted\" is open to whoever wants to
implement it.

### Write Documentation

amakihi could always use more documentation, whether as part of the
official amakihi docs, in docstrings, or even on the web in blog posts,
articles, and such.

### Submit Feedback

The best way to send feedback is to file an issue at
<https://github.com/nvieira-mcgill/amakihi/issues>.

If you are proposing a feature:

- Explain in detail how it would work.
- Keep the scope as narrow as possible, to make it easier to implement.
- Remember that this is a volunteer-driven project, and that contributions are welcome :)

## Getting Started

Ready to contribute? Here\'s how to set up [amakihi](https://github.com/nvieira-mcgill/amakihi) for
local development.

1. Fork the [amakihi](https://github.com/nvieira-mcgill/amakihi) repo on GitHub.

2. Clone your fork locally:

    ```shell
    git clone git@github.com:your_name_here/amakihi.git
    ```

3. Amakihi uses Poetry[https://python-poetry.org] for dependency management. To install Poetry, follow the instructions [here](https://python-poetry.org/docs/#installation).

    Once Poetry is installed, install the dependencies for amakihi:

    ``` shell
    cd /where/you/cloned/amakihi
    poetry install
    ```

    This will install all dependencies, including development dependencies in a virtualenv.

    To activate the virtualenv, run:

    ```shell
    poetry shell
    ```

    To add a new dependency, run:

    ```shell
    poetry add <package-name>
    ```

    To add a new development dependency, run:

    ```shell
    poetry add --group=dev <package-name>
    ```

    To remove a dependency, run:

    ```shell
    poetry remove <package-name>
    ```

4. Create a branch for local development and make your changes locally:

   ``` shell
   git checkout -b name-of-your-bugfix-or-feature
   ```

5. When you\'re done making changes, check that your changes pass pre-commit checks. As the name suggests, these checks are run before you commit your changes. To run these checks, run:

    ``` shell
    pre-commit run --all-files
    ```

    Alternatively, to run these checks automatically before every commit, run:

    ``` shell
    pre-commit install
    ```

    This will install a git hook that will run the pre-commit checks before every commit. If the checks fail, the commit will be aborted.

6. To commit your changes and push your branch to GitHub:

   For standardized commit messages, amakihi uses [commitizen](https://github.com/commitizen/cz-cli). To use commitizen, run:

    ``` shell
    git add .
    $ cz commit
    ```

    This will prompt you to fill out any necessary fields for your commit message.

    **Note:** We use standardized commit messages to generate the changelog and automatically bump the version number when a new release is made.

    ``` shell
    git push origin name-of-your-bugfix-or-feature
    ```

7. Submit a pull request through the GitHub website.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated.Put your new functionality into a function with a docstring.
3. The pull request should work for Python >=3.8

## Tips

To run tests, you can use the following command:

```shell
poetry shell
pytest
```

This will run all tests in the tests directory and subdirectories.

To run a subset of tests:

``` shell
python pytests tests/test_amakihi.py
```
