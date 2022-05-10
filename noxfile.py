"""Nox sessions."""

import nox
from nox.sessions import Session


nox.options.sessions = "black", "lint", "mypy", "tests"
locations = "src", "noxfile.py"


@nox.session(python="3.9")
def black(session: Session) -> None:
    """Run black code formatter."""
    args = session.posargs or locations
    session.install("black")
    session.run("black", *args)


@nox.session(python="3.9")
def lint(session: Session) -> None:
    args = session.posargs or locations
    session.install("flake8")
    session.run("flake8", *args)


@nox.session(python="3.9")
def mypy(session: Session) -> None:
    """Type-check using mypy."""
    args = session.posargs or locations
    session.install("mypy")
    session.run("mypy", *args)


@nox.session(python="3.9")
def tests(session: Session) -> None:
    """Run the test suite."""
    session.install("poetry")
    session.run("poetry", "run", "pytest")
