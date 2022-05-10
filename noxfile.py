"""Nox sessions."""

from typing import Any

import nox
from nox.sessions import Session


nox.options.sessions = "black", "flake8", "mypy", "tests"
locations = "src", "noxfile.py"


def install_with_constraints(session: Session, *args: str, **kwargs: Any) -> None:
    session.install("poetry")
    session.run("poetry", "install")


@nox.session(python="3.9")
def black(session: Session) -> None:
    """Run black code formatter."""
    args = session.posargs or locations
    install_with_constraints(session, "black")
    session.run("poetry", "run", "black", *args)


@nox.session(python="3.9")
def flake8(session: Session) -> None:
    """Type-check using mypy."""
    args = session.posargs or locations
    install_with_constraints(session, "flake8")
    session.run("poetry", "run", "flake8", *args)


@nox.session(python="3.9")
def mypy(session: Session) -> None:
    """Type-check using mypy."""
    args = session.posargs or locations
    install_with_constraints(session, "mypy")
    session.run("poetry", "run", "mypy", *args)


@nox.session(python="3.9")
def tests(session: Session) -> None:
    """Run the test suite."""
    args = session.posargs
    install_with_constraints(session, "pytest")
    #session.run("poetry", "install", "--no-dev", external=True)
    session.run("poetry", "run", "pytest", *args)
