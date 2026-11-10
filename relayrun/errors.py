"""Errors that the CLI prints and turns into a non-zero exit."""


class RelayError(Exception):
    """A failed command. ``exit_code`` 2 is usage; 1 is a check or tool failure."""

    def __init__(self, message: str, exit_code: int = 1):
        super().__init__(message)
        self.exit_code = exit_code
