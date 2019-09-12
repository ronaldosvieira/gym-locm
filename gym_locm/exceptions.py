class GameError(Exception):
    pass


class ActionError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


class FullHandError(GameError):
    pass


class EmptyDeckError(GameError):
    pass


class WardShieldError(GameError):
    pass


class NotEnoughManaError(ActionError):
    def __init__(self, message="Not enough mana"):
        super().__init__(message)


class FullLaneError(ActionError):
    def __init__(self, message="Lane is full"):
        super().__init__(message)


class MalformedActionError(ActionError):
    pass


class GameIsEndedError(ActionError):
    def __init__(self, message="Game is ended"):
        super().__init__(message)


class InvalidCardRefError(ActionError):
    def __init__(self, instance_id=None, message="Invalid card reference"):
        if instance_id is not None:
            message += f": {instance_id}"

        super().__init__(message)
