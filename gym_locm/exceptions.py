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
    def __init__(self, remaining_draws=1):
        self.remaining_draws = remaining_draws


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


class InvalidCardError(ActionError):
    def __init__(self, instance_id=None, message="Invalid instance id"):
        if instance_id is not None:
            message += f": {instance_id}"

        super().__init__(message)
