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
    pass


class FullLaneError(ActionError):
    pass


class MalformedActionError(ActionError):
    pass
