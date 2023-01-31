class Action:
    def __init__(self, action_type, origin=None, target=None):
        self.type = action_type
        self.origin = origin
        self.target = target

    def __eq__(self, other):
        return (
            other is not None
            and self.type == other.type
            and self.origin == other.origin
            and self.target == other.target
        )

    def __repr__(self):
        return f"{self.type} {self.origin} {self.target}"
