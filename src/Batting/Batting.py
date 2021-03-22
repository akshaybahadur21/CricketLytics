class Batting:
    def __init__(self):
        pass

    def front_batting(self):
        raise NotImplementedError("dispatch to subclass")

    def side_batting(self):
        raise NotImplementedError("dispatch to subclass")

    def bat(self, view):
        if view == "front":
            self.front_batting()
        elif view == "side":
            self.side_batting()
