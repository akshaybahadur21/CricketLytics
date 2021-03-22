from src.Batting.Batting import Batting
from src.Bowling.Bowling import Bowling
import argparse

from src.Bowling.LeftHandedBowling import LeftHandedBowling
from src.Bowling.RightHandedBowling import RightHandedBowling
from src.Batting.LeftHandedBatting import LeftHandedBatting
from src.Batting.RightHandedBatting import RightHandedBatting

class Cricket:
    def __init__(self):
        self.batting = Batting()
        self.bowling = Bowling()

    def play(self, option, hand, view):
        if option.lower() == str("batting"):
            if hand.lower() == str("left"):
                left_handed = LeftHandedBatting()
                left_handed.bat(view)
            elif hand.lower() == str("right"):
                right_handed = RightHandedBatting()
                right_handed.bat(view)

        elif option.lower() == str("bowling"):
            if hand.lower() == str("left"):
                left_handed = LeftHandedBowling()
                left_handed.bowl(view)
            elif hand.lower() == str("right"):
                right_handed = RightHandedBowling()
                right_handed.bowl(view)
        else:
            raise ValueError("Input can only be either Batting or Bowling")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-option', "--option", required=True, help="batting or bowling",
                        type=str)
    parser.add_argument('-hand', "--hand", required=True, help="left or right", type=str)
    parser.add_argument('-view', "--view", required=True, help="Front or side", type=str)
    args = parser.parse_args()
    option = args.option
    hand = args.hand
    view = args.view
    cricket = Cricket()
    cricket.play(option, hand, view)
