"""Example plugin: any file in this folder that defines @skill functions is
loaded at startup. After editing, say "reload skills" or run
`jarvis skills reload`. Try this one with:

    jarvis skills test roll_dice sides=20 count=2
"""

import random

from jarvis import skill
from jarvis.skills.base import SkillError


@skill(tags=["custom", "fun"])
def roll_dice(sides: int = 6, count: int = 1) -> dict:
    """Roll one or more dice and return the results.

    Use when the user asks for a dice roll or a random pick between numbers.

    Args:
        sides: Number of sides on each die.
        count: How many dice to roll.
    """
    if not 2 <= sides <= 1000 or not 1 <= count <= 20:
        raise SkillError("Use 2-1000 sides and 1-20 dice.")
    rolls = [random.randint(1, sides) for _ in range(count)]
    return {"rolls": rolls, "total": sum(rolls)}
