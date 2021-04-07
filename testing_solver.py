from rubik.cube import Cube
from rubik.solve import Solver

from rubik_solver import utils
c = Cube("OOOOOOOOOYYYWWWGGGBBBYYYWWWGGGBBBYYYWWWGGGBBBRRRRRRRRR")

# c = Cube("WWWWWWWWWBBBOOOGGGRRROOOGGGRRRBBBOOOGGGRRRBBBYYYYYYYYY")
# c = Cube("WWWWWWWWWOOOGGGRRRBBBOOOGGGRRRBBBOOOGGGRRRBBBYYYYYYYYY")

print(c.is_solved())
solver = Solver(c)
solver.solve()
print(solver.moves)

print(c.is_solved())


# cube = 'wwwwwwwwwbbbooooooooogggggggggrrrrrrrrrbbbbbbyyyyyyyyy'
# solved = utils.solve(cube, 'Beginner')

# print(solved)
