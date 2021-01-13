import pygame
from pygame.locals import *
import sys
import random
import tkinter as tk

pygame.init()

try:
    from screeninfo import get_monitors
    screenInfoMissing = False
except:
    screenInfoMissing = True


class Monitor():
    def __init__(self):
        self.width = 1300
        self.height = 900


class node():
    def __init__(self, x, y):
        self.neighbours = []
        self.x = x
        self.y = y
        self.top_left = (x*unit_width, y*unit_height)
        self.top_right = ((x+1)*unit_width, y*unit_height)
        self.bottom_left = (x*unit_width, (y+1)*unit_height)
        self.bottom_right = ((x+1)*unit_width, (y+1)*unit_width)
        self.middle = (((self.top_left[0] + self.top_right[0])/2),
                       ((self.top_left[1]+self.bottom_right[1])/2))
        self.set = y*x_cells + x

    def get_set(self):
        return self.set

    def set_set(self, s):
        self.set = s

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_coords(self):
        return (self.x, self.y)

    def get_top_left(self):
        return self.top_left

    def get_top_right(self):
        return self.top_right

    def get_bottom_left(self):
        return self.bottom_left

    def get_bottom_right(self):
        return self.bottom_right

    def get_middle(self):
        return self.middle

    def get_neighbours(self):
        return self.neighbours

    def add_neighbour(self, neighbour):
        self.neighbours.append(neighbour)
        neighbour.neighbours.append(self)

    def remove_neighbour(self, neighbour):
        neighbour.neighbours.remove(self)
        i = 0
        found = False
        while not found:
            if self.neighbours[i].get_x() == neighbour.get_x() and self.neighbours[i].get_y() == neighbour.get_y():
                self.neighbours.pop(i)
                found = True
            i += 1


class traversal_stack():
    def __init__(self):
        self.max_size = x_cells*y_cells
        self.stack = [None for nodes in range(self.max_size+1)]
        self.top = -1

    def add(self, to_be_added):
        self.top += 1
        self.stack[self.top] = to_be_added

    def pop_stack(self):
        if self.top != -1:
            to_remove = self.stack[self.top]
            self.top -= 1
            return to_remove

    def peek(self):
        return self.stack[self.top]

    def peek_below_top(self):
        if self.top != 0:
            return self.stack[self.top-1]

    def get_list(self):
        return self.stack

    def get_top(self):
        return self.top


class queue():
    def __init__(self):
        self.front = -1
        self.rear = 0
        self.size = 0
        self.max_size = int((x_cells*y_cells)/4)
        self.queue = [None for i in range(self.max_size)]

    def enqueue(self, to_add):
        if self.size != self.max_size:
            self.queue[self.rear] = to_add
            self.rear = (self.rear + 1) % self.max_size
            self.size += 1
        else:
            print("Error: queue full")

    def dequeue(self):
        if self.size != 0:
            self.front = (self.front+1) % self.max_size
            self.size -= 1
            return self.queue[self.front]
        else:
            print("Error: queue empty")


class menu(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        menu_window = tk.Frame(self)
        menu_window.grid(row=0, column=0)
        menu_window.grid_rowconfigure(0, weight=1)
        menu_window.grid_columnconfigure(0, weight=1)
        tk.Frame.__init__(menu_window, self)
        self.add_title()
        self.add_note()
        self.add_sliders()
        self.add_gen_drop_down()
        self.add_sol_drop_down()
        self.add_show_gen_button()
        self.add_show_sol_button()
        self.add_save_image_button()
        self.add_create_button()

    def add_title(self):
        label = tk.Label(self, text="Maze Generator", font=HEADING_FONT)
        label.grid(row=0, column=0, pady=5, padx=10)

    def add_note(self):
        label = tk.Label(self, text="""Choose algorithm, maze
            dimensions,\nwhether the maze should be shown\nduring
            generation and solving, \nand whether the image should be
            saved. \nIf it is to be saved, it will \nbe saved in the same
            folder as the program is.""", font=LABEL_FONT)
        label.grid(row=1, column=2)

    def add_sliders(self):
        self.x_slider = tk.Scale(self, from_=2, to=MAX_SIZE_X, orient="horizontal",
                                 label="Size: x", font=LABEL_FONT, length=int(MAX_SIZE_X*1.3))
        self.x_slider.grid(row=1, column=0)
        self.y_slider = tk.Scale(self, from_=2, to=MAX_SIZE_Y, orient="vertical",
                                 label="Size: y", font=LABEL_FONT, length=int(MAX_SIZE_Y*1.3))
        self.y_slider.grid(row=1, column=1)

    def add_gen_drop_down(self):
        algorithm_list_label = tk.Label(self, text="Generation Algorithm", font=LABEL_FONT)
        algorithm_list_label.grid(row=3, column=0, padx=5)
        self.algorithm = tk.StringVar(self)
        self.algorithm.set("Kruskal: Standard")
        self.algorithm_list = tk.OptionMenu(
            self, self.algorithm, "Kruskal: Standard", "Backtracker: Extra Windy", "Eller: Fast", "Divisor: Blocks")
        self.algorithm_list.configure(font=LABEL_FONT)
        self.algorithm_list.grid(row=4, column=0)

    def add_sol_drop_down(self):
        solve_algorithm_list_label = tk.Label(self, text="Solution Algorithm", font=LABEL_FONT)
        solve_algorithm_list_label.grid(row=3, column=1, padx=5)
        self.solve_algorithm = tk.StringVar(self)
        self.solve_algorithm.set("Depth First: Explore all paths")
        self.solve_algorithm_list = tk.OptionMenu(
            self, self.solve_algorithm, "Depth First: Explore all paths", "Breadth First: Spread out")
        self.solve_algorithm_list.configure(font=LABEL_FONT)
        self.solve_algorithm_list.grid(row=4, column=1)

    def add_show_gen_button(self):
        self.show_generation_var = tk.IntVar()
        self.show_generation_var.set(1)
        self.show_generation_button = tk.Checkbutton(
            self, text="Show Generation", font=LABEL_FONT, variable=self.show_generation_var)
        self.show_generation_button.grid(row=3, column=2)

    def add_show_sol_button(self):
        self.show_solve_var = tk.IntVar()
        self.show_solve_var.set(1)
        self.show_solve_button = tk.Checkbutton(
            self, text="Show Solving", font=LABEL_FONT, variable=self.show_solve_var)
        self.show_solve_button.grid(row=4, column=2)

    def add_save_image_button(self):
        self.save_image_var = tk.IntVar()
        self.save_image_var.set(1)
        self.show_solve_button = tk.Checkbutton(
            self, text="Save Image", font=LABEL_FONT, variable=self.save_image_var)
        self.show_solve_button.grid(row=2, column=2)

    def add_create_button(self):
        self.exited = False
        exit_button = tk.Button(self, text="Create Maze", command=self.enter, font=LABEL_FONT)
        exit_button.grid(row=5, column=2, pady=10, padx=7)

    def enter(self):
        self.x_cells = self.x_slider.get()
        self.y_cells = self.y_slider.get()
        generate_algorithm = self.algorithm.get()
        solve_algorithm = self.solve_algorithm.get()
        gen_algorithms = {"Kruskal: Standard": 1, "Backtracker: Extra Windy": 2,
                          "Eller: Fast": 3, "Divisor: Blocks": 4}
        self.gen_algorithm = gen_algorithms[generate_algorithm]
        solve_algorithms = {"Depth First: Explore all paths": 1, "Breadth First: Spread out": 2}
        self.sol_algorithm = solve_algorithms[solve_algorithm]
        self.show_generation = bool(self.show_generation_var.get())
        self.show_solving = bool(self.show_solve_var.get())
        self.save_image = bool(self.save_image_var.get())
        self.exited = True
        self.destroy()


BLACK = (0, 0, 0, 255)
SKYBLUE = (135, 206, 250, 255)
RED = (255, 0, 0, 255)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
SILVER = (192, 192, 192)
GREY = (128, 128, 128)
MAROON = (128, 0, 0)
OLIVE = (128, 128, 0)
GREEN = (0, 255, 0)
PURPLE = (128, 0, 128)
TEAL = (0, 128, 128)
NAVY = (0, 0, 128)

SAND = (237, 201, 175, 255)
LIMEGREEN = (191, 255, 0, 255)
YELLOW = (255, 255, 0, 255)
WHITE = (255, 255, 255, 255)
BLUE = (0, 0, 255, 255)
sys.setrecursionlimit(10000)
screen = None

HEADING_FONT = ("Source Code Pro", 18)
LABEL_FONT = ("Source Code Pro", 10)

MAX_SIZE_X = 100
MAX_SIZE_Y = 100

if screenInfoMissing:
    monitor = Monitor()
else:
    monitor = get_monitors().pop()

MAZE_COLOUR = WHITE
WALL_COLOUR = BLACK
NOT_IN_MAZE_COLOUR = RED
BEING_ADDED_COLOUR = GREEN
SOLUTION_COLOUR = RED
START_COLOUR = GREEN
END_COLOUR = SKYBLUE

plus_minus_one = [(0, 1), (0, -1), (1, 0), (-1, 0)]


def render_solution(surface, visited_list, num_items):
    for i in range(num_items):
        pygame.draw.line(surface, SOLUTION_COLOUR,
                         visited_list[i].get_middle(), visited_list[i+1].get_middle(), 1)


def renderMaze(surface, maze, stage, extra_data=None):
    surface.fill(MAZE_COLOUR)
    if stage == "recursive division":
        pygame.draw.rect(surface, BEING_ADDED_COLOUR, (maze[extra_data[0]][extra_data[1]].get_top_left()[
                         0], maze[extra_data[0]][extra_data[1]].get_top_left()[1], extra_data[2] * unit_width, extra_data[3] * unit_width))
    if stage == "Breadth first":
        for cell in extra_data:
            highlight_square(surface, cell, SOLUTION_COLOUR)
            for x in range(0, x_cells):
                for y in range(0, y_cells):
                    to_check = maze[x][y].get_neighbours()

        if stage == "recursive backtracker":
            if to_check == []:
                highlight_square(surface, maze[x][y], NOT_IN_MAZE_COLOUR)

        elif stage == "kruskal":
            highlight_square(surface, maze[x][y], kruskal_colour[maze[x][y].get_set()])

        elif stage == "eller":
            if to_check == []:
                highlight_square(surface, maze[x][y], NOT_IN_MAZE_COLOUR)

        else:
            if (x, y) == start:
                highlight_square(surface, maze[start[0]][start[1]], START_COLOUR)
            elif (x, y) == end:
                highlight_square(surface, maze[end[0]][end[1]], END_COLOUR)

        if maze[x-1][y] not in to_check:
            pygame.draw.line(surface, WALL_COLOUR,
                             maze[x][y].get_top_left(), maze[x][y].get_bottom_left(), 1)

        if maze[x][y-1] not in to_check:
            pygame.draw.line(surface, WALL_COLOUR, maze[x]
                             [y].get_top_left(), maze[x][y].get_top_right(), 1)
            pygame.draw.line(surface, WALL_COLOUR, (0, height-1), (width-1, height-1), 1)
            pygame.draw.line(surface, WALL_COLOUR, (width-1, 0), (width-1, height-1), 1)


def highlight_square(surface, cell, colour):
    pygame.draw.rect(surface, colour, (cell.get_top_left(), (unit_width, unit_height)))


def initialise(window_width, window_height, window_name, window_colour):
    pygame.init()
    screen = pygame.display.set_mode((window_width, window_height), 0, 32)
    pygame.display.set_caption(window_name)
    screen.fill(window_colour)
    return screen


def eventLoop():
    pygame.display.update()
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit


def generate_recursive_division_maze():
    print("Runnning recursive division")
    maze = [[node(x, y) for y in range(y_cells)] for x in range(x_cells)]
    if show_generation:
        renderMaze(screen, maze, "None")
        pygame.display.update()
    maze = recursive_division(maze, 0, 0, x_cells, y_cells)
    return maze


def recursive_division(maze, x, y, local_x_cells, local_y_cells):
    if local_x_cells != 1 or local_y_cells != 1:
        if local_x_cells <= 2:
            gap_x = x
        else:
            gap_x = x+random.randint(0, local_x_cells-2)
        if local_y_cells <= 2:
            gap_y = y
        else:
            gap_y = y+random.randint(0, local_y_cells-2)
        if show_generation:
            eventLoop()
            renderMaze(screen, maze, "recursive division", (x, y, local_x_cells, local_y_cells))
        if local_x_cells >= local_y_cells:
            maze = recursive_division(maze, x, y, gap_x-x+1, local_y_cells)
            if show_generation:
                eventLoop()
                renderMaze(screen, maze, "recursive division",
                           (x, y, local_x_cells, local_y_cells))
            maze = recursive_division(maze, gap_x+1, y, x+local_x_cells-gap_x-1, local_y_cells)
            maze[gap_x][gap_y].add_neighbour(maze[gap_x+1][gap_y])
        else:
            maze = recursive_division(maze, x, y, local_x_cells, gap_y-y+1)
            if show_generation:
                eventLoop()
                renderMaze(screen, maze, "recursive division",
                           (x, y, local_x_cells, local_y_cells))
            maze = recursive_division(maze, x, gap_y+1, local_x_cells, y+local_y_cells-gap_y-1)
            maze[gap_x][gap_y].add_neighbour(maze[gap_x][gap_y+1])
            return maze


def generate_recursive_backtracker_maze():
    print("Running recursive backtracker")
    maze = [[node(x, y) for y in range(y_cells)] for x in range(x_cells)]
    starting_y = random.randint(0, y_cells-1)
    starting_x = random.randint(0, x_cells-1)
    maze = recursive_backtracker(starting_x, starting_y, maze)
    return maze


def recursive_backtracker(x, y, maze):
    possible_adjacent = list(plus_minus_one)
    random.shuffle(possible_adjacent)
    for i in range(len(possible_adjacent)):
        checking = possible_adjacent[i]
        if (not (x == x_cells-1 and checking[0] == 1)) and (not(y == y_cells-1 and checking[1] == 1)) and (not(x == 0 and checking[0] == -1)) and (not(y == 0 and checking[1] == -1)):
            if show_generation:
                eventLoop()
                renderMaze(screen, maze, "recursive backtracker")
                highlight_square(screen, maze[x][y], BEING_ADDED_COLOUR)
            if maze[checking[0]+x][y+checking[1]].get_neighbours() == []:
                maze[x+checking[0]][y+checking[1]].add_neighbour(maze[x][y])
                maze = recursive_backtracker(x+checking[0], y+checking[1], maze)
                return maze


def generate_ellers_maze():
    print("Running Eller’s algorithm")
    maze = [[node(x, y) for y in range(y_cells)] for x in range(x_cells)]
    for x in range(x_cells-1):
        for y in range(y_cells-1):
            if random_boolean() and maze[x][y].get_set() != maze[x][y+1].get_set():
                maze = eller_add_vertical_neighbour(maze, x, y)
        for y in range(y_cells):
            if random_boolean():
                maze = eller_add_horizontal_neighbour(maze, x, y)
        maze = eller_guarantee_horizontal_connections(maze, x)
    x += 1
    for y in range(y_cells-1):
        if maze[x][y].get_set() != maze[x][y+1].get_set():
            maze = eller_add_vertical_neighbour(maze, x, y)
            return maze


def eller_guarantee_horizontal_connections(maze, x):
    sets = []
    y_coords = []
    for y in range(y_cells):
        current_considered = maze[x][y].get_set()
        if current_considered not in sets:
            sets.append(current_considered)
            y_coords.append([y])
        else:
            i = 0
            while sets[i] != current_considered:
                i += 1
                y_coords[i].append(y)
    for i in range(len(sets)):
        horizontal_connection = False
        j = 0
        while True:
            if maze[x+1][y_coords[i][j]] in maze[x][y_coords[i][j]].get_neighbours():
                horizontal_connection = True
                break
            j += 1
            if j == len(y_coords[i]):
                break
        if not horizontal_connection:
            to_be_connected = random.choice(y_coords[i])
            maze = eller_add_horizontal_neighbour(maze, x, to_be_connected)
            return maze


def random_boolean():
    return bool(random.getrandbits(1))


def eller_add_horizontal_neighbour(maze, x, y):
    maze[x][y].add_neighbour(maze[x+1][y])
    maze[x+1][y].set_set(maze[x][y].get_set())
    if show_generation:
        eventLoop()
        renderMaze(screen, maze, "eller")
        highlight_square(screen, maze[x][y], BEING_ADDED_COLOUR)
        highlight_square(screen, maze[x+1][y], BEING_ADDED_COLOUR)
    return maze


def eller_add_vertical_neighbour(maze, x, y):
    maze[x][y].add_neighbour(maze[x][y+1])
    maze = eller_merge_sets(maze, maze[x][y].get_set(), maze[x][y+1].get_set(), x)

    if show_generation:
        eventLoop()
        renderMaze(screen, maze, "eller")
        highlight_square(screen, maze[x][y], BEING_ADDED_COLOUR)
        highlight_square(screen, maze[x][y+1], BEING_ADDED_COLOUR)
    return maze


def eller_merge_sets(maze, set_to_be_added_to, set_to_be_merged, x):
    for y in range(y_cells):
        if maze[x][y].get_set() == set_to_be_merged:
            maze[x][y].set_set(set_to_be_added_to)
    return maze


def generate_kruskal_maze_v2():
    print("Running Kruskal’s algorithm")
    maze = [[node(x, y) for y in range(y_cells)] for x in range(x_cells)]
    possible_edges = find_possible_edges(maze)
    random.shuffle(possible_edges)
    i = 0
    number_of_sets = x_cells*y_cells
    sets = {}
    for x in range(x_cells):
        for y in range(y_cells):
            sets[maze[x][y].get_set()] = [maze[x][y]]
    while number_of_sets != 1:
        considered_edge = possible_edges[i]
        x1 = considered_edge[0].get_x()
        y1 = considered_edge[0].get_y()
        x2 = considered_edge[1].get_x()
        y2 = considered_edge[1].get_y()

        if maze[x1][y1].get_set() != maze[x2][y2].get_set():
            maze[x2][y2].add_neighbour(maze[x1][y1])
            set_to_be_added_to = maze[x2][y2].get_set()
            set_to_be_merged = maze[x1][y1].get_set()
            if show_generation:
                eventLoop()
                renderMaze(screen, maze, "kruskal")
                highlight_square(screen, maze[x1][y1], BEING_ADDED_COLOUR)
                highlight_square(screen, maze[x2][y2], BEING_ADDED_COLOUR)
        for i in range(len(sets[set_to_be_merged])):
            sets[set_to_be_merged][i].set_set(set_to_be_added_to)
            sets[set_to_be_added_to] += sets[set_to_be_merged]
            number_of_sets -= 1
            i += 1
        return maze


def find_possible_edges(maze):
    possible_edges = []
    for x in range(x_cells):
        for y in range(y_cells):
            if x != 0:
                possible_edges.append((maze[x][y], maze[x-1][y]))
            if y != 0:
                possible_edges.append((maze[x][y], maze[x][y-1]))
    return possible_edges


def depth_first_solve(maze):
    print("Running depth first search")
    start_x = start[0]
    start_y = start[1]
    visited = traversal_stack()
    visited.add(maze[start_x][start_y])
    visited, found = recursive_depth_first_solve(maze, visited)
    return visited.get_list(), visited.get_top()


def recursive_depth_first_solve(maze, visited):
    to_check = list(visited.peek().get_neighbours())
    try:
        to_check.remove(visited.peek_below_top())
    except:
        pass
    found = False
    i = 0
    while i < len(to_check) and (not found):
        visited.add(to_check[i])
        if show_solving:
            eventLoop()
            renderMaze(screen, maze, None)
            render_solution(screen, visited.get_list(), visited.get_top())
        if (to_check[i].get_x(), to_check[i].get_y()) == end:
            found = True
        else:
            visited, found = recursive_depth_first_solve(maze, visited)
            if not found:
                visited.pop_stack()
        i += 1
    return visited, found


def breadth_first_search(maze):
    print("Running breadth first search")
    found = False
    to_check = queue()
    to_check.enqueue((maze[start[0]][start[1]], None))
    visited = []

    while not found:
        checking = to_check.dequeue()
        visited.append(checking)
        if checking[0].get_x() == end[0] and checking[0].get_y() == end[1]:
            found = True
        else:
            for i in range(len(checking[0].get_neighbours())):
                if checking[1] is not None:
                    if checking[0].get_neighbours()[i] != visited[checking[1]][0]:
                        to_check.enqueue((checking[0].get_neighbours()[i], len(visited)-1))
                else:
                    to_check.enqueue((checking[0].get_neighbours()[i], len(visited)-1))
        if show_solving:
            eventLoop()
            renderMaze(screen, maze, "Breadth first", extra_data=[
                       visited[i][0] for i in range(len(visited))])
    route = [visited[len(visited)-1][0]]
    index = visited[len(visited)-1][1]
    while True:
        route.append(visited[index][0])
        if index == 0:
            break
        index = visited[index][1]
    return route, len(route)-1


def select_options():
    options_menu = menu()
    options_menu.mainloop()
    x_cells = options_menu.x_cells
    y_cells = options_menu.y_cells
    show_generation = options_menu.show_generation
    show_solving = options_menu.show_solving
    save_image = options_menu.save_image
    gen_choice = options_menu.gen_algorithm
    sol_choice = options_menu.sol_algorithm
    height = y_cells
    width = x_cells
    while height < int(int(monitor.height)) and width < int(int(monitor.width)):
        height = height * 2
        width = width * 2
    height = int(height/2)
    width = int(width/2)
    unit_width = int(width/x_cells)
    unit_height = int(height/y_cells)
    kruskal_colour = {}
    for x in range(x_cells):
        for y in range(y_cells):
            kruskal_colour[y*x_cells + x] = (255, x*255/x_cells, y*255/y_cells)
    start = (random.randint(0, x_cells-1), random.randint(0, y_cells-1))
    while True:
        end = (random.randint(0, x_cells-1), random.randint(0, y_cells-1))
        if (start[0]-end[0])**2+(start[1]-end[1])**2 >= (((x_cells+y_cells)/2)**2)/3:
            if end != start:
                break
    total_cells = x_cells*y_cells

    return x_cells, y_cells, show_generation, show_solving, save_image, gen_choice, sol_choice, height, width, unit_height, unit_width, kruskal_colour, start, end, total_cells


def generate_maze(show_generation, gen_choice):
    if gen_choice == 1:
        maze = generate_kruskal_maze_v2()
    elif gen_choice == 2:
        maze = generate_recursive_backtracker_maze()
    elif gen_choice == 3:
        maze = generate_ellers_maze()
    elif gen_choice == 4:
        maze = generate_recursive_division_maze()
    if show_generation:
        eventLoop()
        renderMaze(screen, maze, None)
    return maze


def solve_maze(sol_choice):
    if sol_choice == 1:
        visited, num_items = depth_first_solve(maze)
    elif sol_choice == 2:
        visited, num_items = breadth_first_search(maze)
    return visited, num_items


def save_images(visited, num_items, maze):
    image = pygame.Surface((width, height), 0, 24)
    image_solved = pygame.Surface((width, height), 0, 24)
    renderMaze(image, maze, None)
    renderMaze(image_solved, maze, None)
    render_solution(image_solved, visited, num_items)
    pygame.image.save(image, "maze.png")
    pygame.image.save(image_solved, "solved_maze.png")


x_cells, y_cells, show_generation, show_solving, save_image, gen_choice, sol_choice, height, width, unit_height, unit_width, kruskal_colour, start, end, total_cells = select_options()

print("x_cells:", x_cells, "\ny_cells:", y_cells,
      "\nshow_generation: ", show_generation, "\nshow_solving:",
      show_solving, "\nsave_image:", save_image)

if show_generation:
    screen = initialise(width, height, "Maze Generator", BLACK)
maze = generate_maze(show_generation, gen_choice)

if show_solving and not show_generation:
    screen = initialise(width, height, "Maze Generator", BLACK)
visited, num_items = solve_maze(sol_choice)

if show_generation or show_solving:
    renderMaze(screen, maze, None)
if show_solving:
    render_solution(screen, visited, num_items)

if save_image:
    save_images(visited, num_items, maze)

if show_generation or show_solving:
    while True:
        eventLoop()
