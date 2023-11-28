import tkinter as tk
import numpy as np
import parameters as p
import visualizationParams as vp
import util as u
import initializer as i

def _checkDim(array: p.ADTYPE):
    return array.shape[0] == 2

def _translateCoords(lb: p.ADTYPE, ub: p.ADTYPE, xpos: p.ADTYPE, canvas_width: int, canvas_height: int):

    if not _checkDim(xpos):
        raise Exception("not a two by two")

    new_lb = np.array((0, 0), dtype=p.DTYPE)
    new_ub = np.array((canvas_width, canvas_height), dtype=p.DTYPE)
    coords = u.Project(lb, ub, new_lb, new_ub, xpos)

    # Tkinter uses an odd scheme where y increases down, instead of up for coordinates.  This swapping of the y
    # axis conforms to the coordinate system of tkinrer.
    coords[1] = np.ones((1, coords.shape[1]))*canvas_height - coords[1]
    return coords

class Visualization:

    lb: p.ADTYPE
    ub: p.ADTYPE
    root: tk.Tk
    width: int
    height: int

    particles: list
    
    def __init__(self, root: tk.Tk):
    
        self.root = root
        self.root.title("Visualization")
        self.width = self.root.winfo_screenwidth()
        self.height = self.root.winfo_screenheight()
        self.root.geometry(f"{self.width}x{self.height}")

        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack()

        self.control_menu = tk.Frame(self.main_frame, width=int(self.width*.8), height=(int(self.height*.15)), highlightbackground="blue", highlightthickness=3)
        self.control_menu.pack(pady=10)

        self.display = tk.Frame(self.main_frame, width=int(self.width*.8), height = (int(self.height*.75)), highlightbackground="black", highlightthickness=5)
        self.display.pack(pady=10)
        self.root.update_idletasks()

        self.canvas = tk.Canvas(self.display, width = self.display.winfo_width(), height=self.display.winfo_height())
        self.canvas.pack()
        self.root.update_idletasks()

    def start(self, xpos: p.ADTYPE, lower_bounds: p.ADTYPE, upper_bounds: p.ADTYPE):
        self.lb = lower_bounds
        self.ub = upper_bounds
        coords = _translateCoords(self.lb, self.ub, xpos, self.canvas.winfo_width(), self.canvas.winfo_height())
        self.particles = []

        for i in range(coords.shape[1]):
            x = coords[0][i]
            y = coords[1][i]
            particle = self.canvas.create_oval(x, y, x, y, outline="blue", fill="blue", width = 10)
            self.particles.append(particle)
        self.root.update()

    def updateParticles(self, xpos: p.ADTYPE):
        self.root.update_idletasks()
        coords = _translateCoords(self.lb, self.ub, xpos, self.canvas.winfo_width(), self.canvas.winfo_height())
        for i in range(len(self.particles)):
            x = coords[0][i]
            y = coords[1][i]
            particle = self.particles[i]
            self.canvas.moveto(particle, x, y)
        self.root.update()
        self.root.after(10)


def TestVisualizer():
    lb = np.array((0, 0))
    ub = np.array((5, 5))

    pos_matrix, _, _, _, _ = i.initializer(10, 2, p.ALPHA, ub, lb)

    Visualization(lb, ub, pos_matrix)
