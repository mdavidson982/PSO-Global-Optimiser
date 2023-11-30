import tkinter as tk
import numpy as np
import parameters as p
import util as u
import pso as pSO
import testfuncts as tf

FPS = 20
FRAME_MS = 1000//FPS

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

class Particle:
    delta_x: int
    delta_y: int

    text: int
    dot: int

    def __init__(self, text, dot):
        self.text = text
        self.dot = dot

        

class Visualization:

    root: tk.Tk
    width: int
    height: int

    pso: pSO.PSO
    particles: list

    update_time: int
    
    def __init__(self, root: tk.Tk, pso: pSO.PSO, update_time: int = 1000):
        self.update_time = update_time - update_time % FRAME_MS #make the update time a multiple of 25 for convenience
        self.root = root
        self.root.title("Visualization")
        self.width = self.root.winfo_screenwidth()
        self.height = self.root.winfo_screenheight()
        self.root.geometry(f"{self.width}x{self.height}")

        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack()

        self.control_menu = tk.Frame(self.main_frame, width=int(self.width*.8), height=(int(self.height*.15)), highlightbackground="blue", highlightthickness=3)
        self.control_menu.pack(pady=10)

        self.root.update_idletasks()
        self.display = tk.Frame(self.main_frame, width=int(self.width*.8), height = (int(self.height*.75)), highlightbackground="black", highlightthickness=5)
        self.display.pack(pady=10)
        self.root.update_idletasks()

        self.canvas = tk.Canvas(self.display, width = self.display.winfo_width(), height=self.display.winfo_height())
        self.canvas.pack()
        self.root.update_idletasks()

        self.pso = pso

    def start(self):
        self.pso.initialize()

        coords = _translateCoords(self.pso.lower_bound, self.pso.upper_bound, self.pso.pos_matrix, self.canvas.winfo_width(), self.canvas.winfo_height())
        self.particles = []

        for i in range(coords.shape[1]):
            canvas_x = coords[0][i]
            canvas_y = coords[0][i]
            space_x = np.round(self.pso.pos_matrix[0][i], 2)
            space_y = np.round(self.pso.pos_matrix[1][i], 2)
            text = self.canvas.create_text(canvas_x, canvas_y-10, text=f"({space_x}, {space_y})", fill="black", font=("Times New Roman", 10))
            dot = self.canvas.create_oval(canvas_x, canvas_y, canvas_x, canvas_y, outline="blue", fill="blue", width = 10)
            particle = Particle(text, dot)
            
            self.particles.append(particle)
        self.root.update()
        self.root.after(FRAME_MS, self.updateParticles)

    def updateParticles(self):
        
        shouldTerminate = self.pso.update()
        coords = _translateCoords(self.pso.lower_bound, self.pso.upper_bound, self.pso.pos_matrix, self.canvas.winfo_width(), self.canvas.winfo_height())
        
        steps = self.update_time // FRAME_MS
        for i in range(len(self.particles)):
            canvas_x = coords[0][i]
            canvas_y = coords[1][i]
            space_x = np.round(self.pso.pos_matrix[0][i], 2)
            space_y = np.round(self.pso.pos_matrix[1][i], 2)
            particle: Particle = self.particles[i]
            
            self.canvas.itemconfig(particle.text, text=f"({space_x}, {space_y})", fill="black")
            particle_coords = self.canvas.coords(particle.dot)
            
            particle.delta_x = canvas_x - particle_coords[0]
            particle.delta_y = canvas_y - particle_coords[1]

        for _ in range(steps):
            for i in range(len(self.particles)):
                particle = self.particles[i]

                self.canvas.move(particle.dot, particle.delta_x/steps, particle.delta_y/steps)
                particle_coords = self.canvas.coords(particle.dot)
                self.canvas.moveto(particle.text, particle_coords[0], particle_coords[1])
            self.root.after(FRAME_MS)
            self.root.update()

        if not shouldTerminate:
            self.root.update()
            self.root.after(0, self.updateParticles)


def TestVisualizer():
    Func = tf.rosenbrockGenerator(p.OPTIMUM)
    root = tk.Tk()
    pso = pSO.PSO(num_part = p.NUM_PART, num_dim=p.NUM_DIM, alpha = p.ALPHA, upper_bound=p.UPPER_BOUND, lower_bound=p.LOWER_BOUND,
    max_iterations=p.MAX_ITERATIONS, w=p.W, c1=p.C1, c2=p.C2, tolerance=p.TOLERANCE, mv_iteration=p.NO_MOVEMENT_TERMINATION,
    function = Func)

    vis = Visualization(root=root, pso=pso, update_time = 1000)
    vis.start()
    root.mainloop()


TestVisualizer()
