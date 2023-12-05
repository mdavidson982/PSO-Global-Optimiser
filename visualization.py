import tkinter as tk
import numpy as np
import parameters as p
import util as u
import pso as pSO
import testfuncts as tf
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm

FPS = 20
FRAME_MS = 1000//FPS
DPI = 100 #dots per inch

def _translateCoords(lb: p.ADTYPE, ub: p.ADTYPE, xpos: p.ADTYPE, canvas_width: int, canvas_height: int):

    if not u.check_dim(xpos, 2):
        raise Exception("not a two by two")

    new_lb = np.array((0, 0), dtype=p.DTYPE)
    new_ub = np.array((canvas_width, canvas_height), dtype=p.DTYPE)
    coords = u.project(lb, ub, new_lb, new_ub, xpos)

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

    #Width and height of the visualization screen
    width: int
    height: int

    #Internal data about PSO
    pso: pSO.PSO
    particles: list

    #Various elements of the visualization
    root: tk.Tk
    control_menu: tk.Frame
    display: tk.Frame
    part_canv: tk.Canvas
    contour_img: tk.PhotoImage

    #Misc data
    update_time: int
    contour_img_path: str
    
    def __init__(self, root: tk.Tk, pso: pSO.PSO, update_time: int = 1000):
        u.clear_temp() #Clear temporary png files
        self.update_time = update_time - update_time % FRAME_MS #make the update time a multiple of 25 for convenience
        self.root = root
        self.pso = pso
        self.contour_img_path = u.make_tempfile_path()

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

        self.part_canv = tk.Canvas(self.display, width = self.display.winfo_width(), height=self.display.winfo_height())
        self.part_canv.pack(fill=tk.BOTH, expand=True)
        self.root.update_idletasks()
        self.make_contour()
        self.part_canv.create_image(0, 0, anchor=tk.NW, image=self.contour_img)

        self.root.update_idletasks()

    def start(self):
        self.pso.initialize()

        coords = _translateCoords(self.pso.lower_bound, self.pso.upper_bound, self.pso.pos_matrix, self.part_canv.winfo_width(), self.part_canv.winfo_height())
        self.particles = []

        for i in range(coords.shape[1]):
            canvas_x = coords[0][i]
            canvas_y = coords[1][i]
            space_x = np.round(self.pso.pos_matrix[0][i], 2)
            space_y = np.round(self.pso.pos_matrix[1][i], 2)
            text = self.part_canv.create_text(canvas_x, canvas_y-10, text=f"({space_x}, {space_y})", fill="black", font=("Times New Roman", 10))
            dot = self.part_canv.create_oval(canvas_x, canvas_y, canvas_x, canvas_y, outline="blue", fill="blue", width = 10)
            particle = Particle(text, dot)
            self.part_canv.tag_raise(particle)
            
            self.particles.append(particle)

        self.root.update()
        self.root.after(FRAME_MS, self.update_particles)

    def make_contour(self):
        x_bounds, y_bounds = u.dimension_to_xy_bounds(self.pso.lower_bound, self.pso.upper_bound)
        x, y, z = tf.TF.generate_contour(self.pso.functionID, self.pso.optimum, self.pso.bias, self.pso.lower_bound, self.pso.upper_bound)
        fig = Figure(figsize=(self.part_canv.winfo_width()/DPI, self.part_canv.winfo_height()/DPI), dpi=DPI) #Make a figure object

        ax = fig.add_subplot(111)

        levels = np.logspace(np.log10(np.min(z)),np.log10(np.max(z)),100)

        contour = ax.contourf(x, y, z, cmap="viridis",  
                              extent=[x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]],
                              norm=LogNorm(vmin=np.min(z), vmax=np.max(z)), levels=levels
                              ) #Make the contour
        #fig.colorbar(contour)
        
        #The following removes whitespace
        #ax.axis('off')
        #fig.gca().set_axis_off()
        fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        
        fig.savefig(self.contour_img_path) #Save file for use as a background
        self.contour_img = tk.PhotoImage(file=self.contour_img_path+".png")

    def update_particles(self):
        
        shouldTerminate = self.pso.update()
        coords = _translateCoords(self.pso.lower_bound, self.pso.upper_bound, self.pso.pos_matrix, self.part_canv.winfo_width(), self.part_canv.winfo_height())
        
        steps = self.update_time // FRAME_MS
        for i in range(len(self.particles)):
            canvas_x = coords[0][i]
            canvas_y = coords[1][i]
            space_x = np.round(self.pso.pos_matrix[0][i], 2)
            space_y = np.round(self.pso.pos_matrix[1][i], 2)
            particle: Particle = self.particles[i]
            
            self.part_canv.itemconfig(particle.text, text=f"({space_x}, {space_y})", fill="black")
            particle_coords = self.part_canv.coords(particle.dot)
            
            particle.delta_x = canvas_x - particle_coords[0]
            particle.delta_y = canvas_y - particle_coords[1]

        # Move the particles to the new location smoothly
        for _ in range(steps):
            for i in range(len(self.particles)):
                particle = self.particles[i]

                self.part_canv.move(particle.dot, particle.delta_x/steps, particle.delta_y/steps)
                particle_coords = self.part_canv.coords(particle.dot)
                self.part_canv.moveto(particle.text, particle_coords[0], particle_coords[1])
            self.root.after(FRAME_MS)
            self.root.update()

        if not shouldTerminate:
            self.root.update()
            self.root.after(0, self.update_particles)


def TestVisualizer():
    root = tk.Tk()
    pso = pSO.PSO(num_part = p.NUM_PART, num_dim=p.NUM_DIM, alpha = p.ALPHA, upper_bound=p.UPPER_BOUND, lower_bound=p.LOWER_BOUND,
    max_iterations=p.MAX_ITERATIONS, w=p.W, c1=p.C1, c2=p.C2, tolerance=p.TOLERANCE, mv_iteration=p.NO_MOVEMENT_TERMINATION,
    optimum=p.OPTIMUM, bias=p.BIAS, functionID = p.FUNCT)

    vis = Visualization(root=root, pso=pso, update_time = 10000)
    vis.start()
    root.mainloop()


TestVisualizer()
