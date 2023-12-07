import tkinter as tk
import numpy as np
import parameters as p
import util as u
import pso as pSO
import testfuncts as tf
import consts as c
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm
from matplotlib.transforms import Bbox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.pyplot import subplots

FPS = 20 #How many frames per second the animation should run at
FRAME_MS = 1000//FPS #How many milliseconds each frame appears on screen
DPI = 100 #dots per inch that matplotlib should use for graphics

CONTOUR_DIM = Bbox(np.array([(0.05, 0.05), (0.95, 0.95)]))
COLOR_LEVELS = 100

class Particle:
    """
    Keeps track of the visual representation of a particle.

    delta_x:  amount particle should move per animation frame in x direction
    delta_y:  amount particle should move per animation frame in y direction
    text:  tkinter ID of the text label which shows the coordinates of the particle
    dot:  tkinter ID of the circle which represents the particle
    """
    delta_x: int
    delta_y: int

    text: int
    dot: int

    def __init__(self, text, dot):
        self.text = text
        self.dot = dot

class Visualization:
    """
    Runs a visual representation of test functions.

    width:  width of visualization screen
    height:  height of visualization screen
    pso:  PSO instance which this visualizer controls
    particles:  Visual representation of the particles from PSO
    
    root:  tkinter root
    control_menu:  frame for top bar, which allows user to control the visualization
    display:  frame which visually displays the PSO algorithm
    part_canv:  Particle canvas.  Shows the contour plot and PSO particles.
    contour_img:  Contour plot of the function to be optimized.  Used as a background for part_canv
    contour_loc:  Location of the contour plot within the figure.  Used to ensure particles stay within the contour plot.
    
    update_time:  How many milliseconds each iteration of PSO should last.  Default is 1 second.
    contour_img_path:  Path to temporary image of the contour diagram.
    """

    #Width and height of the visualization screen
    width: int
    height: int

    #Internal data about PSO
    pso: pSO.PSO
    particles: list
    g_best: int

    #Various elements of the visualization
    root: tk.Tk
    control_menu: tk.Frame
    display: tk.Frame
    part_canv: tk.Canvas
    contour_img: tk.PhotoImage
    contour_loc: Bbox

    #Misc data
    update_time: int
    contour_img_path: str
    
    def __init__(self, root: tk.Tk, pso: pSO.PSO, update_time: int = 1000):
        u.clear_temp() #Clear temporary png files
        self.update_time = update_time - update_time % FRAME_MS # Ensures that an iteration of PSO can evenly be divided into frames
        self.root = root
        self.pso = pso
        self.contour_img_path = u.make_tempfile_path() # Path to the contour image that will be saved later.

        self.root.title("Visualization")
        # Make the visualization full screen
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

        self.make_contour() # Generate contour plot, and set as background of particle canvas
        self.make_gbest_iterations()
        #root.destroy()

        # Num_iterations vs. g_best

        self.root.update_idletasks()

    def start(self):
        """Initializes PSO, and begins the visualization"""
        self.pso.initialize()

        # Map domain coordinates to the canvas
        coords = self._translate_coords(self.pso.pos_matrix)
        
        # Vector of g_best coordinates translated to canvas coordinates
        coordsgb = self._translate_coords(self.pso.g_best[:-1])

        gbx = coordsgb[c.XDIM]
        gby = coordsgb[c.YDIM]
        
        self.g_best = self.part_canv.create_oval(gbx, gby, gbx, gby, outline="red", fill="red", width=15)
        
        # Based on particles 
        self.particles = []
        for i in range(coords.shape[1]):
            canvas_x = coords[c.XDIM][i]
            canvas_y = coords[c.YDIM][i]
            space_x = np.round(self.pso.pos_matrix[c.XDIM][i], 2)
            space_y = np.round(self.pso.pos_matrix[c.YDIM][i], 2)
            text = self.part_canv.create_text(canvas_x, canvas_y-10, text=f"({space_x}, {space_y})", fill="black", font=("Times New Roman", 10))
            dot = self.part_canv.create_oval(canvas_x, canvas_y, canvas_x, canvas_y, outline="blue", fill="blue", width = 10)
            particle = Particle(text, dot)
            self.particles.append(particle)
        
        self.root.update()
        self.root.after(FRAME_MS, self.update_particles)

    def update_particles(self):
        shouldTerminate = self.pso.update()

        # Map domain coordinates to the canvas
        coords = self._translate_coords(self.pso.pos_matrix)
        
        # Vector of g_best coordinates translated to canvas coordinates
        coordsgb = self._translate_coords(self.pso.g_best[:-1])

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

        # Update the g_best
        gbx = coordsgb[c.XDIM]
        gby = coordsgb[c.YDIM]
        self.part_canv.moveto(self.g_best, gbx, gby)
        
        if not shouldTerminate:
            self.root.update()
            self.root.after(0, self.update_particles)
        else:
            print(f"The best position was {self.pso.g_best[:-1]} with a value of {self.pso.g_best[-1]}")

    def make_gbest_iterations(self):
        self.root.update()
        self.root.update_idletasks()
        fig, ax = subplots()
        ax.plot([1, 2, 3, 4], [1, 2, 3, 4])
        canvas = FigureCanvasTkAgg(fig, master=self.display)
        canvas.draw()
        canvas.get_tk_widget().grid(row=1, column=0)
        self.root.update()
        self.root.update_idletasks()

    def make_contour(self):
        # Particle canvas setup.

        self.part_canv = tk.Canvas(self.display, width = self.display.winfo_width(), height=self.display.winfo_height())
        self.part_canv.grid(row=0, column=0)

        self.root.update_idletasks()
        x_bounds, y_bounds = u.dimension_to_xy_bounds(self.pso.lower_bound, self.pso.upper_bound)
        
        x, y, z = tf.TF.generate_contour(self.pso.function, self.pso.lower_bound, self.pso.upper_bound)
        fig = Figure(figsize=(self.part_canv.winfo_width()/DPI, self.part_canv.winfo_height()/DPI), dpi=DPI) #Make a figure object

        # Make the figure take up as much space as possible
        ax = fig.add_subplot(111)
        
        ax.set_position(CONTOUR_DIM)
        self.contour_loc = ax.get_position() # Store this for mapping purposes later

        # Apply a logarithmic scale to the colors so that the minima takes more colors
        levels = np.logspace(np.log10(np.min(z)),np.log10(np.max(z)),COLOR_LEVELS)

        # Make the contour
        ax.contourf(x, y, z, cmap="viridis",  
                    extent=[x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]],
                    norm=LogNorm(vmin=np.min(z), vmax=np.max(z)), levels=levels) 
        
        fig.savefig(self.contour_img_path) #Save file for use as a background
        self.contour_img = tk.PhotoImage(file=self.contour_img_path+".png")
        self.part_canv.create_image(0, 0, anchor=tk.NW, image=self.contour_img)
        
        self.root.update()
        self.root.update_idletasks()

    def _translate_coords(self, array: p.ADTYPE):
        """ Function which takes an array of values in a domain, and maps them to their appropriate posiiton on the canvas."""
        if not u.check_dim(array, 2):
            raise Exception("not a two by two")
        
        cwidth = self.part_canv.winfo_width()
        cheight = self.part_canv.winfo_height()

        # The actual contour plot only takes up xmin - xmax and ymin-max of the figure.  Therefore, we multiply cwidth and cheight
        # Respectively so that all particles show up on the contour plot.
        new_lb = np.array((cwidth*self.contour_loc.xmin, cheight*self.contour_loc.ymin), dtype=p.DTYPE)
        new_ub = np.array((cwidth*self.contour_loc.xmax, cheight*self.contour_loc.ymax), dtype=p.DTYPE)
        coords = u.project(self.pso.lower_bound, self.pso.upper_bound, new_lb, new_ub, array) #Coordinates on the tkinter canvas

        # Tkinter uses an odd scheme where y increases down, instead of up for coordinates.  This swapping of the y
        # axis conforms to the coordinate system of tkinter.
        dims = array.ndim
        if dims == 2:
            coords[1] = np.ones((1, coords.shape[1]))*cheight - coords[1]
        if dims == 1:
            coords[1] = cheight - coords[1]
        return coords


def TestVisualizer():
    root = tk.Tk()
    pso = pSO.PSO(num_part = p.NUM_PART, num_dim=p.NUM_DIM, alpha = p.ALPHA, upper_bound=p.UPPER_BOUND, lower_bound=p.LOWER_BOUND,
    max_iterations=p.MAX_ITERATIONS, w=p.W, c1=p.C1, c2=p.C2, tolerance=p.TOLERANCE, mv_iteration=p.NO_MOVEMENT_TERMINATION,
    optimum=p.OPTIMUM, bias=p.BIAS, functionID = p.FUNCT)

    vis = Visualization(root=root, pso=pso, update_time = 500)
    
    vis.start()
    root.mainloop()


TestVisualizer()
