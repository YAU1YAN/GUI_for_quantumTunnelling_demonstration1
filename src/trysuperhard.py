import numpy as np
from matplotlib import pyplot as pl
from matplotlib import animation
from scipy.fftpack import fft,ifft    #special function imported to aid the solving of time-dependent schrodinger equation.
import Tkinter as tk
import webbrowser
import tkMessageBox as box
import schrodinger  #unique schrodinger equation solver at all time, space and potential levels
# first define functions that would be used all throught out the classes
def theta(x):   # this together with square barrier, work together to construct the potential step in this investigation
    """
    theta function :
      returns 0 if x<=0, and 1 if x>0
    """
    x = np.asarray(x)
    y = np.zeros(x.shape)
    y[x > 0] = 1.0
    return y
def gauss_x(x, a, x0, k0):     #this is a standard probability distribution of a singular electron which we will see it move when applied to the TDSE
    """
    a gaussian wave packet of width a, centered at x0, with momentum k0
    """ 
    return ((a * np.sqrt(np.pi)) ** (-0.5)
            * np.exp(-0.5 * ((x - x0) * 1. / a) ** 2 + 1j * x * k0))

def square_barrier(x, width, height):
    return height * (theta(x) - theta(x - width))


#-------------------------------------------------------------------------------------------------
# Set Fonts for Titles and Subtitles
TITLE_FONT = ("Helvetica", 18, "bold italic")
SUBTITLE_FONT = ("Times", 12, "italic" )
MID_FONT=("Times",16)
TINY_FONT=("Helvetica",8,"bold italic")
# Create super class that manages the Tkinker GUI
class Main(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        
        # the container is where we'll stack a bunch of frames
        # on top of each other, then the one we want visible
        # will be raised above the others
        container = tk.Frame(self)
        container.pack(side="top", fill="both",expand=True)
        container.grid_rowconfigure(1, weight=0)
        container.grid_columnconfigure(1, weight=0)
        
        # List of frames
        self.frames = {}
        for F in (StartPage, Info,Sph_harm, Advance, Quit):
            frame = F(container, self)
            self.frames[F] = frame
            # put all of the pages in the same location; 
            # the one on the top of the stacking order
            # will be the one that is visible.
            frame.grid(row=1, column=1, sticky="nsew")
             
        self.show_frame(StartPage)
        self.title("The Quantum Tunneling") # Title of the main frame
        
        # Create help bar that permeat through all frames
        menubar = tk.Menu(self)
        self.config(menu=menubar)
        fileMenu = tk.Menu(menubar)
        menubar.add_cascade(label="Program", menu=fileMenu)
        fileMenu.add_command(label="Help", command=self.ProgInfo)
        
    def ProgInfo(self):  #text installed inside the help bar
        box.showinfo("Program Description", """The program was created to to generate animation of the Physical concept of quantum tunneling famously resulted from the schrodinger equation.This interface allow user to investigate the phenomonon of quantum particle going into a potential barrier undervarious different condition such as variation of initial momentum and energy.All the plots are generated using matplotlib.""")  

    def show_frame(self, c):
        '''Show a frame for the given class'''
        frame = self.frames[c]
        frame.tkraise()    
        
class StartPage(tk.Frame): #decoration of the starting page with words and buttons
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent) 
        label1 = tk.Label(self, text="Main Menu", font=TITLE_FONT)
        label1.pack(side="top", fill="both", pady=40, padx=30)
        
        label2 = tk.Label(self,text = "Please choose one of the following options (preferably Action!!! before ", 
                                font = SUBTITLE_FONT)
        label2.pack()

        button1 = tk.Button(self, text="Info about quantum tunneling", activebackground= 'green',bd=5, 
                            command=lambda: controller.show_frame(Info))
        button2 = tk.Button(self, text="Action!!!!!",activebackground= 'green',bd=5,
                            command=lambda: controller.show_frame(Sph_harm))
        button3 = tk.Button(self, text="Advance action",activebackground= 'green',bd=5,
                            command=lambda: controller.show_frame(Advance))
        button4 = tk.Button(self, text="Quit Program",activebackground= 'green',bd=5,
                            command=lambda: controller.show_frame(Quit))                            
        button1.pack()
        button2.pack()
        button3.pack()
        button4.pack()


class Info(tk.Frame):  #a sub page that leads you to a introductory wiki page
    
    
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Info about quantum tunneling", activebackground= 'green',bd=5,
                                font=TITLE_FONT)
        label.pack(side="top", fill="both", pady=10)
        button1 = tk.Button(self, text="Hyperlink", activebackground= 'green',bd=5, 
                           command=self.OpenUrl)
        button2 = tk.Button(self, text="Go back to the Main Menu",activebackground= 'green',bd=5,
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()
        button2.pack(side="bottom")
        
    def OpenUrl(self):
        url = 'http://en.wikipedia.org/wiki/Quantum_tunnelling'
        webbrowser.open_new(url)
        
class Sph_harm(tk.Frame): #main subpage where the use interaction happens
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Tunneling Action", font=TITLE_FONT)    #decorating the page with button, texts
        label.pack(side="top", fill="both", pady=10)
        label2 = tk.Label(self, text="""Welcome! here you can pick your electrons incidence momentum(P) and potetial barrier(V0),
            please pay close attention the energy level of the electron(derive from selected momentum vs the selected barrier,
            the incidence momentum dictate its energy P^2/2m (mass of electron is set as one), if the energy is higher than V0, 
            more transmission will occur over reflection, to begin I recommed 1.8 momentum vs 1 potential """ , font=SUBTITLE_FONT)
        label2.pack()        
        L1 = tk.Label(self, text='Please Choose the electrons momentum Using the Slider:',font=MID_FONT) #the total sets of scroller users can use to adjust input variables
        L1.pack()
        self.S1 = tk.Scale(self, orient = tk.HORIZONTAL, from_=1, to=4,resolution=0.1, bd=6)
        self.S1.pack()
        L2 = tk.Label(self, text='Please Choose potential barrier size:',font=MID_FONT)
        L2.pack()
        self.S2 = tk.Scale(self, orient = tk.HORIZONTAL, from_=1, to=4,resolution=0.1, bd=6)#total sets of buttons and their corresponding position
        self.S2.pack()
        
        button2 = tk.Button(self, text="Go back to the Main Menu",activebackground= 'green',bd=5,
                            command=lambda: controller.show_frame(StartPage))
        button2.pack(side="bottom")
        button3 = tk.Button(self, text="!!Click me before the action!!", activebackground= 'green',bd=6, 
                           command=self.info) 
    
        button3.pack()
        button4 = tk.Button(self, text="action now!", activebackground= 'green',bd=6, 
                           command=self.action) 
    
        button4.pack()
        #All the independent variables
        #creating time frames
        self.dt = 0.04
        self.N_steps = 50
        self.t_max = 120
        self.frames = int(self.t_max / float(self.N_steps * self.dt)) 
        
        #make the x axis with 6800 distinct points ranging from -340 to +340
        self.N = 6800
        self.dx = 0.1
        self.x = self.dx * (np.arange(self.N) - 0.5 * self.N)
      
        #arbitary constant( electrons mass, planks constant, potential step level)
        self.m=1.0
        self.hbar=1.0
        self.V0=1.
        
        #Geometry of initial setup (initial pos
        self.x0 = -100.
        self.barrierwidth=200
        
        self.V_x = square_barrier(self.x,self.barrierwidth,self.V0 )

        #the properties of such an electron
        self.d = 10
        self.k0=1
        self.psi_x0 = gauss_x(self.x,self.d,self.x0,self.k0)
        
    def action(self): #actual ploting of the all of the graph based on variables from the scrolls 
        self.V0=float(self.S2.get())
        self.barrierwidth=200        
        self.V_x = square_barrier(self.x,self.barrierwidth,self.V0 )
        self.k0=float(self.S1.get())
        self.psi_x0 = gauss_x(self.x,self.d,self.x0,self.k0)
        self.S = schrodinger.Schrodinger(x=self.x,
                psi_x0=self.psi_x0,
                V_x=self.V_x,
                hbar=self.hbar,
                m=self.m,
                k0=-28)
        # Set up plot
        self.fig = pl.figure()

        # plotting limits
        self.xlim = (-200, 200)

        # set up the axis, and some empty plot ready to be filled 

        self.ax1 = self.fig.add_subplot(111, xlim=self.xlim,
                            ylim=(0,
                                    self.V0 + 0.3))
        self.psi_x_line, = self.ax1.plot([], [], c='b', label=r'$|\psi(x)|$') #this will become the wave graph
        self.V_x_line, = self.ax1.plot([], [], c='r', label=r'$V(x)$')   #this will become the red potential step graph


        self.title = self.ax1.set_title("at time t")
        self.ax1.legend(prop=dict(size=12))
        self.ax1.set_xlabel('$x$')
        self.ax1.set_ylabel(r'$|\psi(x)|$')
        
        def init(): #the intial frame on the animation
            self.psi_x_line.set_data([], [])
            self.V_x_line.set_data([], [])
            self.title.set_text("click with mover to find out current time ")
            return (self.psi_x_line, self.V_x_line, self.title)

        def animate(i): #the subsquent frame of the animation
            self.S.time_step(self.dt, self.N_steps) #this feed and refresh the the wavefunction base every time instances from t=0 to infinity
            self.psi_x_line.set_data(self.S.x,  2*abs(self.S.psi_x)) 
            self.V_x_line.set_data(self.S.x, self.S.V_x)
            self.title.set_text("Tunelling of an electron @ t = %.2f" % self.S.t)
            if self.S.t > 500.:
                self.psi_x_line.set_data([], [])  #this stops the plot once t reaches 500, to prevent CPU usage when multiple animation

            return (self.psi_x_line, self.V_x_line, self.title)
        
        self.anim = animation.FuncAnimation(self.fig, animate, init_func=init,
                               frames=self.frames, interval=10, blit=True)
        pl.show()
                               
    def info(self): # this is responsible for the info box when you click "click me before action button"
        self.n = float(self.S1.get())
        box.showinfo("input energy(abitary):",(self.n**2)/2.)
        box.showinfo("Bug description:",'''The following may happen to you:
        The frame around the animation is blacked out?        Minimize then Maximize the figure window.
        
        The title is supposed to update you with the current time(arbitary) since intitiation. If not?         Use the fourth tool from the left("the cross move"), click rapidly with the special move cursor (please do that, it takes me a long time to make the title update itself)
        
        IF the animation is too slow?      Use a faster computer or be patient, yes, I have tried reducing the frame renew rate, it works only on good computer
        
        !!!Don't close the animation too soon after its has initiated, or it will crash and there will be much wailing and gnashing of teeth
        
        Have Fun :)
        ''')

class Advance(tk.Frame): # this does almost the same thing as above but with more variable in users control
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Tunneling Action", font=TITLE_FONT)
        label.pack(side="top", fill="both", pady=10)
        label2 = tk.Label(self, text="""Welcome to the advance action! here you can adjust initial condition with much wilder flexibility,
        such as changing the mass of the electron!! You can make the momentum negative so it never hits the barrier at all!!
        You can adjust the intial position so it starts above the potential steps and watch it step down!!
        You can adjust the width of the potential step and watch internal reflection between the edges of potential steps""" , font=SUBTITLE_FONT)
        label2.pack()        
        L1 = tk.Label(self, text='Please Choose the electrons momentum Using the Slider:',font=MID_FONT)
        L1.pack()
        self.S1 = tk.Scale(self, orient = tk.HORIZONTAL, from_=-4, to=4,resolution=0.1, bd=6)
        self.S1.pack()
        L2 = tk.Label(self, text='Please Choose potential barrier height(V0):',font=MID_FONT)
        L2.pack()
        self.S2 = tk.Scale(self, orient = tk.HORIZONTAL, from_=1, to=4,resolution=0.1, bd=6)
        self.S2.pack()
        L3 = tk.Label(self, text='Please Choose potential barrier width:',font=MID_FONT)
        L3.pack()
        self.S3 = tk.Scale(self, orient = tk.HORIZONTAL, from_=5, to=200,resolution=1, bd=6)
        self.S3.pack()
        L4 = tk.Label(self, text='please choose initial electron position',font=MID_FONT)
        L4.pack()
        self.S4 = tk.Scale(self, orient = tk.HORIZONTAL, from_=-150, to=150,resolution=1, bd=6)
        self.S4.pack()
        L5 = tk.Label(self, text='please choose desirable electron mass:)',font=MID_FONT)
        L5.pack()
        self.S5 = tk.Scale(self, orient = tk.HORIZONTAL, from_=1, to=4,resolution=0.1, bd=6)
        self.S5.pack()
        #self.n = int(self.S1.get())
        button2 = tk.Button(self, text="Go back to the Main Menu",activebackground= 'green',bd=5,
                            command=lambda: controller.show_frame(StartPage))
        button2.pack(side="bottom")
        button3 = tk.Button(self, text="!!Click me before the action!!", activebackground= 'green',bd=6, 
                           command=self.info) 
    
        button3.pack()
        button4 = tk.Button(self, text="action now!", activebackground= 'green',bd=6, 
                           command=self.action) 
    
        button4.pack()
        #All the independent variables
        #creating time frames
        self.dt = 0.04
        self.N_steps = 50
        self.t_max = 120
        self.frames = int(self.t_max / float(self.N_steps * self.dt)) 
        
        #make the x axis
        self.N = 6800
        self.dx = 0.1
        self.x = self.dx * (np.arange(self.N) - 0.5 * self.N)
      
        #arbitary constant
        self.m=1.0
        self.hbar=1.0
        self.V0=1.
        
        #Geometry of initial setup
        self.x0 = -100.
        self.barrierwidth=200
        
        self.V_x = square_barrier(self.x,self.barrierwidth,self.V0 )

        #the properties of such an electron
        self.d = 10
        self.k0=1
        self.psi_x0 = gauss_x(self.x,self.d,self.x0,self.k0)
        
    def action(self):
        self.m=float(self.S5.get())
        self.V0=float(self.S2.get())
        self.barrierwidth=float(self.S3.get())        
        self.V_x = square_barrier(self.x,self.barrierwidth,self.V0 )
        self.k0=float(self.S1.get())
        self.x0 = float(self.S4.get())
        self.psi_x0 = gauss_x(self.x,self.d,self.x0,self.k0)
        self.S = schrodinger.Schrodinger(x=self.x,
                psi_x0=self.psi_x0,
                V_x=self.V_x,
                hbar=self.hbar,
                m=self.m,
                k0=-28)
        # Set up plot
        self.fig = pl.figure()

        # plotting limits
        self.xlim = (-200, 200)

        # top axes show the x-space data

        self.ax1 = self.fig.add_subplot(111, xlim=self.xlim,
                            ylim=(0,
                                    self.V0 + 0.3))
        self.psi_x_line, = self.ax1.plot([], [], c='b', label=r'$|\psi(x)|$')
        self.V_x_line, = self.ax1.plot([], [], c='r', label=r'$V(x)$')


        self.title = self.ax1.set_title("at time t")
        self.ax1.legend(prop=dict(size=12))
        self.ax1.set_xlabel('$x$')
        self.ax1.set_ylabel(r'$|\psi(x)|$')
        
        def init():
            self.psi_x_line.set_data([], [])
            self.V_x_line.set_data([], [])
            self.title.set_text("click with mover to ")
            return (self.psi_x_line, self.V_x_line, self.title)

        def animate(i):
            self.S.time_step(self.dt, self.N_steps)
            self.psi_x_line.set_data(self.S.x,  2*abs(self.S.psi_x))
            self.V_x_line.set_data(self.S.x, self.S.V_x)
            self.title.set_text("Tunelling of an electron @ t = %.2f" % self.S.t)
            if self.S.t > 500.:
                self.psi_x_line.set_data([], [])

            return (self.psi_x_line, self.V_x_line, self.title)
        
        self.anim = animation.FuncAnimation(self.fig, animate, init_func=init,
                               frames=self.frames, interval=10, blit=True)
        pl.show()
                               
    def info(self):
        self.n = float(self.S1.get())
        self.m=float(self.S5.get())
        box.showinfo("input energy(abitary):",(self.n**2)/(self.m*2))
        box.showinfo("Bug description:",'''The following may happen to you:
        The frame around the animation is blacked out?        Minimize then Maximize the figure window.
        
        The title is supposed to update you with the current time(arbitary) since intitiation. If not?         Use the fourth tool from the left("the cross move"), click rapidly with the special move cursor (please do that, it takes me a long time to make the title update itself)
        
        IF the animation is too slow?      Use a faster computer or be patient, yes, I have tried reducing the frame renew rate, it works only on good computer
        
        !!!Don't close the animation too soon after its has initiated, or it will crash and there will be much wailing and gnashing of teeth!!
        
        Have Fun :)
        ''')

    
    
class Quit(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Do you REALLY want to QUIT??", font=TITLE_FONT)
        label.pack(side="top", fill="both", pady=10)
        button1 = tk.Button(self, text="Go back to the Main Menu",activebackground= 'green',bd=6, 
                           command=lambda: controller.show_frame(StartPage))
        button2 = tk.Button(self, text="Yes", activebackground= 'green',bd=6,
                            command=self.quit_program)  
                            
        button1.pack(side="bottom")
        button2.pack()
                                  
    def quit_program(self):
        app.destroy() 
        
        

#run the programme                                                                                                                                                                                                                                                                                    
if __name__ == "__main__":
    app = Main()
    app.mainloop()