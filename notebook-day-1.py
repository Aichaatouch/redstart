import marimo

__generated_with = "0.13.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Redstart: A Lightweight Reusable Booster""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(src="public/images/redstart.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Project Redstart is an attempt to design the control systems of a reusable booster during landing.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In principle, it is similar to SpaceX's Falcon Heavy Booster.

    >The Falcon Heavy booster is the first stage of SpaceX's powerful Falcon Heavy rocket, which consists of three modified Falcon 9 boosters strapped together. These boosters provide the massive thrust needed to lift heavy payloads—like satellites or spacecraft—into orbit. After launch, the two side boosters separate and land back on Earth for reuse, while the center booster either lands on a droneship or is discarded in high-energy missions.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.center(
        mo.Html("""
    <iframe width="560" height="315" src="https://www.youtube.com/embed/RYUr-5PYA7s?si=EXPnjNVnqmJSsIjc" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>""")
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Dependencies""")
    return


@app.cell
def _():
    import scipy
    import scipy.integrate as sci
    from scipy.integrate import solve_ivp
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    import numpy as np
    from tqdm import tqdm
    from scipy.interpolate import interp1d
    # The use of autograd is optional in this project, but it may come in handy!
    import autograd
    import autograd.numpy as np
    import autograd.numpy.linalg as la
    from autograd import isinstance, tuple
    return FFMpegWriter, FuncAnimation, np, plt, solve_ivp, tqdm


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## The Model

    The Redstart booster in model as a rigid tube of length $2 \ell$ and negligible diameter whose mass $M$ is uniformly spread along its length. It may be located in 2D space by the coordinates $(x, y)$ of its center of mass and the angle $\theta$ it makes with respect to the vertical (with the convention that $\theta > 0$ for a left tilt, i.e. the angle is measured counterclockwise)

    This booster has an orientable reactor at its base ; the force that it generates is of amplitude $f>0$ and the angle of the force with respect to the booster axis is $\phi$ (with a counterclockwise convention).

    We assume that the booster is subject to gravity, the reactor force and that the friction of the air is negligible.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.center(mo.image(src="public/images/geometry.svg"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Constants

    For the sake of simplicity (this is merely a toy model!) in the sequel we assume that: 

      - the total length $2 \ell$ of the booster is 2 meters,
      - its mass $M$ is 1 kg,
      - the gravity constant $g$ is 1 m/s^2.

    This set of values is not realistic, but will simplify our computations and do not impact the structure of the booster dynamics.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Helpers

    ### Rotation matrix

    $$ 
    \begin{bmatrix}
    \cos \alpha & - \sin \alpha \\
    \sin \alpha &  \cos \alpha  \\
    \end{bmatrix}
    $$
    """
    )
    return


@app.cell
def _(np):
    def R(alpha):
        return np.array([
            [np.cos(alpha), -np.sin(alpha)], 
            [np.sin(alpha),  np.cos(alpha)]
        ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Videos

    It will be very handy to make small videos to visualize the evolution of our booster!
    Here is an example of how such videos can be made with Matplotlib and displayed in marimo.
    """
    )
    return


@app.cell
def _(FFMpegWriter, FuncAnimation, mo, np, plt, tqdm):
    def make_video(output):
        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        num_frames = 100
        fps = 30 # Number of frames per second

        def animate(frame_index):    
            # Clear the canvas and redraw everything at each step
            plt.clf()
            plt.xlim(0, 2*np.pi)
            plt.ylim(-1.5, 1.5)
            plt.title(f"Sine Wave Animation - Frame {frame_index+1}/{num_frames}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid(True)

            x = np.linspace(0, 2*np.pi, 100)
            phase = frame_index / 10
            y = np.sin(x + phase)
            plt.plot(x, y, "r-", lw=2, label=f"sin(x + {phase:.1f})")
            plt.legend()

            pbar.update(1)

        pbar = tqdm(total=num_frames, desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=num_frames)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")

    _filename = "wave_animation.mp4"
    make_video(_filename)
    (mo.video(src=_filename))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Getting Started""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Constants

    Define the Python constants `g`, `M` and `l` that correspond to the gravity constant, the mass and half-length of the booster.
    """
    )
    return


@app.cell
def _():
    g=1
    M=1
    l=1

    return (l,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Forces

    Compute the force $(f_x, f_y) \in \mathbb{R}^2$ applied to the booster by the reactor.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    f_x = -f \sin(\theta + \phi), \quad f_y = f \cos(\theta + \phi)
    $$

    Et :

    $$
    (\ddot{x}, \ddot{y}) = (f_x, f_y - 1)
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Center of Mass

    Give the ordinary differential equation that governs $(x, y)$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    (\ddot{x} , \ddot{y}) = (fx, fy - 1) 
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Moment of inertia

    Compute the moment of inertia $J$ of the booster and define the corresponding Python variable `J`.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Puisque la barre est considérée tourner autour de son centre, on utilise l'expression du moment cinétique suivante :


    $$
    J = \frac{1}{12} m (2*l)^2
    $$

    Donc avec \( m = 1 \) et \( l = 1 \) :

    $$
    J = \frac{1}{3}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Tilt

    Give the ordinary differential equation that governs the tilt angle $\theta$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    On a l'équation :

    $$
    \ddot{\theta} = \frac{-3 \cdot l}{M \cdot l^2} \cdot \sin(\varphi)
    $$

    Ce qui se simplifie en :

    $$
    \ddot{\theta} = \frac{-3}{M \cdot l} \cdot \sin(\varphi)
    $$

    Donc, on a finalement :

    $$
    \boxed{\ddot{\theta} = -3f \cdot \sin(\varphi)}
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Simulation

    Define a function `redstart_solve` that, given as input parameters: 

      - `t_span`: a pair of initial time `t_0` and final time `t_f`,
      - `y0`: the value of the state `[x, dx, y, dy, theta, dtheta]` at `t_0`,
      - `f_phi`: a function that given the current time `t` and current state value `y`
         returns the values of the inputs `f` and `phi` in an array.

    returns:

      - `sol`: a function that given a time `t` returns the value of the state `[x, dx, y, dy, theta, dtheta]` at time `t` (and that also accepts 1d-arrays of times for multiple state evaluations).

    A typical usage would be:

    ```python
    def free_fall_example():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi(t, y):
            return np.array([0.0, 0.0]) # input [f, phi]
        sol = redstart_solve(t_span, y0, f_phi)
        t = np.linspace(t_span[0], t_span[1], 1000)
        y_t = sol(t)[2]
        plt.plot(t, y_t, label=r"$y(t)$ (height in meters)")
        plt.plot(t, l * np.ones_like(t), color="grey", ls="--", label=r"$y=\ell$")
        plt.title("Free Fall")
        plt.xlabel("time $t$")
        plt.grid(True)
        plt.legend()
        return plt.gcf()
    free_fall_example()
    ```

    Test this typical example with your function `redstart_solve` and check that its graphical output makes sense.
    """
    )
    return


@app.cell
def _(l, np, plt, solve_ivp):


    def free_fall_example():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]  # [x, dx, y, dy, theta, dtheta]

        def f_phi(t, y):
            return np.array([0.0, 0.0])  # pas de poussée

        def f(t, y):
            x, dx, yp, dy, theta, dtheta = y
            f_val, phi = f_phi(t, y)

            fx = -f_val * np.sin(theta + phi)
            fy = f_val * np.cos(theta + phi)

            ddx = fx
            ddy = fy - 1
            ddtheta = -3 * f_val * np.sin(phi)

            return [dx, ddx, dy, ddy, dtheta, ddtheta]

        # Résolution avec solve_ivp + interpolation
        t_eval = np.linspace(t_span[0], t_span[1], 1000)
        sol_ivp = solve_ivp(f, t_span, y0, t_eval=t_eval)

        y_t = sol_ivp.y[2]  # position verticale y

        plt.plot(t_eval, y_t, label=r"$y(t)$ (height in meters)")
        plt.plot(t_eval, l * np.ones_like(t_eval), color="grey", ls="--", label=r"$y=\ell$")
        plt.title("Free Fall")
        plt.xlabel("time $t$")
        plt.grid(True)
        plt.legend()
        return plt.gcf()

    free_fall_example()


    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controlled Landing

    Assume that $x$, $\dot{x}$, $\theta$ and $\dot{\theta}$ are null at $t=0$. For $y(0)= 10$ and $\dot{y}(0)$, can you find a time-varying force $f(t)$ which, when applied in the booster axis ($\theta=0$), yields $y(5)=\ell$ and $\dot{y}(5)=0$?

    Simulate the corresponding scenario to check that your solution works as expected.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Nous disposons de 4 données :  
    \( y(0) \), \( \dot{y}(0) \), \( y(5) \), \( \dot{y}(5) \)

    Il nous suffit donc de disposer d’un polynôme de degré 3 (avec 4 inconnues) pour trouver une expression satisfaisant les 4 conditions.  
    Particulièrement pour un choix de \( \theta = 0 ,\phi = 0 \), soit donc :

    $$
    y(t) = at^3 + bt^2 + ct + d
    $$

    Conditions :
    \( y(0) = 10 \)
    , \( \dot{y}(0) = -2 \)
    , \( y(5) = 1 \)
    , \( \dot{y}(5) = 0 \)

    Nous obtenons :
    \( a = \frac{8}{125} \)
    , \( b = \frac{7}{25} \)
    , \( c = -2 \)
    , \( d = 10 \)

    ---

    Et de plus, d’après \( \dot{X} = f(t, X) \),  
    on montre que :

    $$
    f(t) = \ddot{y}(t) = 6at + 2b
    $$

    On peut donc prendre :

    $$
    f(t) = 48/125 t + 14/25
    $$

    ---

    Expression finale de \( y(t) \) :

    $$
    y(t) = -\frac{28}{125} t^3 + \frac{27}{25} t^2 - 2t + 10
    $$

    Et on a le résultat voulu.
    """
    )
    return


@app.cell
def _(np, plt):
    a = 8 / 125
    b = -7 / 25
    c = -2
    d = 10

    # Définition de la fonction y(t)
    def y(t):
        return a * t**3 + b * t**2 + c * t + d

    # Plage de t de 0 à 5
    t_vals = np.linspace(0, 5, 200)
    y_vals = y(t_vals)

    # Tracé
    plt.figure(figsize=(8, 4))
    plt.plot(t_vals, y_vals, label='y(t)', color='blue')
    plt.title('Tracé de y(t) = at³ + bt² + ct + d')
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.grid(True)
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Drawing

    Create a function that draws the body of the booster, the flame of its reactor as well as its target landing zone on the ground (of coordinates $(0, 0)$).

    The drawing can be very simple (a rectangle for the body and another one of a different color for the flame will do perfectly!).
    """
    )
    return


@app.cell
def _(np, plt):

    def draw_booster(x, y, theta, title):
        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot the target landing zone
        ax.scatter(0, 0, color='red', s=100, label='Target Landing Zone')

        # Define the booster body dimensions
        body_length = 1.0
        body_height = 2.0

        # Define the corners of the booster body
        corners = np.array([[-body_length/2, -body_length/2, body_length/2, body_length/2],
                            [0, body_height, body_height, 0]])

        # Rotation matrix for angle theta
        rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                               [np.sin(theta), np.cos(theta)]])

        # Apply the rotation
        rotated_corners = rot_matrix @ corners
        rotated_corners[0, :] += x
        rotated_corners[1, :] += y

        # Draw the booster body
        ax.fill(rotated_corners[0, :], rotated_corners[1, :], color='blue', label='Booster Body')

        # Define the flame dimensions
        flame_length = 0.6
        flame_height = 1.5

        # Define the flame shape
        flame = np.array([[0, -flame_length/2, flame_length/2],
                          [-flame_height, 0, 0]])

        # Apply the same rotation to the flame
        rotated_flame = rot_matrix @ flame
        rotated_flame[0, :] += x
        rotated_flame[1, :] += y

        # Draw the flame
        ax.fill(rotated_flame[0, :], rotated_flame[1, :], color='orange', label='Reactor Flame')

        # Set plot limits and labels
        ax.set_xlim(-5, 5)
        ax.set_ylim(-2, 12)
        ax.set_aspect('equal')
        ax.set_xlabel('Horizontal Position (m)')
        ax.set_ylabel('Vertical Position (m)')
        ax.set_title(title)
        ax.grid(True)
        ax.legend()

        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.center(mo.image("public/images/booster_drawing.png"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Make sure that the orientation of the flame is correct and that its length is proportional to the force $f$ with the length equal to $\ell$ when $f=Mg$.

    The function shall accept the parameters `x`, `y`, `theta`, `f` and `phi`.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Visualization

    Produce a video of the booster for 5 seconds when

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=0$ and $\phi=0$

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=Mg$ and $\phi=0$

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=Mg$ and $\phi=\pi/8$

      - the parameters are those of the controlled landing studied above.

    As an intermediary step, you can begin with production of image snapshots of the booster location (every 1 sec).
    """
    )
    return


if __name__ == "__main__":
    app.run()
