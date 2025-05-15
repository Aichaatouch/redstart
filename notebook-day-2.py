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


@app.cell
def _(mo):
    mo.md(r"""## Dependencies""")
    return


@app.cell(hide_code=True)
def _():
    import scipy
    import scipy.integrate as sci
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    from tqdm import tqdm

    # The use of autograd is optional in this project, but it may come in handy!
    import autograd
    import autograd.numpy as np
    import autograd.numpy.linalg as la
    from autograd import isinstance, tuple
    return FFMpegWriter, FuncAnimation, mpl, np, plt, sci, scipy, tqdm


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


@app.cell(hide_code=True)
def _(np):
    def R(alpha):
        return np.array([
            [np.cos(alpha), -np.sin(alpha)], 
            [np.sin(alpha),  np.cos(alpha)]
        ])
    return (R,)


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


@app.cell(hide_code=True)
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
    mo.show_code(mo.video(src=_filename))
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


@app.cell(hide_code=True)
def _():
    g = 1.0
    M = 1.0
    l = 1
    return M, g, l


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
    \begin{align*}
    f_x & = -f \sin (\theta + \phi) \\
    f_y & = +f \cos(\theta +\phi)
    \end{align*}
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
    \begin{align*}
    M \ddot{x} & = -f \sin (\theta + \phi) \\
    M \ddot{y} & = +f \cos(\theta +\phi) - Mg
    \end{align*}
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


@app.cell
def _(M, l):
    J = M * l * l / 3
    J
    return (J,)


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
    $$
    J \ddot{\theta} = - \ell (\sin \phi)  f
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


@app.cell(hide_code=True)
def _(J, M, g, l, np, scipy):
    def redstart_solve(t_span, y0, f_phi):
        def fun(t, state):
            x, dx, y, dy, theta, dtheta = state
            f, phi = f_phi(t, state)
            d2x = (-f * np.sin(theta + phi)) / M
            d2y = (+ f * np.cos(theta + phi)) / M - g
            d2theta = (- l * np.sin(phi)) * f / J
            return np.array([dx, d2x, dy, d2y, dtheta, d2theta])
        r = scipy.integrate.solve_ivp(fun, t_span, y0, dense_output=True)
        return r.sol
    return (redstart_solve,)


@app.cell(hide_code=True)
def _(l, np, plt, redstart_solve):
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controlled Landing

    Assume that $x$, $\dot{x}$, $\theta$ and $\dot{\theta}$ are null at $t=0$. For $y(0)= 10$ and $\dot{y}(0) = - 2*\ell$,  can you find a time-varying force $f(t)$ which, when applied in the booster axis ($\theta=0$), yields $y(5)=\ell$ and $\dot{y}(5)=0$?

    Simulate the corresponding scenario to check that your solution works as expected.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    % y(t)
    y(t)
    = \frac{2(5-\ell)}{125}\,t^3
      + \frac{3\ell-10}{25}\,t^2
      - 2\,t
      + 10
    $$

    $$
    % f(t)
    f(t)
    = M\!\Bigl[
        \frac{12(5-\ell)}{125}\,t
        + \frac{6\ell-20}{25}
        + g
      \Bigr].
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(M, g, l, np, plt, redstart_solve):

    def smooth_landing_example():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, -2.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi_smooth_landing(t, state):
            a = 12 * (5 - l) / 125
            b = (6 * l - 20) / 25
            return np.array([M * (a * t + b + g), 0])
        sol = redstart_solve(t_span, y0, f_phi=f_phi_smooth_landing)
        t = np.linspace(t_span[0], t_span[1], 1000)
        y_t = sol(t)[2]
        plt.plot(t, y_t, label=r"$y(t)$ (height in meters)")
        plt.plot(t, l * np.ones_like(t), color="grey", ls="--", label=r"$y=\ell$")
        plt.title("Controlled Landing")
        plt.xlabel("time $t$")
        plt.grid(True)
        plt.legend()
        return plt.gcf()
    smooth_landing_example()
    return


@app.cell
def _(M, g, l, np, plt, redstart_solve):
    def smooth_landing_example_force():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, -2.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi_smooth_landing(t, state):
            a = 12 * (5 - l) / 125
            b = (6 * l - 20) / 25
            return np.array([M * (a * t + b + g), 0])
        sol = redstart_solve(t_span, y0, f_phi=f_phi_smooth_landing)
        t = np.linspace(t_span[0], t_span[1], 1000)
        y_t = sol(t)[2]
        plt.plot(t, y_t, label=r"$y(t)$ (height in meters)")
        plt.plot(t, l * np.ones_like(t), color="grey", ls="--", label=r"$y=\ell$")
        plt.title("Controlled Landing")
        plt.xlabel("time $t$")
        plt.grid(True)
        plt.legend()
        return plt.gcf()
    smooth_landing_example_force()
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
def _(M, R, g, l, mo, mpl, np, plt):
    def draw_booster(x=0, y=l, theta=0.0, f=0.0, phi=0.0, axes=None, **options):
        L = 2 * l
        if axes is None:
            _fig, axes = plt.subplots()

        axes.set_facecolor('#F0F9FF') 

        ground = np.array([[-2*l, 0], [2*l, 0], [2*l, -l], [-2*l, -l], [-2*l, 0]]).T
        axes.fill(ground[0], ground[1], color="#E3A857", **options)

        b = np.array([
            [l/10, -l], 
            [l/10, l], 
            [0, l+l/10], 
            [-l/10, l], 
            [-l/10, -l], 
            [l/10, -l]
        ]).T
        b = R(theta) @ b
        axes.fill(b[0]+x, b[1]+y, color="black", **options)

        ratio = l / (M*g) # when f= +MG, the flame length is l 

        flame = np.array([
            [l/10, 0], 
            [l/10, - ratio * f], 
            [-l/10, - ratio * f], 
            [-l/10, 0], 
            [l/10, 0]
        ]).T
        flame = R(theta+phi) @ flame
        axes.fill(
            flame[0] + x + l * np.sin(theta), 
            flame[1] + y - l * np.cos(theta), 
            color="#FF4500", 
            **options
        )

        return axes

    _axes = draw_booster(x=0.0, y=20*l, theta=np.pi/8, f=M*g, phi=np.pi/8)
    _fig = _axes.figure
    _axes.set_xlim(-4*l, 4*l)
    _axes.set_ylim(-2*l, 24*l)
    _axes.set_aspect("equal")
    _axes.grid(True)
    _MaxNLocator = mpl.ticker.MaxNLocator
    _axes.xaxis.set_major_locator(_MaxNLocator(integer=True))
    _axes.yaxis.set_major_locator(_MaxNLocator(integer=True))
    _axes.set_axisbelow(True)
    mo.center(_fig)
    return (draw_booster,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Visualisation

    Produce a video of the booster for 5 seconds when

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=0$ and $\phi=0$

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=Mg$ and $\phi=0$

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=Mg$ and $\phi=\pi/8$

      - the parameters are those of the controlled landing studied above.

    As an intermediary step, you can begin the with production of image snapshots of the booster location (every 1 sec).
    """
    )
    return


@app.cell(hide_code=True)
def _(draw_booster, l, mo, np, plt, redstart_solve):
    def sim_1():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([0.0, 0.0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig, axes = plt.subplots(1, 6, sharey=True)
        num_snapshots = 6
        for i, t in enumerate(np.linspace(t_span[0], 5.0, num_snapshots)):
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            ax = draw_booster(x, y, theta, f, phi, axes=axes[i])
            ax.set_xticks([0.0])
            ax.set_xlim(-4*l, +4*l)
            ax.set_ylim(-2*l, +12*l)
            ax.set_aspect("equal")
            ax.set_xlabel(rf"$t={t}$")
            ax.set_axisbelow(True)
            ax.grid(True)
        w, h = fig.get_size_inches()
        fig.set_size_inches(12, h) 
        return mo.center(fig)

    sim_1()
    return


@app.cell(hide_code=True)
def _(M, draw_booster, g, l, mo, np, plt, redstart_solve):
    def sim_2():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([M*g, 0.0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig, axes = plt.subplots(1, 6, sharey=True)
        num_snapshots = 6
        for i, t in enumerate(np.linspace(t_span[0], 5.0, num_snapshots)):
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            ax = draw_booster(x, y, theta, f, phi, axes=axes[i])
            ax.set_xticks([0.0])
            ax.set_xlim(-4*l, +4*l)
            ax.set_ylim(-2*l, +12*l)
            ax.set_aspect("equal")
            ax.set_xlabel(rf"$t={t}$")
            ax.set_axisbelow(True)
            ax.grid(True)
        w, h = fig.get_size_inches()
        fig.set_size_inches(12, h) 
        return mo.center(fig)

    sim_2()
    return


@app.cell(hide_code=True)
def _(M, draw_booster, g, l, mo, np, plt, redstart_solve):
    def sim_3():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([M*g, np.pi/8])
        sol = redstart_solve(t_span, y0, f_phi)

        fig, axes = plt.subplots(1, 6, sharey=True)
        num_snapshots = 6
        for i, t in enumerate(np.linspace(t_span[0], 5.0, num_snapshots)):
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            ax = draw_booster(x, y, theta, f, phi, axes=axes[i])
            ax.set_xticks([0.0])
            ax.set_xlim(-4*l, +4*l)
            ax.set_ylim(-2*l, +12*l)
            ax.set_aspect("equal")
            ax.set_xlabel(rf"$t={t}$")
            ax.set_axisbelow(True)
            ax.grid(True)
        w, h = fig.get_size_inches()
        fig.set_size_inches(12, h) 
        return mo.center(fig)

    sim_3()
    return


@app.cell(hide_code=True)
def _(M, draw_booster, g, l, mo, np, plt, redstart_solve):
    def sim_4():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, -2.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi(t, state):
            a = 12 * (5 - l) / 125
            b = (6 * l - 20) / 25
            return np.array([M * (a * t + b + g), 0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig, axes = plt.subplots(1, 6, sharey=True)
        num_snapshots = 6
        for i, t in enumerate(np.linspace(t_span[0], 5.0, num_snapshots)):
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            ax = draw_booster(x, y, theta, f, phi, axes=axes[i])
            ax.set_xticks([0.0])
            ax.set_xlim(-4*l, +4*l)
            ax.set_ylim(-2*l, +12*l)
            ax.set_aspect("equal")
            ax.set_xlabel(rf"$t={t}$")
            ax.set_axisbelow(True)
            ax.grid(True)
        w, h = fig.get_size_inches()
        fig.set_size_inches(12, h) 
        return mo.center(fig)

    sim_4()
    return


@app.cell(hide_code=True)
def _(
    FFMpegWriter,
    FuncAnimation,
    draw_booster,
    l,
    mo,
    np,
    plt,
    redstart_solve,
    tqdm,
):
    def video_sim_1():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([0.0, 0.0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        axes = plt.gca()
        num_frames = 100
        fps = 30 # 30 frames per second
        ts = np.linspace(t_span[0], t_span[1], int(np.round(t_span[1] * fps)) + 1)
        output = "sim_1.mp4"

        def animate(t):    
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            axes.clear()
            draw_booster(x, y, theta, f, phi, axes=axes)
            axes.set_xticks([0.0])
            axes.set_xlim(-4*l, +4*l)
            axes.set_ylim(-2*l, +12*l)
            axes.set_aspect("equal")
            axes.set_xlabel(rf"$t={t:.1f}$")
            axes.set_axisbelow(True)
            axes.grid(True)

            pbar.update(1)

        pbar = tqdm(total=len(ts), desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=ts)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")
        return output

    mo.video(src=video_sim_1())
    return


@app.cell(hide_code=True)
def _(
    FFMpegWriter,
    FuncAnimation,
    M,
    draw_booster,
    g,
    l,
    mo,
    np,
    plt,
    redstart_solve,
    tqdm,
):
    def video_sim_2():
        L = 2*l

        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([M*g, 0.0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        axes = plt.gca()
        num_frames = 100
        fps = 30 # 30 frames per second
        ts = np.linspace(t_span[0], t_span[1], int(np.round(t_span[1] * fps)) + 1)
        output = "sim_2.mp4"

        def animate(t):    
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            axes.clear()
            draw_booster(x, y, theta, f, phi, axes=axes)
            axes.set_xticks([0.0])
            axes.set_xlim(-4*l, +4*l)
            axes.set_ylim(-2*l, +12*l)
            axes.set_aspect("equal")
            axes.set_xlabel(rf"$t={t:.1f}$")
            axes.set_axisbelow(True)
            axes.grid(True)

            pbar.update(1)

        pbar = tqdm(total=len(ts), desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=ts)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")
        return output

    mo.video(src=video_sim_2())
    return


@app.cell(hide_code=True)
def _(
    FFMpegWriter,
    FuncAnimation,
    M,
    draw_booster,
    g,
    l,
    mo,
    np,
    plt,
    redstart_solve,
    tqdm,
):
    def video_sim_3():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        def f_phi(t, state):
            return np.array([M*g, np.pi/8])
        sol = redstart_solve(t_span, y0, f_phi)

        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        axes = plt.gca()
        num_frames = 100
        fps = 30 # 30 frames per second
        ts = np.linspace(t_span[0], t_span[1], int(np.round(t_span[1] * fps)) + 1)
        output = "sim_3.mp4"

        def animate(t):    
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            axes.clear()
            draw_booster(x, y, theta, f, phi, axes=axes)
            axes.set_xticks([0.0])
            axes.set_xlim(-4*l, +4*l)
            axes.set_ylim(-2*l, +12*l)
            axes.set_aspect("equal")
            axes.set_xlabel(rf"$t={t:.1f}$")
            axes.set_axisbelow(True)
            axes.grid(True)

            pbar.update(1)

        pbar = tqdm(total=len(ts), desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=ts)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")
        return output

    mo.video(src=video_sim_3())
    return


@app.cell(hide_code=True)
def _(
    FFMpegWriter,
    FuncAnimation,
    M,
    draw_booster,
    g,
    l,
    mo,
    np,
    plt,
    redstart_solve,
    tqdm,
):
    def video_sim_4():
        L = 2*l
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, -2.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi(t, state):
            a = 12 * (5 - l) / 125
            b = (6 * l - 20) / 25
            return np.array([M * (a * t + b + g), 0])
        sol = redstart_solve(t_span, y0, f_phi)

        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        axes = plt.gca()
        num_frames = 100
        fps = 30 # 30 frames per second
        ts = np.linspace(t_span[0], t_span[1], int(np.round(t_span[1] * fps)) + 1)
        output = "sim_4.mp4"

        def animate(t):    
            x, dx, y, dy, theta, dtheta = state = sol(t)
            f, phi = f_phi(t, state)
            axes.clear()
            draw_booster(x, y, theta, f, phi, axes=axes)
            axes.set_xticks([0.0])
            axes.set_xlim(-4*l, +4*l)
            axes.set_ylim(-2*l, +12*l)
            axes.set_aspect("equal")
            axes.set_xlabel(rf"$t={t:.1f}$")
            axes.set_axisbelow(True)
            axes.grid(True)

            pbar.update(1)

        pbar = tqdm(total=len(ts), desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=ts)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")
        return output

    mo.video(src=video_sim_4())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Linearized Dynamics""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Equilibria

    We assume that $|\theta| < \pi/2$, $|\phi| < \pi/2$ and that $f > 0$. What are the possible equilibria of the system for constant inputs $f$ and $\phi$ and what are the corresponding values of these inputs?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""


    On a :

    \[
    X = \begin{pmatrix}
    x \\
    \dot{x} \\
    y \\
    \dot{y} \\
    \theta \\
    \dot{\theta}
    \end{pmatrix}
    \]

    La dérivée du vecteur d’état s’écrit :

    \[
    \dot{X} = \begin{pmatrix}
    \dot{x} \\
    \frac{-f}{M} \sin(\theta + \phi) \\
    \dot{y} \\
    \frac{f}{M} \cos(\theta + \phi) - g \\
    \dot{\theta} \\
    -\frac{3f}{Ml} \sin(\phi)
    \end{pmatrix}
    \]

    On note :

    \[
    \dot{X} = f(X, u)
    \quad \text{avec} \quad u = \begin{pmatrix} f \\ \phi \end{pmatrix}
    \]



    Un état d’équilibre est tel que :

    \[
    \dot{X} = 0
    \]

    Ce qui donne le système :

    \[
    \begin{cases}
    \dot{x} = 0 \\
    \frac{f}{M} \sin(\theta + \phi) = 0 \\
    \dot{y} = 0 \\
    \frac{f}{M} \cos(\theta + \phi) - g = 0 \\
    \dot{\theta} = 0 \\
    -\frac{3f}{Ml} \sin(\phi) = 0
    \end{cases}
    \]

    On obtient alors :

    \[
    \begin{aligned}
    &x, y, \theta = \text{constantes} \\
    &\frac{f}{M} \sin(\theta + \phi) = 0 \Rightarrow \boxed{f = 0 \text{ ou } \sin(\theta + \phi) = 0} \\
    &\frac{f}{M} \cos(\theta + \phi) = g \\
    &\sin(\phi) = 0 \Rightarrow \boxed{\phi = 0}
    \end{aligned}
    \]

    ---

    ### Cas 1 : \( f = 0 \)

    Alors :

    - \( \frac{f}{M} \cos(\theta + \phi) = 0 \Rightarrow g = 0 \), ce qui est impossible.
    Donc ce cas est à rejeter.

    ---

    ### Cas 2 : \( \sin(\theta + \phi) = 0 \)

    Et puisque \( |\theta| < \pi/2 \), \( |\phi| < \pi/2 \), on a :

    - \( \theta + \phi = 0 \)

    Mais aussi :

    - \( \sin(\phi) = 0 \Rightarrow \phi = 0 \)

    Pour satisfaire à la fois \( \theta + \phi = 0 \) et \( \phi = 0 \), on trouve :

    \[
    \boxed{\theta = 0 \quad \text{et} \quad \phi = 0}
    \]

    Et donc :

    \[
    \frac{f}{M} = g \Rightarrow \boxed{f = Mg}
    \]

    ---

    ### État d’équilibre

    L’état d’équilibre est atteint pour :

    \[
    \boxed{
    x=constante  , \quad \dot{x} = 0  , \quad y=constante,\quad\dot{y} = 0  , \quad \theta = 0  ,  \quad\dot{\theta} = 0  ,  \quad \phi = 0 , \quad f = Mg
    }
    \]

    Ce qui correspond à une position verticale en équilibre.


    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Linearized Model

    Introduce the error variables $\Delta x$, $\Delta y$, $\Delta \theta$, and $\Delta f$ and $\Delta \phi$ of the state and input values with respect to the generic equilibrium configuration.
    What are the linear ordinary differential equations that govern (approximately) these variables in a neighbourhood of the equilibrium?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    On a :

    \[
    X = 
    \begin{bmatrix}
    x \\
    \dot{x} \\
    y \\
    \dot{y} \\
    \theta \\
    \dot{\theta}
    \end{bmatrix}, \quad 
    \text{entrée } u = 
    \begin{bmatrix}
    f \\
    \varphi
    \end{bmatrix}
    \]

    Les équations sont :

    \[
    \begin{cases}
    \ddot{x} = -\frac{f}{M} \sin(\theta + \varphi) \\
    \ddot{y} = \frac{f}{M} \cos(\theta + \varphi) - g \\
    \ddot{\theta} = -\frac{3f}{Ml} \sin(\varphi)
    \end{cases}
    \]

    ---

    \[
    \frac{d}{dt}
    \underbrace{
    \begin{bmatrix}
    x \\
    \dot{x} \\
    y \\
    \dot{y} \\
    \theta \\
    \dot{\theta}
    \end{bmatrix}
    }_{X}
    =
    \underbrace{
    \begin{bmatrix}
    \dot{x} \\
    -\frac{f}{M} \sin(\theta + \varphi) \\
    \dot{y} \\
    \frac{f}{M} \cos(\theta + \varphi) - g \\
    \dot{\theta} \\
    -\frac{3f}{Ml}\sin(\varphi)
    \end{bmatrix}
    }_{F(X,u)}
    \]

    ---

    On a l’équilibre \( (X_e, u_e) \) tel que :

    \[
    \theta = 0, \quad \varphi = 0, \quad f = Mg
    \]

    \[
    \dot{x} = \dot{y} = \dot{\theta} = 0
    \]

    Donc l’équilibre est :

    \[
    X_e =
    \begin{bmatrix}
    x_0 \\
    0 \\
    y_0 \\
    0 \\
    0 \\
    0
    \end{bmatrix},
    \quad
    u_e =
    \begin{bmatrix}
    Mg \\
    0
    \end{bmatrix}
    \]

    ---

    De tout système dynamique ayant un équilibre \( (X_e, u_e) \), nous pouvons définir le système linéaire suivant :

    \[
    \Delta X = X - X_e, \quad \Delta u = u - u_e
    \]

    et :

    \[
    \frac{d}{dt} \Delta X \approx A \Delta X + B \Delta u
    \]


    Les équations du système linéarisé sont :

    \[
    \begin{cases}
    \Delta \ddot{x} = -g ( \theta +  \varphi) \\
    \Delta \ddot{y} = \frac{f}{M} - g \\
    \Delta \ddot{\theta} = -\frac{3 g}{l}  \varphi
    \end{cases}
    \]



    où :

    \[
    A = \left. \frac{\partial F}{\partial X} \right|_{X_e, u_e}, \quad
    B = \left. \frac{\partial F}{\partial u} \right|_{X_e, u_e}
    \]

    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Standard Form

    What are the matrices $A$ and $B$ associated to this linear model in standard form?
    Define the corresponding NumPy arrays `A` and `B`.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    On a :

    \[
    F(X, u) =
    \begin{bmatrix}
    \dot{x} \\
    -\frac{f}{M} \sin(\theta + \varphi) \\
    \dot{y} \\
    \frac{f}{M} \cos(\theta + \varphi) - g \\
    \dot{\theta} \\
    -\frac{3f}{Ml} \sin(\varphi)
    \end{bmatrix}
    \]

    ---

    ## Matrice \( A = \frac{\partial F}{\partial X} \Big|_{X_e, u_e} \)

    Les dérivées partielles par rapport à \( X = (x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) \) :

    - \( \frac{\partial}{\partial \theta} \left( -\frac{f}{M} \sin(\theta + \varphi) \right) = -\frac{f}{M} \cos(\theta + \varphi) \rightarrow -g \)
    - \( \frac{\partial}{\partial \theta} \left( \frac{f}{M} \cos(\theta + \varphi) \right) = -\frac{f}{M} \sin(\theta + \varphi) \rightarrow 0 \)
    - Les autres dérivées sont nulles.

    Donc :

    \[
    A =
    \begin{bmatrix}
    0 & 1 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & -g & 0 \\
    0 & 0 & 0 & 1 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 1 \\
    0 & 0 & 0 & 0 & 0 & 0
    \end{bmatrix}
    \]

    ---

    ## Matrice \( B = \frac{\partial F}{\partial u} \Big|_{X_e, u_e} \)

    Avec \( u = (f, \varphi) \) :

    - \( \frac{\partial}{\partial f} \left( -\frac{f}{M} \sin(\theta + \varphi) \right) = -\frac{1}{M} \sin(\theta + \varphi) \rightarrow 0 \)
    - \( \frac{\partial}{\partial \varphi} \left( -\frac{f}{M} \sin(\theta + \varphi) \right) = -\frac{f}{M} \cos(\theta + \varphi) \rightarrow -g \)
    - \( \frac{\partial}{\partial f} \left( \frac{f}{M} \cos(\theta + \varphi) \right) = \frac{1}{M} \cos(\theta + \varphi) \rightarrow \frac{1}{M} \)
    - \( \frac{\partial}{\partial \varphi} \left( \frac{f}{M} \cos(\theta + \varphi) \right) = -\frac{f}{M} \sin(\theta + \varphi) \rightarrow 0 \)
    - \( \frac{\partial}{\partial f} \left( -\frac{3f}{Ml} \sin(\varphi) \right) = -\frac{3}{Ml} \sin(\varphi) \rightarrow 0 \)
    - \( \frac{\partial}{\partial \varphi} \left( -\frac{3f}{Ml} \sin(\varphi) \right) = -\frac{3f}{Ml} \cos(\varphi) \rightarrow -\frac{3g}{l} \)

    Donc :

    \[
    B =
    \begin{bmatrix}
    0 & 0 \\
    0 & -g \\
    0 & 0 \\
    \frac{1}{M} & 0 \\
    0 & 0 \\
    0 & -\frac{3g}{l}
    \end{bmatrix}
    \]

    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Stability

    Is the generic equilibrium asymptotically stable?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    D’après le critère de stabilité,le système est stable si les parties réelles de toutes les valeurs propres de la matrice A sont  
    strictement négatives?.
    Ici A est triangulaire supérieure, les  
    valeurs propres sont donc toutes nulles.
    A ne vérifie donc pas le critère de stabilité.  
    Le systeme n'est donc pas stable.

    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controllability

    Is the linearized model controllable?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Étude de la contrôlabilité:

    D'après le critère de KALMAN le rang de la matrice suivante = 6:



    \[
    \mathcal{C} =
    \left[
    B \; \middle|\;
    AB \; \middle|\;
    A^2B \; \middle|\;
    A^3B \; \middle|\;
    A^4B \; \middle|\;
    A^5B
    \right]
    \
    \]

    Nous avons :

    \[
    A =
    \begin{bmatrix}
    0 & 1 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & -g & 0 \\
    0 & 0 & 0 & 1 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 1 \\
    0 & 0 & 0 & 0 & 0 & 0
    \end{bmatrix}, \quad
    B =
    \begin{bmatrix}
    0 & 0 \\
    0 & -g \\
    0 & 0 \\
    \frac{1}{M} & 0 \\
    0 & 0 \\
    0 & -\frac{3g}{l}
    \end{bmatrix}
    \]


    Calcul de \( AB \):

    \[
    AB = A \cdot B =
    \begin{bmatrix}
    0 & -g \\
    g^2 & 0 \\
    \frac{1}{M} & 0 \\
    0 & 0 \\
    0 & -\frac{3g}{l} \\
    0 & 0
    \end{bmatrix}
    \]



    Calcul de \( A^2B \):

    \[
    A^2B = A \cdot AB =
    \begin{bmatrix}
    g^2 & 0 \\
    0 & g^2 \\
    0 & 0 \\
    0 & -\frac{3g}{l} \\
    0 & 0 \\
    0 & 0
    \end{bmatrix}
    \]



    Calcul de \( A^3B \):

    \[
    A^3B = A \cdot A^2B =
    \begin{bmatrix}
    0 & g^2 \\
    -g^3 & 0 \\
    0 & -\frac{3g}{l} \\
    0 & 0 \\
    0 & 0 \\
    0 & 0
    \end{bmatrix}
    \]



    Calcul de \( A^4B \):

    \[
    A^4B = A \cdot A^3B =
    \begin{bmatrix}
    -g^3 & 0 \\
    0 & -g^3 \\
    0 & 0 \\
    0 & 0 \\
    0 & 0 \\
    0 & 0
    \end{bmatrix}
    \]



    Calcul de \( A^5B \):

    \[
    A^5B = A \cdot A^4B =
    \begin{bmatrix}
    0 & -g^3 \\
    g^4 & 0 \\
    0 & 0 \\
    0 & 0 \\
    0 & 0 \\
    0 & 0
    \end{bmatrix}
    \]


    La Matrice de contrôlabilité :


    \[
    \mathcal{C} =
    \left[
    B \; \middle|\;
    AB \; \middle|\;
    A^2B \; \middle|\;
    A^3B \; \middle|\;
    A^4B \; \middle|\;
    A^5B
    \right]
    \in \mathbb{R}^{6 \times 12}
    \]

    \[
    \mathcal{C} =
    \begin{bmatrix}
    0 & 0 & 0 & -g & g^2 & 0 & 0 & g^2 & -g^3 & 0 & 0 & -g^3 \\
    0 & -g & g^2 & 0 & 0 & g^2 & -g^3 & 0 & 0 & -g^3 & g^4 & 0 \\
    0 & 0 & \frac{1}{M} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
    \frac{1}{M} & 0 & 0 & 0 & 0 & -\frac{3g}{l} & 0 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & -\frac{3g}{l} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
    0 & -\frac{3g}{l} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0
    \end{bmatrix}
    \]

    Nous observons que les 6 lignes sont **linéairement indépendantes** : on vérifie directement que la matrice est de rang = 6


    \[
    \Rightarrow \boxed{\text{rang}(\mathcal{C}) = 6}
    \]



    Le système linéarisé est contrôlable

    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Lateral Dynamics

    We limit our interest in the lateral position $x$, the tilt $\theta$ and their derivatives (we are for the moment fine with letting $y$ and $\dot{y}$ be uncontrolled). We also set $f = M g$ and control the system only with $\phi$.

    What are the new (reduced) matrices $A$ and $B$ for this reduced system?
    Check the controllability of this new system.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Analyse de la contrôlabilité du système

    Les matrices réduites \( A_r \) et \( B_r \) s'écrivent comme suit :

    \[
    A_r = 
    \begin{pmatrix}
    0 & 1 & 0 & 0 \\
    0 & 0 & -g & 0 \\
    0 & 0 & 0 & 1 \\
    0 & 0 & 0 & 0
    \end{pmatrix}
    \quad
    B_r = 
    \begin{pmatrix}
    0&-1 \\
    0&0 \\
    0&\frac{-3g}{l} \\
    0&0 \\
    \end{pmatrix}
    \]

    Le critère de **Kalman** donne la stabilité sous condition d'avoir :

    \[
    \text{rg}(C) = 4, \quad \text{avec} \quad 
    C = \left[ B, AB, A^2B, A^3B \right]
    \]

    \[
    C = 
    \begin{pmatrix}
    0 & 0 & 0 & -1 & 0 & 0 &0 & -\frac{3g^2}{\ell} \\
    0 & -1 & 0 & 0 & 0& -\frac{3g^2}{\ell} & 0& 0 \\
    0 & 0 & 0 & -\frac{3g}{\ell} & 0 & 0& 0 & 0 \\
    0 &-\frac{-3g}{\ell} & 0 & 0 & 0 & 0 & 0& 0
    \end{pmatrix}
    \]

    Effectivement, \( \text{rg}(C) = 4 \), **le système est donc contrôlable**.

    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Linear Model in Free Fall

    Make graphs of $y(t)$ and $\theta(t)$ for the linearized model when $\phi(t)=0$,
    $x(0)=0$, $\dot{x}(0)=0$, $\theta(0) = 45 / 180  \times \pi$  and $\dot{\theta}(0) =0$. What do you see? How do you explain it?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""


    On considère les matrices :

    \[
    A_r = \begin{pmatrix}
    0 & 1 & 0 & 0 \\
    0 & 0 & -g & 0 \\
    0 & 0 & 0 & 1 \\
    0 & 0 & \frac{3g}{\ell} & 0
    \end{pmatrix}, \quad
    B_r = \begin{pmatrix}
    0 & -1 \\
    0 &0 \\
    0 & -\frac{3g}{l}\\
    0& 0 \\
    \end{pmatrix}
    \]

    ## Évolution de l'état du système

    On considère une perturbation de l'état \(\Delta X\) et une commande \(\Delta u\), alors :

    \[
    A_r \cdot \Delta X =
    \begin{pmatrix}
    0 & 1 & 0 & 0 \\
    0 & 0 & -g & 0 \\
    0 & 0 & 0 & 1 \\
    0 & 0 & 0 & 0
    \end{pmatrix}
    \begin{pmatrix}
    x \\
    \dot{x} \\
    \theta   \\
    \dot{\theta}
    \end{pmatrix}
    =
    \begin{pmatrix}
    \dot{x} \\
    -g\theta \\
    \dot{\theta} \\
    0
    \end{pmatrix}
    \]

    \[
    B_r \cdot \Delta u =
    \begin{pmatrix}
    0&0 \\
    0&-1 \\
    0&0 \\
    0&-\frac{3g}{l}
    \end{pmatrix}
    \begin{pmatrix}
    0 \\
    \varphi \\
    \end{pmatrix}=
    \begin{pmatrix}
    0 \\
    -\varphi \\
    0 \\
    -\frac{3g}{\ell} \varphi
    \end{pmatrix}
    \]

    Donc :

    \[
    \Delta \dot{X} = A_r \cdot \Delta X + B_r \cdot \Delta u
    =
    \begin{pmatrix}
    \dot{x} \\
    -g\theta - \varphi \\
    \dot{\theta} \\
    \frac{3g}{\ell} \varphi
    \end{pmatrix}
    \]


    Alors :

    \[
    \ddot{x} = -g\theta
    \]

    \[
    \ddot{\theta} = 0
    \]

    """
    )
    return


@app.cell(hide_code=True)
def _(g, np, plt, sci):

    # Système différentiel
    def system(t, y):
        x, dx, theta, dtheta = y
        return [dx, -g * theta, dtheta, 0]

    # Conditions initiales : x(0), x'(0), theta(0), theta'(0)
    y0 = [0.0, 0.0, (np.pi)/4, 0.0]  # Exemple : petite oscillation initiale pour theta

    # Intervalle de temps
    t_span = (0, 10)
    t_eval = np.linspace(t_span[0], t_span[1], 500)

    # Résolution numérique
    sol = sci.solve_ivp(system, t_span, y0, t_eval=t_eval)

    # Tracé des courbes
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(sol.t, sol.y[0], label='x(t)', color='blue')
    plt.xlabel('Temps [s]')
    plt.ylabel('x(t)')
    plt.title('Position x(t)')
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(sol.t, sol.y[2], label='theta(t)', color='orange')
    plt.xlabel('Temps [s]')
    plt.ylabel('theta(t) [rad]')
    plt.title('Angle theta(t)')

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Manually Tuned Controller

    Try to find the two missing coefficients of the matrix 

    $$
    K =
    \begin{bmatrix}
    0 & 0 & ? & ?
    \end{bmatrix}
    \in \mathbb{R}^{4\times 1}
    $$ 

    such that the control law 

    $$
    \Delta \phi(t)
    = 
    - K \cdot
    \begin{bmatrix}
    \Delta x(t) \\
    \Delta \dot{x}(t) \\
    \Delta \theta(t) \\
    \Delta \dot{\theta}(t)
    \end{bmatrix} \in \mathbb{R}
    $$

    manages  when
    $\Delta x(0)=0$, $\Delta \dot{x}(0)=0$, $\Delta \theta(0) = 45 / 180  \times \pi$  and $\Delta \dot{\theta}(0) =0$ to: 

      - make $\Delta \theta(t) \to 0$ in approximately $20$ sec (or less),
      - $|\Delta \theta(t)| < \pi/2$ and $|\Delta \phi(t)| < \pi/2$ at all times,
      - (but we don't care about a possible drift of $\Delta x(t)$).

    Explain your thought process, show your iterations!

    Is your closed-loop model asymptotically stable?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controller Tuned with Pole Assignment

    Using pole assignement, find a matrix

    $$
    K_{pp} =
    \begin{bmatrix}
    ? & ? & ? & ?
    \end{bmatrix}
    \in \mathbb{R}^{4\times 1}
    $$ 

    such that the control law 

    $$
    \Delta \phi(t)
    = 
    - K_{pp} \cdot
    \begin{bmatrix}
    \Delta x(t) \\
    \Delta \dot{x}(t) \\
    \Delta \theta(t) \\
    \Delta \dot{\theta}(t)
    \end{bmatrix} \in \mathbb{R}
    $$

    satisfies the conditions defined for the manually tuned controller and additionally:

      - result in an asymptotically stable closed-loop dynamics,

      - make $\Delta x(t) \to 0$ in approximately $20$ sec (or less).

    Explain how you find the proper design parameters!
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controller Tuned with Optimal Control

    Using optimal, find a gain matrix $K_{oc}$ that satisfies the same set of requirements that the one defined using pole placement.

    Explain how you find the proper design parameters!
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Validation

    Test the two control strategies (pole placement and optimal control) on the "true" (nonlinear) model and check that they achieve their goal. Otherwise, go back to the drawing board and tweak the design parameters until they do!
    """
    )
    return


if __name__ == "__main__":
    app.run()
