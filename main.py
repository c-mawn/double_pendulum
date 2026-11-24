from pendulum import Pendulum
from viz import PendulumViz

t1 = 0.2
t2 = 0.9
viz = PendulumViz(Pendulum(theta1=t1, theta2=t2))  # your starting angles
viz.save_clip(f"gifs/{t1},{t2}.gif", seconds=10)
