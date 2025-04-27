import rebound
import airball
import airball.units as u
import numpy as np
from sys import argv

twopi = 2.0*np.pi

def setup(ms=[9.55e-04, 2.86e-04, 4.37e-05, 5.15e-05]):
    sim = rebound.Simulation()
    sim.add(m=1)
    sim.add(m=ms[0], a=5.2038, f='uniform')
    sim.add(m=ms[1], a=9.5397, f='uniform')
    sim.add(m=ms[2], a=19.1927, f='uniform')
    sim.add(m=ms[3], a=30.0760, f='uniform')
    sim.move_to_com()
    return sim


Nstars = int(argv[1])
index = int(argv[2])
stars = airball.Stars(f'data/giants-{Nstars}.stars')

sim = setup()
filename = f'secular/giants-{Nstars}-{index:05d}.sa'
sim.save_to_file(filename, delete_file=True)
res = airball.flyby(sim, stars[index], hash='flybystar')

airball.tools.rotate_into_plane(sim, 'invariable')
orbs = sim.orbits()
ai = np.asarray([o.a for o in orbs])
ei = np.asarray([o.e for o in orbs])
mi = np.sum([o.m for o in sim.particles])
sim.dt = airball.tools.timestep_for_perihelion_resolution(sim)
sim.integrator = 'whckl'
sim.ri_whfast.safe_mode = False
sim.ri_whfast.keep_unsynchronized = True
sim.exit_max_distance = 100
sim.move_to_com()

step_size = int(np.round((1e4*twopi)/sim.dt))
sim.save_to_file(filename, step=step_size, delete_file=False)
try: sim.integrate(sim.t + 2047*step_size*sim.dt)
except rebound.Escape as esc: sim.save_to_file(filename, delete_file=False)