from casadi import *
from multiprocessing import Pool
import time

x = SX.sym("x")

solver = nlpsol("solver","ipopt",{"x":x,"f":sin(x)**2})

def mysolve(x0):
  return solver(x0=x0)["x"]

start_time1=time.time()
for i  in range(6):
  solution = mysolve(3*i)
final_time1=time.time()-start_time1


p = Pool(4)
start_time=time.time()
print(p.map(mysolve, [0, 3, 6,9,12,15]))
print("final time=",time.time()-start_time)
print("final time 1=",final_time1)