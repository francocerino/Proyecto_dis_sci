from scipy.integrate import odeint
import numpy as np
from skreducedmodel import ReducedModel

def pend(y, t, b, λ):
    θ, ω = y
    dydt = [ω, -b*ω - λ*np.sin(θ)]

    return dydt

def test_ReducedModelFit():
    
    b = 0.2
    y0 = [np.pi/2, 0.]
    
    param = np.linspace(1,5,101)
    times = np.linspace(0,50,1001)
     
    training = []
    for λ in param:
        sol = odeint(pend,y0, times, (b,λ))
        training.append(sol[:,0])
    
    training_set = np.array(training) 
    parameters = param
    physical_points = times
    nmax = 10
        
    model = ReducedModel(
                     seed_global_rb = 0,
                     greedy_tol = 1e-16,
                     lmax = 1, 
                     nmax = nmax,
                     normalize = True
                     )
    
    rb = model.fit(
               training_set = training_set, 
               parameters = parameters, 
               physical_points = physical_points,
               )
    
    print(rb.errors[nmax-1],rb.errors[0])
    
    assert rb.errors[0]>rb.errors[nmax-1]
    assert rb.errors[5]>rb.errors[nmax-1]
    assert len(rb.indices) == nmax
    assert rb.indices[9] == 92  ## todos los numeros salieron del ejemplo del Pendulo