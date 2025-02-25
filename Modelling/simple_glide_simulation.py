# -*- coding: utf-8 -*-
"""
@author: kernke
"""

import h5py
import numpy as np
from numba import njit
import numba
from datetime import datetime

# use clock inside numba njit functions
#https://github.com/numba/numba/issues/4003
import ctypes
import platform
if platform.system() == "Windows":
    from ctypes.util import find_msvcrt
    __LIB = find_msvcrt()
    if __LIB is None:
        __LIB = "msvcrt.dll"
else:
    from ctypes.util import find_library
    __LIB = find_library("c")
clock = ctypes.CDLL(__LIB).clock
clock.argtypes = []



@njit
def single_line_exponential(E_0,E_delta,Temperature,duration,frequency,sampling_dt,fixed_prob=0,
                            E_barrier=None,remaining_time=None,prev_temp=None,timelim_sec=30):
    """
    Compute the temporal evolution of a single dislocation at a single temperature. 
    The dislocation is either created in the beginning or the evolution of an already 
    simulated dislocation can be continued by passing its last energy barrier, remainging time 
    and previous temperature.

    Args:
        E_0 (float): constant minimum energy in eV.
        E_delta (float): mean of exponential energy-distribution in eV.
        Temperature (float): temperature in Kelvin.
        duration (float): amount of time at given temperature in seconds.
        frequency (float): Debye frequency in 1/s (per second)
        sampling_dt (float): sampling time step in seconds.
        fixed_prob (float, optional): constant probability (between 0 and 1) of permanent blocking. Defaults to 0.
        E_barrier (float, optional): starting energy barrier in eV. Defaults to None.
        remaining_time (float, optional): time left to cross previous barrier in seconds. Defaults to None.
        prev_temp (float, optional): temperature of previous run. Defaults to None.
        timelim_sec (float, optional): time limit for computation in seconds. Defaults to 30.

    Returns:
        history_of_glide_steps (array): number of glide steps since the beginning of this simulation (cumulative summed)
        history_of_E_barriers (array): values of E_barrier at sampling times (between sampling time windows)
        remaining_time (float): remaining time on last barrier in seconds
        Temperature (float): used temperature (same as input) in Kelvin

    """
    kB=8.617*10**-5 # Boltzmann constant [eV/K]
    kBT=kB*Temperature # energy given by temperature vibrations in eV
    Nequisteps=int(duration//sampling_dt)+1 # number of simulation steps
    t=0 # start time
    t_end=duration*frequency # end time in multiples of inverse-Debye-frequency
    sample_dt=sampling_dt*frequency # temporal step size in multiples of inverse-Debye-frequency
    
    # initialize arrays to save the temporal evolution
    history_of_glide_steps=np.empty(Nequisteps,dtype=numba.int64)
    history_of_E_barriers=np.empty(Nequisteps)
    
    number_of_glide_steps=np.int64(0)
    history_of_glide_steps[0]=number_of_glide_steps

    if E_barrier is None:
        E_barrier=E_0+np.random.exponential(E_delta) #create a new energy barrier
    history_of_E_barriers[0]=E_barrier
    
    # the negative value -1 for the energy barrier represents permanent blocking
    if E_barrier == -1:
        dt=2*t_end # with dt > t_end the simulation loop will end immediately 
    else:
        if remaining_time is None:
            dt=np.random.exponential(np.exp(E_barrier/kBT))+1 
            # +1 prevents time steps below the inverse Debye frequency
        else:
            dt=remaining_time*frequency*np.exp(E_barrier/kBT)/np.exp(E_barrier/(kB*prev_temp))
    
    counter=1 # just counts
    prev_temporal_index=1 # corresponds to the number of the sampling time window
    
    timelim_millisec=timelim_sec*1000
    computation_time0=clock()
    while True:
        computation_time1=clock()
        if computation_time1-computation_time0 > timelim_millisec:
            print("computation time above threshold")
            break
        
        t_new=t+dt
        
        # end the simulation loop, if t+dt exceeds the duration of the simulation
        if t_new > t_end: 
            history_of_glide_steps[prev_temporal_index:]=number_of_glide_steps
            history_of_E_barriers[prev_temporal_index:]=E_barrier
            dt=t_new-t_end
            break
        
        # because dt is a variable time step, the sampling time window index needs to be calculated
        temporal_index=int(t_new//sample_dt)+1
        
        # if temporal_index < counter the time to cross the barrier did not reach the next sampling
        # time window, so the time step and the glide step will be executed, but not immediately
        # written into the array for saving
        if temporal_index>counter:
            history_of_glide_steps[prev_temporal_index:temporal_index]=number_of_glide_steps
            history_of_E_barriers[prev_temporal_index:temporal_index]=E_barrier
            counter+=1
            prev_temporal_index=temporal_index
        
        t+=dt
        number_of_glide_steps+=1
        
        if fixed_prob>np.random.uniform():
            E_barrier=-1 # the negative value -1 for the energy barrier represents permanent blocking
            dt=2*t_end # with dt > t_end the simulation loop will end immediately 
        
        else:
            E_barrier=E_0+np.random.exponential(E_delta) #create a new energy barrier
            dt=np.random.exponential(np.exp(E_barrier/kBT))+1 
            # +1 prevents time steps below the inverse Debye frequency
    
    return history_of_glide_steps,history_of_E_barriers,dt/frequency,Temperature


#%% simulation
def simulation(simulation_parameters,printing=True):
    """
    wrapper to execute single_line_exponential multiple times 
    to obtain the given sequence of temperatures and
    the given number of lines

    Args:
        simulation_parameters (dict): containing all information.
        printing (bool, optional): Defaults to True.

    Returns:
        lines (dict): result.
    """
    
    N_lines=simulation_parameters["number_of_lines"]
    
    lines=dict()
    lines["parameters"]=simulation_parameters
    initial_energy=simulation_parameters["initial_energy"]
    
    old_energies=[initial_energy for i in range(N_lines)]
    remaining_times=[None for i in range(N_lines)]
    previous_temps=[None for i in range(N_lines)]

    for i in range(len(simulation_parameters["temperatures"])):
        
        T=simulation_parameters["temperatures"][i]+273.15
        duration=simulation_parameters["durations"][i]
        #if printing:
        #    print(simulation_parameters["temperatures"][i])

        Nequisteps=int(duration//simulation_parameters["sampling_time_intervals"][i])+1
        
        istring=str(i)
        lines[istring]=dict()
        lines[istring]["temperature_K"]=T
        lines[istring]["temperature_C"]=simulation_parameters["temperatures"][i]
        lines[istring]["times"]=np.arange(Nequisteps,dtype=np.float64)*simulation_parameters["sampling_time_intervals"][i]
        lines[istring]["lengths"]=np.empty((N_lines,Nequisteps))
        lines[istring]["energies"]=np.empty((N_lines,Nequisteps))

        for j in range(N_lines):

            glide_length,E_barriers,remaining_time,previous_temp = single_line_exponential(
                simulation_parameters["location_parameter"],
                simulation_parameters["scale_parameter"],
                T,duration,
                simulation_parameters["frequency"],
                simulation_parameters["sampling_time_intervals"][i],
                fixed_prob=simulation_parameters["fixed_probability"],
                E_barrier=old_energies[j],
                remaining_time=remaining_times[j],
                prev_temp=previous_temps[j],
                timelim_sec=simulation_parameters["single_line_computation_time_threshold"]
                )

            remaining_times[j]=remaining_time
            previous_temps[j]=previous_temp
            old_energies[j]=E_barriers[-1]# last barrier

            lines[istring]["lengths"][j]=glide_length*simulation_parameters["length_conversion"]
            lines[istring]["energies"][j]=E_barriers

    return lines
#%%
def multi_simulation(number_of_simulations,simulation_parameters,printing=True,velocity_bins=None,
                     minimum_velocity=0,velocity_conversion_factor=1,number_of_full_saves=3,
                     timeoffset=0):
    """
    execute multiple simulations with the same parameters

    Args:
        number_of_simulations (int):.
        simulation_parameters (dict): input parameters.
        printing (bool, optional):  Defaults to True.
        velocity_bins (list/array, optional): bins for a histogram of velocities. Defaults to None.
        minimum_velocity (float, optional):  minimum velocity, below which activity is ignored.
            (useful for comparison to real data, where single atom changes usually can not be resolved).
            Defaults to 0.
        velocity_conversion_factor (float, optional): convert velocities e.g. from m/s to microns/s 
            by using 10**6.. Defaults to 1.
        number_of_full_saves (int, optional): saving full history_of_glide_steps and history_of_E_barriers.
            Defaults to 3.
        timeoffset (float, optional): shift the time axis (for convenience. Defaults to 0.

    Returns:
        sim_data (dict): result.

    """
    multi_sim=None
    save_lines=[]
    for knum in range(number_of_simulations):
        if printing:
            print("simulation "+str(knum))
            
        lines=simulation(simulation_parameters,printing=printing)
        count_changes(lines,simulation_parameters["sampling_time_intervals"],minv=minimum_velocity)
        
        multi_sim=multi_sim_save_step(lines,multi_sim,velocity_bins,velocity_conversion_factor)
        
        if knum < number_of_full_saves:
            save_lines.append(lines)

    sim_data=prepare_multisim_for_saving(multi_sim,lines,velocity_bins,
                                    timeoffset=timeoffset)
    sim_data["examples"]=dict()
    for i in range(number_of_full_saves):
        sim_data["examples"][str(i)]=save_lines[i]
    
    return sim_data

#%%
def count_changes(lines,sampling_dts,minv=0):
    """
    calcuclate single simulation (multiple lines) properties of the temporal evolution
    
    Args:
        lines (dict): as produced by single_line_exponential.
        sampling_dts (list/array): sampling times.
        minv (float, optional): minimum velocity, below which activity is ignored.
            (useful for comparison to real data, where single atom changes usually can not be resolved)
            Defaults to 0.

    Returns:
        None.(results are directly written into lines)

    """
    keylist=[i for i in lines.keys() if i.isnumeric()]
    for c,i in enumerate(keylist): 
        speeds_abs=np.diff(lines[i]["lengths"],axis=-1)
        speeds=speeds_abs/sampling_dts[c]
        speeds_binary=speeds>minv
        lines[i]["velocities"]=speeds[speeds_binary]
        lines[i]["all_events"]=len(lines[i]["velocities"])
        lines[i]["event_dist"]=np.sum(speeds_binary,axis=0)
        lines[i]["length_dist"]=np.sum(speeds_abs,axis=0)
        lines[i]["total_length"]=np.cumsum(lines[i]["length_dist"])

#%% multiple sims
def multi_sim_save_step(lines,multi=None,velocity_bins=None,velocity_conversion_factor=1.):#10**6
    """
    create properties that contain results from multiple simulations
    e.g. for determining a velocity histogramm or mean values of all simulations

    Args:
        lines (dict): .
        multi (dict, optional): . Defaults to None.
        velocity_bins (list/array, optional): bins for a histogram of velocities. Defaults to None.
        velocity_conversion_factor (float, optional): convert velocities e.g. from m/s to microns/s by using 10**6.
        Defaults to 1..

    Returns:
        multi (dict): dictionary containing information from multiple simulations.
    """
    if multi is None:
        multi=dict()
    if "cumulative_events" not in multi:
        multi["cumulative_events"]=[]
    if "cumulative_distance" not in multi:
        multi["cumulative_distance"]=[]
    if "velocity_hist" not in multi:
        multi["velocity_hist"]=[]
    if "energy_dist" not in multi:
        multi["energy_dist"]=[]
    if "number_of_simulations" not in multi:
        multi["number_of_simulations"]=1
    else:
        multi["number_of_simulations"]+=1
    
    keylist=[i for i in lines.keys() if i.isnumeric()]
    
    energies=[]
    cumu_events=[]
    cumu_dist=[]

    for c,i in enumerate(keylist):
        energies.append(lines[i]["energies"])
        cumu_events.append(np.cumsum(lines[i]["event_dist"]))
        cumu_dist.append(np.cumsum(lines[i]["length_dist"]))
        
    multi["energy_dist"].append(energies)
    multi["cumulative_events"].append(cumu_events)
    multi["cumulative_distance"].append(cumu_dist)


    if velocity_bins is None:
        velmax=0
        velmin=np.inf
        for c,i in enumerate(keylist):
            velmaxi=np.max(lines[i]["velocities"] *velocity_conversion_factor)
            velmax=max(velmax,velmaxi)
            velmini=np.min(lines[i]["velocities"] *velocity_conversion_factor)
            velmin=min(velmin,velmini)
        
        velocity_bins=np.linspace(velmin, velmax)
            
            
    velos=[]
    for c,i in enumerate(keylist):
        simhist,bins=np.histogram(lines[i]["velocities"] *velocity_conversion_factor,
                                  velocity_bins)
        velos.append(simhist)
    multi["velocity_hist"].append(velos)
    
    return multi


#%%

def prepare_multisim_for_saving(multi_sim,lines,vel_bins,timeoffset=0):#,timedelta=0):

    sim_data=dict()
    sim_data["parameters"]=lines["parameters"]
        
    eventcurves=[]
    distancecurves=[]
    velocitycurves=[]
    energycurves=[]
    newtimes=[]
    
    temperatures=lines["parameters"]["temperatures"]
    stis=lines["parameters"]["sampling_time_intervals"]
    durations=lines["parameters"]["durations"]
    number_of_lines=lines["parameters"]["number_of_lines"]
    number_of_linesets=multi_sim["number_of_simulations"]
    sim_data["parameters"]["number_of_simulations"]=number_of_linesets
    
    tempnum=len(temperatures)
    for i in range(tempnum):
        Ntimesteps=int(durations[i]//stis[i])
        eventcurves.append(np.zeros([number_of_linesets,Ntimesteps]))
        distancecurves.append(np.zeros([number_of_linesets,Ntimesteps]))
        velocitycurves.append(np.zeros([number_of_linesets,len(vel_bins)-1]))
        energycurves.append(np.zeros([number_of_linesets*number_of_lines,Ntimesteps+1]))
        
        newtimes.append(np.arange(Ntimesteps)*stis[i]+timeoffset)
        timeoffset=np.max(newtimes[-1])#+timedelta
    
    for i in range(tempnum):#
        for j in range(number_of_linesets):
            eventcurves[i][j]=multi_sim["cumulative_events"][j][i]
            distancecurves[i][j]=multi_sim["cumulative_distance"][j][i]
            velocitycurves[i][j]=multi_sim["velocity_hist"][j][i]
            for k in range(number_of_lines):
                energycurves[i][j*number_of_lines+k]=multi_sim["energy_dist"][j][i][k]
    
        valid=np.ma.array(energycurves[i],
                    mask=energycurves[i] == -1)
        
        sim_data[str(i)]=dict()
        sim_data[str(i)]["energies"]=dict()
        sim_data[str(i)]["distances"]=dict()
        sim_data[str(i)]["events"]=dict()
        sim_data[str(i)]["velocities"]=dict()
        sim_data[str(i)]["blocked"]=dict()
        
        sim_data[str(i)]["time"]=newtimes[i]
        
        nblock=np.sum(np.ma.getmask(valid),axis=0)
        sim_data[str(i)]["blocked"]["number_total"]=nblock
        sim_data[str(i)]["blocked"]["ratio"]=nblock/(number_of_lines*number_of_linesets)

        sim_data[str(i)]["energies"]["mean"]=valid.mean(axis=0)
        sim_data[str(i)]["energies"]["min"]=valid.min(axis=0)
        sim_data[str(i)]["energies"]["max"]=valid.max(axis=0)
        sim_data[str(i)]["energies"]["std"]=valid.std(axis=0)
        
        sim_data[str(i)]["velocities"]["mean"]=np.mean(velocitycurves[i],axis=0)
        sim_data[str(i)]["velocities"]["min"]=np.min(velocitycurves[i],axis=0)
        sim_data[str(i)]["velocities"]["max"]=np.max(velocitycurves[i],axis=0)
        sim_data[str(i)]["velocities"]["std"]=np.std(velocitycurves[i],axis=0)

        sim_data[str(i)]["events"]["mean"]=np.mean(eventcurves[i],axis=0)
        sim_data[str(i)]["events"]["min"]=np.min(eventcurves[i],axis=0)
        sim_data[str(i)]["events"]["max"]=np.max(eventcurves[i],axis=0)
        sim_data[str(i)]["events"]["std"]=np.std(eventcurves[i],axis=0)
        
        sim_data[str(i)]["distances"]["mean"]=np.mean(distancecurves[i],axis=0)
        sim_data[str(i)]["distances"]["min"]=np.min(distancecurves[i],axis=0)
        sim_data[str(i)]["distances"]["max"]=np.max(distancecurves[i],axis=0)
        sim_data[str(i)]["distances"]["std"]=np.std(distancecurves[i],axis=0)

    return sim_data

#%%

def save_h5(lines,hdf,keypath="/"):
    keypaths=[]
    linekeys=[]
    if "ignore_list" in lines:
        ignore_list=lines["ignore_list"]
    else:
        ignore_list=[]
        
    for key, value in lines.items():
        if key in ignore_list:
            #print(key)
            pass
        else:
            if isinstance(value, np.ndarray):
                hdf.create_dataset(keypath+"/"+key,
                                   data=value,
                                   dtype='float32')
            elif isinstance(value, dict):
                linekeys.append(key)
                if keypath=="/":
                    keypaths.append(keypath+key)
                else:
                    keypaths.append(keypath+"/"+key)
            elif value is None:
                hdf[keypath].attrs[key] = "None"
            else:
                hdf[keypath].attrs[key] = value

    for i in range(len(keypaths)):
        hdf.create_group(keypaths[i])
        save_h5(lines[linekeys[i]],hdf,keypath=keypaths[i])


def lines_to_h5(lines,name=None,pref=None):
    if name is None:
        if pref is None:
            name="simulation_"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        else:
            name=pref+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name+=".h5"
    with h5py.File(name, 'w') as hdf:
        save_h5(lines, hdf)



#%% deprecated
import json
def _arrays_to_lists(indic):
    for k, v in indic.items():
        if isinstance(v, np.ndarray):
            indic[k]=v.tolist()
        elif isinstance(v, dict):
            _arrays_to_lists(v)
        elif isinstance(v, list):
            print("list encountered")
        elif isinstance(v, np.int32):
            print(k)
    return indic

def _save_sim(filename,lines):
    with open(filename+".json","w") as f:
        json.dump(lines,f)
        
def sim_loop(simulation_parameters,xbins,timeoffset,timedelta,pref,ignorelist=[]):
    multi_sim=None
    for knum in range(simulation_parameters["number_of_line_sets"]):
        #t1=time.time()
        lines=simulation(simulation_parameters,printing=False,distribution="exponential")    
        #t2=time.time()
        #print(t2-t1)
        
        count_changes(lines,simulation_parameters["sampling_time_intervals"],minv=2*10**-7)
        multi_sim=multi_sim_save_step(lines,multi_sim,xbins)
        
        #print(knum)
    
    #print(i)
    #print(len(sps))
    sim_data=prepare_multisim_for_saving(multi_sim,lines,xbins,
                                    timeoffset=timeoffset,timedelta=timedelta)
    
    sim_data["ignore_list"]=ignorelist#["0","1","2","3","4","5","6","7","8","9"]
    lines_to_h5(sim_data,pref=pref)#"multilines.h5"
    return sim_data