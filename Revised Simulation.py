#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries
import random
import pandas as pd
import numpy as np
import math 
import simpy


# In[2]:


# Read in Excel file
chargepoint = pd.read_excel("chargepoint.xlsx")
chargepoint = chargepoint.fillna(0)
chargepoint["NUM_CHARGER"] = 0


# In[ ]:


# Print first five rows of the dataframe
chargepoint.head()


# In[4]:


# Loop through each row of data in the chargepoint dataframe

for i in range(len(chargepoint)):
#for i in range(10):
    print("row:",i)
    RANDOM_SEED = 42
    NUM_CHARGER1 = chargepoint.loc[i,"EV Level1 EVSE Num"]
    print("# of level 1 chargers:",NUM_CHARGER1)
    NUM_CHARGER2 = chargepoint.loc[i,"EV Level2 EVSE Num"]  
    print("# of level 2 chargers:",NUM_CHARGER2)
    NUM_CHARGER3 = chargepoint.loc[i,"EV DC Fast Count"]
    print("# of level 3 chargers:",NUM_CHARGER3)
    x = random.randrange(120, 480)
    # Prints random number generated in the range above
    print(x)

    #chargepoint.loc[i,"NUM_CHARGER"]

    y = NUM_CHARGER1*x/random.randrange(360, 1320) + NUM_CHARGER2 + NUM_CHARGER3*x/random.randrange(3000, 6000)/100
    y = math.ceil(y)
    print("y is:",y)
    NUM_CHARGERS =  y
    chargeTIME =  random.randrange(30,120)         # Minutes it takes to charge the EV
    T_INTER =  random.randrange(6,60)
    SIM_TIME = 300
    wait = []
    

    class Location(object):
        """A charge location has a limited number of chargers (``NUM_CHARGERS``) to
        charge EVs in parallel.

        EVs have to request one of the chargers. When they got one, they
        can start the charging processes and wait for it to finish (which
        takes ``chargetime`` minutes).

        """

        def __init__(self, env, num_chargers, chargetime):
            self.env = env
            self.machine = simpy.Resource(env, num_chargers)
            self.chargetime = chargetime


        def charge(self, ev):
            """The charging processes. It takes a ``ev`` processes and tries
            to charge it."""

            yield self.env.timeout(chargeTIME)
            print("%s charged to %d%%." %
                  (ev, random.randint(50, 99)))

    def ev(env, name, cw):
        """The ev process (each EV has a ``name``) arrives at the charging location and requests a charger.
        It then starts the charging process, waits for it to finish and
        leaves to never come back ...
        """

        print('%s arrives at the charging location at %.2f.' % (name, env.now))
        arrival_time = env.now
        with cw.machine.request() as request:
            # Request a charging station
            yield request

            print('%s connects to the charger at %.2f.' % (name, env.now))
            enter_time = env.now
            #Charge for 'chargeTIME' time
            yield env.process(cw.charge(name))

            
            print('%s disconnects from the charger at %.2f.' % (name, env.now))

            wait_time = enter_time - arrival_time
            wait.append(wait_time)
            print("%s's wait time is %.2f." % (name, wait_time))
            #print("Wait Time:",wait_time)
        
        

    def setup(env, num_chargers, chargetime, t_inter):
        """Create a charging location, a number of initial EVs and keep creating EVs
        with exponential interarrival times in minutes."""

        # Create the charging location
        charge_loc = Location(env, num_chargers, chargetime)

        # Create 4 initial EVs
        for i in range(4):
            env.process(ev(env, 'EV %d' % i, charge_loc))

        # Create more EVs with exponential interarrival times while the simulation is running

        while True:
            t = random.expovariate(1.0/t_inter)
            print("t:",t)
            
            # wait for the time passed as the argument to be elapsed on the computer’s simulation clock
            yield env.timeout(t)
            #yield env.timeout(random.randint(t_inter - 2, t_inter + 2))
            i += 1
            env.process(ev(env, 'EV %d' % i, charge_loc))


    # Setup and start the simulation
    print('Charging Location')
    #print('Check out http://youtu.be/fXXmeP9TvBg while simulating ... ;-)')
    random.seed(RANDOM_SEED)  # This helps reproducing the results

    # Create an environment and start the setup process
    env = simpy.Environment()
    env.process(setup(env, NUM_CHARGERS, chargeTIME, T_INTER))

    # Execute!
    env.run(until=SIM_TIME)
    print(wait)
    chargepoint.loc[i,"Wait_Time min"] = np.mean(wait)
    print("Avg Wait Time:",np.mean(wait))


# In[5]:


# Save wait times to csv
chargepoint.to_csv("Rev_Wait_Time.csv")


# In[ ]:


### Code Complete

