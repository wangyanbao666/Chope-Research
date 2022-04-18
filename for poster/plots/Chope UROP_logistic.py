# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 02:18:42 2020

@author: joshu
"""

import numpy as np
import time
# import numba
import random as rand
import math
import matplotlib
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
plt.style.use('seaborn-pastel')

import sys
import copy

from mpl_toolkits.mplot3d import Axes3D

start_time=time.time()

"""
Information for entry of each array

customer_list= [  0   ,    1    ,   2  ,   3  ,     4     ,     5    ,     6    ,    7    ,    8    ]
customer_list= [is_inh, is_chope, is_lf, is_ls, is_seating, wait_seat, wait_food, eat_time, time_inh]

---
seat_list = [    0   ,     1    ]
seat_list = [is_taken, seat_time]
"""

looking_for_seat_ls=[]

# @numba.jit
def seat_taken(seats):
    num_seat_taken=0
    for seat in seats:
        if seat[0]==1:
            num_seat_taken+=1
    crowdedness= num_seat_taken/len(seats)
    return num_seat_taken, crowdedness

# @numba.jit
def get_status(customers):
    num_queue = 0
    num_look_seat = 0
    num_eating = 0
    num_in_hawker=0
    
    for customer in customers:
        if customer[0]==1:
            num_in_hawker +=1
            if customer[2]==1:
                num_queue+=1
            elif customer[2]==2:
                num_look_seat+=1
            elif customer[2]==3:
                num_eating+=1
    return num_queue, num_look_seat, num_eating,num_in_hawker
        

def initialize_parameters(hyperparams):
    customer_list=np.zeros((hyperparams[0],7))
    seat_list =np.zeros((hyperparams[1],2))
    inflow_rate = hyperparams[2]
    feed_time = hyperparams[3]
    num_stalls = hyperparams[4]
    prepare_food_time = hyperparams[5]
    return customer_list, seat_list, inflow_rate, feed_time, num_stalls,prepare_food_time

# @numba.jit
def update_hawker_status(customers,seats,feed_time,t):
    leaving_customers=[]
    for person in range (0,len(customers)):

        ###update value of counter for consumer in hawker(looking for seat, eating time, and looking for food)
        if customers[person][0]==1:
            if customers[person][2]==2:  #for customer looking for seat, increase looking seat time counter
                customers[person][3]+=1    
            if customers[person][2]==3:  #for customer seating, reduce eating time so we can kick them later
                customers[person][5]-=1
            if customers[person][2]==1:  #for customer looking for food (queue), reduce queue time
                customers[person][4]-=1
            
        ###once a customer finish queueing
        if customers[person][0]==1 and customers[person][2]==1 and customers[person][4]==0:
            customers[person][2]=0
            if customers[person][1]==0: #if they non_chopper, they will look for seat
                customers[person][2]=2
                looking_for_seat_ls.append(customers[person])
            elif customers[person][1]==1: #if they chopper, they sit down
                customers[person][2]=3
                customers[person][5]=feed_time
                
        ###if a customer is sitting and has finished eating, remove them
        if customers[person][0]==1 and customers[person][2]==3 and customers[person][5]==0:
            leaving_customers.append([customers[person][3],customers[person][6],t])
            customers[person][0]=0
    
    ###update seat status    
    for seat in range (0, len(seats)):
        if seats[seat][0]==1: #if seat is occupied, subtract occupied time by one
            seats[seat][1]-=1
        if seats[seat][1]==0: #when counter reach zero, seat is empty
            seats[seat][0]=0
    return customers, seats, leaving_customers

# @numba.jit
def add_customer(customers,chop_prob,inflow_rate_prob,num_queue,num_stalls,prepare_food_time,crowdedness,t):
    inflow_rate = rand.randint(0,inflow_rate_prob)
    
    for rate in range (0,inflow_rate): 
            for people in range (0,len(customers)):
                ###randomiser for choping probability
                randomiser=rand.random()*100
                if customers[people][0]==0:
                    customers[people][0]=1
                    if randomiser<chop_prob:
                        customers[people][1]=1
                        customers[people][2]=2
                        looking_for_seat_ls.append(customers[people])
                    else:
                        customers[people][1]=0
                        customers[people][2]=1
                        customers[people][4]=prepare_food_time*math.floor(1+num_queue/num_stalls)
                        customers[people][6]=prepare_food_time*math.floor(1+num_queue/num_stalls)
                        num_queue+=1
                    break

    return customers

# @numba.jit
def assign_seat(customers, seats,num_look_seat, num_look_food, feed_time,num_stalls, prepare_food_time):
    #check for all seats
    global looking_for_seat_ls
    counter=0
    for seat in range(0,len(seats)):
        
        #check for all empty seats
        if seats[seat][0] == 0:
            if len(looking_for_seat_ls) != 0:
                people = looking_for_seat_ls.pop(0)
                # print('p', people)
                if people[1] == 1:
                    people[2] = 1
                    people[4] = prepare_food_time * math.floor(1 + num_look_food / num_stalls)
                    people[6] = prepare_food_time * math.floor(1 + num_look_food / num_stalls)
                    seats[seat][0] = 1
                    seats[seat][1] = feed_time + prepare_food_time * math.floor(1 + num_look_food / num_stalls)

                    num_look_food += 1

                else:
                    people[2] = 3
                    people[5] = feed_time

                    seats[seat][0] = 1
                    seats[seat][1] = feed_time
            ###run only for all looking for seats
            # while counter<num_look_seat:
            #
            #     randomiser=rand.randint(0,len(customers)-1) #change distribution?
            #
            #     ###update status from look_seat to look_food for choper (random)
            #     if customers[randomiser][0]==1 and customers[randomiser][2]==2 and customers[randomiser][1]==1:
            #         customers[randomiser][2]=1
            #         customers[randomiser][4]=prepare_food_time*math.floor(1+num_look_food/num_stalls)
            #         customers[randomiser][6]=prepare_food_time*math.floor(1+num_look_food/num_stalls)
            #
            #
            #         seats[seat][0]=1
            #         seats[seat][1]=feed_time+prepare_food_time*math.floor(1+num_look_food/num_stalls)
            #         counter+=1
            #
            #         num_look_food+=1
            #         break
            #
            #     ###update status from look_seat to eat for non-choper (random)
            #     if customers[randomiser][0]==1 and customers[randomiser][2]==2 and customers[randomiser][1]==0:
            #
            #         customers[randomiser][2]=3
            #         customers[randomiser][5]=feed_time
            #
            #         seats[seat][0]=1
            #         seats[seat][1]=feed_time
            #         counter+=1
            #         break
    return customers,seats



def variable_prob():
    #initialise all parameters and hyperparameters
    g_counter=0
    repeat=3
    prob_step = 10

    chop_probs=[np.round(x,5) for x in np.arange(0,100+prob_step,prob_step)]
    chop_probs=np.repeat(chop_probs,repeat)
    running_average = 20
    simulation_length = 500
    critical_time = 30

    max_custs = simulation_length
    # hyperparameters = [[max_custs,70,4,25,3,1],[max_custs,70,5,25,3,1],[max_custs,70,6,25,3,1],[max_custs,60,4,25,3,1],[max_custs,60,5,25,3,1],[max_custs,60,6,25,3,1]]
    hyperparameters = [[4000, 150, 6, 25, 5, 1], [4000, 150, 7, 25, 5, 1], [4000, 150, 8, 25, 5, 1]]

    for hyperparam in hyperparameters:
        global looking_for_seat_ls
        looking_for_seat_ls=[]
        _, _ ,inflow_rate, feed_time, num_stalls, prepare_food_time = initialize_parameters(hyperparam)

        customers, seats, _, _, _, _ = initialize_parameters(hyperparam)
        waiting_time_data = []
        prob_data = []

        for t in range(simulation_length):
            
            num_seat_taken, crowdedness =seat_taken(seats)
            customers, seats, leaving_customers = update_hawker_status(customers,seats,feed_time,t)
            num_queue, num_look_seat, num_eating, num_in_hawker = get_status(customers)

            print(leaving_customers)
            chop_prob = 4 * crowdedness * (1 - crowdedness) * 100
            # chop_prob = math.exp(12 * crowdedness - 6) / (1 + math.exp(12 * crowdedness - 6)) * 100

            if num_in_hawker<len(customers):
                customers=add_customer(customers,chop_prob,inflow_rate,num_queue,num_stalls,prepare_food_time,crowdedness,t)
                
            num_queue, num_look_seat, num_eating, num_in_hawker = get_status(customers)
            customers, seats = assign_seat(customers, seats, num_look_seat, num_queue, feed_time, num_stalls, prepare_food_time)

            prob_data.append(chop_prob)

            # record waiting time of each customers, only for those entering the hawker centre from critical time onwards
            waiting_time=0
            if len(leaving_customers)!=0:
                waiting_time = sum([cust_time[0] + cust_time[1] for cust_time in leaving_customers])/len(leaving_customers)
            waiting_time_data.append(waiting_time)
            

        avg_waiting_time = [np.mean(waiting_time_data[i:i + running_average]) for i in
                            range(len(waiting_time_data) - running_average)]
        trnc_probs = prob_data[:-running_average]

        #plot the macro state of hawker centre (average waiting time against choping probability)
        plt.scatter(trnc_probs,avg_waiting_time,s=3, c=np.arange(len(avg_waiting_time)), cmap='coolwarm')
        plt.title("Choping Probability against waiting time")
        plt.figtext(0.91, 0.6,
                    "Seats:{}\nMax_inflow:{}\nEat_time:{}\nStalls:{}\nFood_prep:{}\nSimulation Length:{}\nRunning Average:{}".format(
                        hyperparam[1], hyperparam[2], hyperparam[3], hyperparam[4], hyperparam[5],
                        simulation_length, running_average), fontsize=6,
                    bbox=dict(facecolor='none', edgecolor='black'))
        plt.xlabel('Choping probability')
        plt.ylabel('Average waiting time')
        plt.savefig('time\quadratic\Run number2 {} Choping Probability against waiting time'.format(g_counter),
                    bbox_inches='tight', dpi=300)
        plt.clf()

        print(1)
        g_counter+=1
   
      
variable_prob()    
print(time.time()-start_time)