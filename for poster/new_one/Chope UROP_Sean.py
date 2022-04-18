# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 02:18:42 2020

@author: joshu
"""

import numpy as np
import time
import numba
import random as rand
import math
import matplotlib
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

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

@numba.jit
def seat_taken(seats):
    num_seat_taken=0
    for seat in seats:
        if seat[0]==1:
            num_seat_taken+=1
    crowdedness= num_seat_taken/len(seats)
    return num_seat_taken, crowdedness

@numba.jit
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

@numba.jit
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
            elif customers[person][1]==1: #if they chopper, they sit down
                customers[person][2]=3
                customers[person][5]=feed_time
                
        ###if a customer is sitting and has finished eating, remove them
        if customers[person][0]==1 and customers[person][2]==3 and customers[person][5]==0:
            leaving_customers.append([customers[person][3],customers[person][6],t])
            customers[person][0]=2
    
    ###update seat status    
    for seat in range (0, len(seats)):
        if seats[seat][0]==1: #if seat is occupied, subtract occupied time by one
            seats[seat][1]-=1
        if seats[seat][1]==0: #when counter reach zero, seat is empty
            seats[seat][0]=0
    return customers, seats, leaving_customers

@numba.jit
def add_customer(customers,chop_prob,inflow_rate_prob,num_queue,num_stalls,prepare_food_time,crowdedness,t):
    rand.seed(t)
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
                    else:
                        customers[people][1]=0
                        customers[people][2]=1
                        customers[people][4]=prepare_food_time*math.floor(1+num_queue/num_stalls)
                        customers[people][6]=prepare_food_time*math.floor(1+num_queue/num_stalls)
                        num_queue+=1
                    break

    return customers

@numba.jit
def assign_seat(customers, seats,num_look_seat, num_look_food, feed_time,num_stalls, prepare_food_time):
    #check for all seats
    counter=0
    for seat in range(0,len(seats)):
        
        #check for all empty seats
        if seats[seat][0]==0:
            
            ###run only for all looking for seats
            while counter<num_look_seat:
            
                randomiser=rand.randint(0,len(customers)-1) #change distribution?
               
                ###update status from look_seat to look_food for choper (random)
                if customers[randomiser][0]==1 and customers[randomiser][2]==2 and customers[randomiser][1]==1:
                    customers[randomiser][2]=1
                    customers[randomiser][4]=prepare_food_time*math.floor(1+num_look_food/num_stalls)
                    customers[randomiser][6]=prepare_food_time*math.floor(1+num_look_food/num_stalls)
                    
                                 
                    seats[seat][0]=1
                    seats[seat][1]=feed_time+prepare_food_time*math.floor(1+num_look_food/num_stalls)
                    counter+=1
                    
                    num_look_food+=1
                    break
    
                ###update status from look_seat to eat for non-choper (random)                    
                if customers[randomiser][0]==1 and customers[randomiser][2]==2 and customers[randomiser][1]==0:
                    
                    customers[randomiser][2]=3
                    customers[randomiser][5]=feed_time
                    
                    seats[seat][0]=1
                    seats[seat][1]=feed_time
                    counter+=1
                    break
    return customers,seats


def main():
    
    
    #initialise all parameters and hyperparameters
    g_counter=0
    repeat=3
    prob_step = 1

    chop_probs=[np.round(x,5) for x in np.arange(0,100+prob_step,prob_step)]
    chop_probs=np.repeat(chop_probs,repeat)   
    hyperparameters = [[1500,40,4,25,2,1],[1500,45,4,25,2,1],[1500,50,4,25,2,1],[1500,55,4,25,2,1],[1500,60,4,25,2,1],[1500,70,4,25,2,1]]
    # hyperparameters = [[1500,60,1,25,2,1],[1500,60,2,25,2,1],[1500,60,3,25,2,1],[1500,60,4,25,2,1],[1500,60,5,25,2,1],[1500,60,6,25,2,1]]
    simulation_length = 390
    critical_time = 210
    
    for hyperparam in hyperparameters:
        _, _ ,inflow_rate, feed_time, num_stalls, prepare_food_time = initialize_parameters(hyperparam)

        result_data=[]
        satisfaction_data=[]
        
        for chop_prob in chop_probs:
            customers, seats, _, _, _, _ = initialize_parameters(hyperparam)           
            customer_data=[]


            for t in range(simulation_length):
                
                num_seat_taken, crowdedness =seat_taken(seats)
                num_queue, num_look_seat, num_eating, num_in_hawker = get_status(customers)
                customers, seats, leaving_customers = update_hawker_status(customers,seats,feed_time,t)
                num_queue, num_look_seat, num_eating, num_in_hawker = get_status(customers)
                
                if num_in_hawker<len(customers):
                    customers=add_customer(customers,chop_prob,inflow_rate,num_queue,num_stalls,prepare_food_time,crowdedness,t)
                    
                num_queue, num_look_seat, num_eating, num_in_hawker = get_status(customers)
                customers, seats = assign_seat(customers, seats, num_look_seat, num_queue, feed_time, num_stalls, prepare_food_time)
                
                customer_data += leaving_customers
                
            #record waiting time of each customers, only for those entering the hawker centre from critical time onwards   
            waiting_time=np.array([cust_time[0]+cust_time[1] for cust_time in customer_data if cust_time[2]-cust_time[0]-cust_time[1]-feed_time>critical_time])
            result_data.append(np.average(waiting_time))
            
            #record the number of customers spending a specific time in hawker centre and store it in satisfaction_data array
            criteria=np.array([0,2,4,6,8,10])
            wait_time_distribution=np.array([(waiting_time>=wtime).sum() for wtime in criteria])-np.append(np.delete(np.array([(waiting_time>=wtime).sum() for wtime in criteria]),0),0)
            satisfaction_data.append(wait_time_distribution)
        
        x=np.array(chop_probs)
        y=np.array(result_data)
        print(x.shape, y.shape)
        
        x=x.reshape(-1,repeat)
        y=y.reshape(-1,repeat)

        satisfaction_data=np.array(satisfaction_data)

        #reshape the arrays and obtain split them according to the time spent by each customer in hawker centre
        satisfaction_data=satisfaction_data.reshape(-1,repeat,len(criteria))
        satisfaction_data=np.mean(satisfaction_data,axis=1)

        
        x=np.mean(x,axis=1)
        y=np.mean(y,axis=1)
        
         
        #plot the macro state of hawker centre (average waiting time against choping probability)
        plt.scatter(x,y,s=3)
        plt.title("Average waiting time of customers against choping probability")
        plt.figtext(0.93,0.6,"Seats:{}\nMax_inflow:{}\nEat_time:{}\nStalls:{}\nFood_prep:{}".format(hyperparam[1],hyperparam[2],hyperparam[3],hyperparam[4],hyperparam[5]),fontsize=6,bbox=dict(facecolor='none', edgecolor='black'))
        plt.xlabel('Choping probability')
        plt.ylabel('Average waiting time')
        plt.savefig('Run number {} Waiting time against probability'.format(g_counter),  bbox_inches='tight', dpi=300)
        plt.clf()
        
        
        xx,yy = np.meshgrid(x,[1,2,3,4,5,6])
        #print(y)
        #Axes3D.plot_surface(xx, yy, y)
        
        g_counter+=1

def variable_prob():
    
    
    #initialise all parameters and hyperparameters
    g_counter=0
    repeat=3
    prob_step = 1

    chop_probs=[np.round(x,5) for x in np.arange(0,100+prob_step,prob_step)]
    chop_probs=np.repeat(chop_probs,repeat)
    running_average = 200
    max_custs = 100000
    hyperparameters = [[max_custs,40,4,25,2,1],[max_custs,45,4,25,2,1],[max_custs,50,4,25,2,1],[max_custs,55,4,25,2,1],[max_custs,60,4,25,2,1],[max_custs,70,4,25,2,1]]
    # hyperparameters = [[1500,60,1,25,2,1],[1500,60,2,25,2,1],[1500,60,3,25,2,1],[1500,60,4,25,2,1],[1500,60,5,25,2,1],[1500,60,6,25,2,1]]
    simulation_length = 10000
    critical_time = 100
    
    for hyperparam in hyperparameters:
        _, _ ,inflow_rate, feed_time, num_stalls, prepare_food_time = initialize_parameters(hyperparam)

        result_data=[]
        satisfaction_data=[]

        customers, seats, _, _, _, _ = initialize_parameters(hyperparam)           

        outflow_data = []
        prob_data = []

        for t in range(simulation_length):
            
            num_seat_taken, crowdedness =seat_taken(seats)
            num_queue, num_look_seat, num_eating, num_in_hawker = get_status(customers)
            customers, seats, leaving_customers = update_hawker_status(customers,seats,feed_time,t)
            num_queue, num_look_seat, num_eating, num_in_hawker = get_status(customers)

            chop_prob = math.exp(12*crowdedness-6)/(1 + math.exp(12*crowdedness-6))
            
            
            if num_in_hawker<len(customers):
                customers=add_customer(customers,chop_prob,inflow_rate,num_queue,num_stalls,prepare_food_time,crowdedness,t)
                
            num_queue, num_look_seat, num_eating, num_in_hawker = get_status(customers)
            customers, seats = assign_seat(customers, seats, num_look_seat, num_queue, feed_time, num_stalls, prepare_food_time)
            
            #if t >= critical_time:
            prob_data.append(chop_prob)
            outflow_data.append(len(leaving_customers))
        
        average_outflow = [np.mean(outflow_data[i:i+running_average]) for i in range(len(outflow_data)-running_average)]

        #plot the macro state of hawker centre (average waiting time against choping probability)
        plt.scatter(prob_data[:-running_average],average_outflow,s=3)
        plt.title("Choping Probability against outflow rate")
        plt.figtext(0.93,0.6,"Seats:{}\nMax_inflow:{}\nEat_time:{}\nStalls:{}\nFood_prep:{}".format(hyperparam[1],hyperparam[2],hyperparam[3],hyperparam[4],hyperparam[5]),fontsize=6,bbox=dict(facecolor='none', edgecolor='black'))
        plt.xlabel('Choping probability')
        plt.ylabel('Outflow Rate')
        plt.savefig('Run number {} Choping Probability against outflow rate'.format(g_counter),  bbox_inches='tight', dpi=300)
        plt.clf()

        print(1)

        # x = np.array(prob_data[:-running_average])
        # y = np.array(average_outflow)
        # timestep = np.array(range(simulation_length))

        # # Create a set of line segments so that we can color them individually
        # # This creates the points as a N x 1 x 2 array so that we can stack points
        # # together easily to get the segments. The segments array for line collection
        # # needs to be (numlines) x (points per line) x 2 (for x and y)
        # points = np.array([x, y]).T.reshape(-1, 1, 2)
        # segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)

        # # Create a continuous norm to map from data points to colors
        # norm = plt.Normalize(timestep.min(), timestep.max())
        # lc = LineCollection(segments, cmap='viridis', norm=norm)
        # # Set the values used for colormapping
        # lc.set_array(timestep)
        # lc.set_linewidth(1)
        # line = axs.add_collection(lc)
        # fig.colorbar(line, ax=axs)

        # axs.set_xlim(x.min(), x.max())
        # axs.set_ylim(y.min(), y.max())
        # plt.show()
        
        g_counter+=1
   
      
variable_prob()    
print(time.time()-start_time)