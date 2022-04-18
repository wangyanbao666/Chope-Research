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
import sys
import copy
import pandas as pd

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
            customers[person][0]=0
    
    ###update seat status    
    for seat in range (0, len(seats)):
        if seats[seat][0]==1: #if seat is occupied, subtract occupied time by one
            seats[seat][1]-=1
        if seats[seat][1]==0: #when counter reach zero, seat is empty
            seats[seat][0]=0
    return customers, seats, leaving_customers

@numba.jit
def add_customer(customers,chop_prob,inflow_rate_prob,num_queue,num_stalls,prepare_food_time,crowdedness,t):
    inflow_rate = rand.randint(0,inflow_rate_prob)
    
    for rate in range (0,inflow_rate): 
            for people in range (0,len(customers)):
                ###randomiser for choping probability
                randomiser=rand.random()*100
                if customers[people][0]==0:
                    customers[people][0]=1;
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

                randomiser=rand.randint(0,len(customers)-1)

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
    # initialise all parameters and hyperparameters
    g_counter = 0
    repeat = 3
    chop_probs = [np.round(x, 5) for x in np.arange(0, 101, 1)]
    chop_probs = np.repeat(chop_probs, repeat)
    num_customer=1000
    capacity=150
    hyperparameters = [[num_customer,capacity, 6, 25, 5, 1], [num_customer,capacity, 7, 25, 5, 1], [num_customer,capacity, 8, 25, 5, 1],
                       [num_customer,capacity, 9, 25, 5, 1], [num_customer,capacity, 10, 25, 5, 1]]
    # hyperparameters=[[num_customer,  capacity, 10, 25, 5, 1]]
    simulation_length = 600
    critical_time = 30
    use_length = simulation_length - critical_time
    for hyperparam in hyperparameters:
        _, _, inflow_rate, feed_time, num_stalls, prepare_food_time = initialize_parameters(hyperparam)
        df = pd.DataFrame()
        tested_prob = []
        result_data = []
        satisfaction_data = []
        outflow_rate = []
        for chop_prob in chop_probs:
            customers, seats, _, _, _, _ = initialize_parameters(hyperparam)
            customer_data = []


            for t in range(simulation_length):

                num_seat_taken, crowdedness = seat_taken(seats)
                num_queue, num_look_seat, num_eating, num_in_hawker = get_status(customers)
                customers, seats, leaving_customers = update_hawker_status(customers, seats, feed_time, t)
                num_queue, num_look_seat, num_eating, num_in_hawker = get_status(customers)

                if num_in_hawker < len(customers):
                    customers = add_customer(customers, chop_prob, inflow_rate, num_queue, num_stalls,
                                             prepare_food_time, crowdedness, t)

                num_queue, num_look_seat, num_eating, num_in_hawker = get_status(customers)
                customers, seats = assign_seat(customers, seats, num_look_seat, num_queue, feed_time, num_stalls,
                                               prepare_food_time)
                num_queue, num_look_seat, num_eating, num_in_hawker = get_status(customers)
                num_seat_taken, crowdedness = seat_taken(seats)
                customer_data += leaving_customers
            crit_leave = [customer for customer in customer_data if customer[2] > critical_time]
            avg_ouflow_rate = len(crit_leave) / use_length
            outflow_rate.append(avg_ouflow_rate)
            # record waiting time of each customers, only for those entering the hawker centre from critical time onwards
            waiting_time = np.array([cust_time[0] + cust_time[1] for cust_time in customer_data if
                                     cust_time[2] - cust_time[0] - cust_time[1] - feed_time > critical_time])
            result_data.append(np.average(waiting_time))

            # record the number of customers spending a specific time in hawker centre and store it in satisfaction_data array
            criteria = np.array([0, 2, 4, 6, 8, 10])
            wait_time_distribution = np.array([(waiting_time >= wtime).sum() for wtime in criteria]) - np.append(
                np.delete(np.array([(waiting_time >= wtime).sum() for wtime in criteria]), 0), 0)
            satisfaction_data.append(wait_time_distribution)

        x = np.array(chop_probs)
        y = np.array(result_data)


        split=101
        yr=np.zeros((101))
        for j in range(repeat):
            yr+=y[split*j:split*(j+1)]

        x = np.array([np.round(x, 5) for x in np.arange(0, 101, 1)])
        y=yr/repeat
        print(x.shape, y.shape)
        # df['chop_prob']=x
        # df['avg_time']=y
        # df.to_csv('waiting time/change inflow_3/Run number {} Waiting time against probability.csv'.format(g_counter))
        # satisfaction_data = np.array(satisfaction_data)

        # plot the macro state of hawker centre (average waiting time against choping probability)
        plt.scatter(x, y, s=3)
        plt.title("Average waiting time of customers against choping probability")
        plt.figtext(0.93, 0.6,
                    "Seats:{}\nMax_inflow:{}\nEat_time:{}\nStalls:{}\nFood_prep:{}".format(hyperparam[1], hyperparam[2],
                                                                                           hyperparam[3], hyperparam[4],
                                                                                           hyperparam[5]), fontsize=10,
                    bbox=dict(facecolor='none', edgecolor='black'))
        plt.xlabel('Choping probability')
        plt.ylabel('Average waiting time')
        plt.savefig('waiting time/change inflow/Run number {} Waiting time against probability'.format(g_counter),
                    bbox_inches='tight', dpi=300)
        plt.clf()

        # plot the macro state of hawker centre (satisfaction level against choping probability --> how many customers spend a certain amount of time in hawker centre)
        # plt.scatter(x,y0,s=3,label='0<=t<{}'.format(criteria[1]))
        # plt.scatter(x,y1,s=3,label='{}<=t<{}'.format(criteria[1],criteria[2]))
        # plt.scatter(x,y2,s=3,label='{}<=t<{}'.format(criteria[2],criteria[3]))
        # plt.scatter(x,y3,s=3,label='{}<=t<{}'.format(criteria[3],criteria[4]))
        # plt.scatter(x,y4,s=3,label='{}<=t<{}'.format(criteria[4],criteria[5]))
        # plt.scatter(x,y5,s=3,label='t>={}'.format(criteria[5]))
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0))
        # plt.title('Satisfaction level against time')
        # plt.xlabel('Choping probability')
        # plt.ylabel('Number of customers')
        # plt.figtext(0.93,0.6,"Seats:{}\nMax_inflow:{}\nEat_time:{}\nStalls:{}\nFood_prep:{}".format(hyperparam[1],hyperparam[2],hyperparam[3],hyperparam[4],hyperparam[5]),fontsize=6,bbox=dict(facecolor='none', edgecolor='black'))
        # plt.savefig('waiting time/Run number {} Satisfaction against time'.format(g_counter), bbox_inches='tight',dpi=300)
        # plt.clf()
        #
        g_counter += 1
main()    
print(time.time()-start_time)