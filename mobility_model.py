import pandas as pd
import numpy as np
location_probabilities = pd.read_csv("data/mobility_model_data.csv")
location_types = location_probabilities.iloc[:,2:].columns.values
num_time_groups = location_probabilities["Time_of_day"].unique().shape[0]

location_probabilities = np.array(location_probabilities.iloc[:,2:])
location_probabilities=np.hstack((np.zeros((location_probabilities.shape[0],1)),np.cumsum(location_probabilities,axis=1)))

def getNextLocType(age,time_of_day,r_num):

    # filter according to age
    if age >=5 and age <22:
        i=0
    elif age >=22 and age <30:
        i=1
    elif age >=30 and age <60:
        i=2
    elif age >=60 and age <=90:
        i=3

    # filter according to time of day (the hour of the day)
    #
    time_of_day_hour = time_of_day.hour
    if time_of_day_hour >= 0 and time_of_day_hour<7:
        j=0
    elif time_of_day_hour >= 7 and time_of_day_hour<12:
        j=1
    elif time_of_day_hour >= 12 and time_of_day_hour<17:
        j=2
    elif time_of_day_hour >= 17 and time_of_day_hour<20:
        j=3
    elif time_of_day_hour >= 20 and time_of_day_hour<24:
        j=4

    prob_intervals = location_probabilities[i*num_time_groups + j, :]

    # get the next location type according to the probability vector
    for i in range(len(prob_intervals)-1):
        if r_num >= prob_intervals[i] and r_num < prob_intervals[i+1]:
            return location_types[i]


################## TESTING CODE  ##################

if __name__ == "__main__":
    import datetime
    import time
    sample_date = datetime.datetime(year=2020,month=1,day=1,hour=18)# 01 jan 2020 0000 hrs
    sample_date = sample_date+ datetime.timedelta(hours=1)# 01 jan 2020 0100 hrs
    t1  = time.time()
    for i in range(10000):
        next_location_type = getNextLocType(32, sample_date,np.random.random())
        print(next_location_type)
    t2 = time.time()
    print(t2-t1)
