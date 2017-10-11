# Assignment 3, Part 3: Pull data about federal jobs
#
# Version 0.1


from urllib2 import urlopen, Request
import json
import pandas as pd


# TASK 3.1
# Implement the get_data function
def get_data(url):
    """ Get response in JSON from URL and convert it into Python dict. """

    # TASK 3.1.1
    # use urlopen() to get a response from the server
    response = urlopen(Request(url))

    # TASK 3.1.2
    # get JSON text from server response
    raw_data = response.read().decode('utf-8')

    # TASK 3.1.3
    # convert JSON text into Python object
    data = json.loads(raw_data)

    return data


# TASK 3.5
# Implement the numeric_value function
def numeric_value(string):
    """ Get rid of dollar sign & commas and convert to float. """

    return float(string.replace('$', '').replace(',', ''))

# the URL for the Public Jobs API
# API specs at:
# https://developer.nrel.gov/docs/transportation/alt-fuel-stations-v1/nearest/
api_key = "8elBa6cla2QydZGC0rVk9O1mP6l4TtmhH5Imv3Bv"
url = 'https://api.data.gov/nrel/alt-fuel-stations/v1/nearest.json?api_key=%s&location=Ann%%20Arbor+MI' % api_key

# get data from the given URL
print 'Getting initial page of results...'
data = get_data(url)

# TASK 3.2.1
# one of the keys in the Python dict will have the actual
# jobs data
job_list = data['total_results']


# TASK 3.4.1
# create a DataFrame from jobs data
df = pd.DataFrame(data['fuel_stations'])

n, d = df.shape
print 'Fetched %d fuel stations. Have %d attributes for each.' % (n, d)

print df['station_name']


## TASK 3.4.2
## get the unique agencies that have jobs
#agencies = df['AgencySubElement'].unique()
#
## for each agency report number of jobs and average min salary
#for i in range(len(agencies)):
#
#    # boolean pandas Series to select current agency
#    agency_jobs = (df['AgencySubElement'] == agencies[i])
#
#    # number of jobs in this agency
#    num_jobs = agency_jobs.sum()
#
#    # average of the minimum salary for jobs in this agency
#    mean_salarymin = (df['SalaryMin'][agency_jobs]).map(numeric_value).mean()
#
#    # print current agency's no. of jobs and average min salary
#    if num_jobs == 1:
#        print '%s has %d job with minimum salary %.2f' \
#            % (agencies[i], num_jobs, mean_salarymin)
#    else:
#        print '%s has %d jobs with average minimum salary %.2f' \
#            % (agencies[i], num_jobs, mean_salarymin)

