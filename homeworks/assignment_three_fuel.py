# Assignment 3, Part 3: Pull data about federal jobs
#
# Version 0.1


from urllib2 import urlopen, Request
import json
import pandas as pd


def get_data(url):
    """ Get response in JSON from URL and convert it into Python dict. """

    # use urlopen() to get a response from the server
    response = urlopen(Request(url))

    # get JSON text from server response
    raw_data = response.read().decode('utf-8')

    # convert JSON text into Python object
    data = json.loads(raw_data)

    return data


# the URL for the Public Jobs API
# API specs at:
# https://developer.nrel.gov/docs/transportation/alt-fuel-stations-v1/nearest/
api_key = "8elBa6cla2QydZGC0rVk9O1mP6l4TtmhH5Imv3Bv"
url = 'https://api.data.gov/nrel/alt-fuel-stations/v1/nearest.json?api_key=%s&location=Ann%%20Arbor+MI' % api_key

# get data from the given URL
print 'Getting initial page of results...'
data = get_data(url)

# one of the keys in the Python dict will have the actual
# jobs data
job_list = data['total_results']

# create a DataFrame from jobs data
df = pd.DataFrame(data['fuel_stations'])

n, d = df.shape
print 'Fetched %d fuel stations. Have %d attributes for each.' % (n, d)

print df['station_name']
