import csv
from collections import defaultdict
#filename sources
TRACK_DATA_FILENAME = 'tracks.csv'
BASE_TICKETS_FILENAME = 'tickets.csv'
BIGCITIES_TICKETS_FILENAME = 'tickets_bigcities.csv'

#main data
CITIES = ['vancouver',
		  'calgary',
		  'winnipeg',
		  'sault st. marie',
		  'montreal',
		  'seattle',
		  'helena',
		  'duluth',
		  'toronto',
		  'boston',
		  'portland',
		  'salt lake city',
		  'omaha',
		  'chicago',
		  'pittsburgh',
		  'new york',
		  'san francisco',
		  'los angeles',
		  'las vegas',
		  'phoenix',
		  'el paso',
		  'santa fe',
		  'denver',
		  'kansas city',
		  'oklaholma city',
		  'dallas',
		  'houston',
		  'new orleans',
		  'little rock',
		  'nashville',
		  'atlanta',
		  'miami',
		  'charleston',
		  'releigh',
		  'washington',
		  'saint louis']

CITIES.sort()

#each city contains a list of other cities that it is connected to
CITY_CITY_CONNECTIONS = defaultdict(dict)
#the full representation of each edge/track
TRACKS = []

with open(TRACK_DATA_FILENAME, 'r') as f:
	reader = csv.reader(f)
	next(reader)
	for i, row in enumerate(reader):
		city1 = row[0]
		city2 = row[1]
		color1 = row[2]
		color2 = row[3]
		length = row[4]
		double = len(color2) > 0
		TRACKS.append({'cities':frozenset([city1,city2]),
			'color1':color1,'color2':color2,
			'length':length,
			'double':double,
			'index':i})
		CITY_CITY_CONNECTIONS[city1].update({city2:[i, length]})
		CITY_CITY_CONNECTIONS[city2].update({city1:[i, length]})

#print CITY_CITY_CONNECTIONS
#print len(CITIES)
#print len(CITY_CITY_CONNECTIONS)
#print len(TRACKS)

TRACK_DECK_LIST = {'green':12,
              'red':12,
              'yellow':12,
              'black':12,
              'white':12,
              'orange':12,
              'pink':12,
              'blue':12,
              'locomotive':14}

TRACK_DECK = reduce(list.__add__, [[color]*n for color, n in TRACK_DECK_LIST.iteritems()])

DECK_COLORS = TRACK_DECK_LIST.keys()
DECK_COLORS.sort()

BASE_TICKETS = []
BIGCITIES_TICKETS = []

with open(BASE_TICKETS_FILENAME, 'r') as f:
	reader = csv.reader(f)
	next(reader)
	for i, row in enumerate(reader):
		city1 = row[0]
		city2 = row[1]
		points = int(row[2])
		BASE_TICKETS.append({'cities':frozenset([city1, city2]),
			                 'points':points,
			                 'index':i})
	MAX_BASE_INDEX = i

with open(BIGCITIES_TICKETS_FILENAME, 'r') as f:
	reader = csv.reader(f)
	next(reader)
	for i, row in enumerate(reader):
		city1 = row[0]
		city2 = row[1]
		points = int(row[2])
		BIGCITIES_TICKETS.append({'cities':frozenset([city1, city2]),
			                 'points':points,
			                 'index':i + 1 + MAX_BASE_INDEX})

#print BASE_TICKETS
#print BIGCITIES_TICKETS

"""
ticket_versions - 'base'/'big_cities'/'both' - what versions of ticket cards are being used?
card_memory - 'none'/[integer]/'infinite' - how well does the ai keep track of other players cards?
discount - [0., 1.] - how much weight is put on victory (1) vs. short-term points?
temperature - [0., 1. (or infinity)] - float representing how random decisions should be based on 
excess_ticket_limit - [integer] - What is the largest number of unfinished tickets a player can have?
                      this is very important, as it limits the architecture of the network under 
                      certain circumstances
big_ticket_limit - [integer]/None - If an integer, players will initially be given a random set of 
                   tickets that includes a ticket of at least this value; tickets of this value or 
                   greater will not be drawable during the game
"""
DEFAULT_GAME_CONFIG = {'ticket_versions':'base',
                       'card_memory':'infinite',
                       'discount':0.8,
                       'temperature':1.,
                       'excess_ticket_limit':7,
                       'big_ticket_limit':None}