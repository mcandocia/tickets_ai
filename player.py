from constants import *
from collections import defaultdict
from ai import AI 

"""
unsupported strategies:
* revealing information about your hand to enhance collusion with other players against a third
  (less necessary if extra info is known about hand composition)
"""

class Player(object):
	def __init__(self, pre_existing_ai=None, id=None):
		self.id = id
		if not pre_existing_ai:
			self.ai = AI()
		else:
			self.ai = pre_existing_ai
		self.trains = defaultdict(dict)
		self.uncompleted_tickets = []
		self.completed_tickets = []
		#set of frozensets
		#each frozenset is a pair of cities
		self.tracks = set()
		#supply of cars
		self.n_cars = 45
		#needs to be defined by game
		self.order = -1
		#points
		self.positive_ticket_points = 0
		self.negative_ticket_points = 0
		self.track_points = 0
		self.total_points = 0
		#not sure if I want this calculated constantly...
		self.has_longest_track = False
		self.longest_track_length = 0

	#these are actions that change the state of the player
	def select_tickets(self, ticket_list, reject_tickets = [], max_top_of_deck_range=4):
		"""
		has cities, points, and index
		"""
		for ticket in ticket_list:
			if self.finished_ticket(ticket):
				self.completed_tickets.append(ticket)
				self.positive_ticket_points += ticket['points']
			else:
				self.uncompleted_tickets.append(ticket)
				self.negative_ticket_points += ticket['points']
			#remove ticket from game's ticket deck
			for top_ticket_index in range(max_top_of_deck_range):
				top_ticket = self.game.tickets[top_ticket_index]
				if top_ticket['index'] == ticket['index']:
					self.game.tickets.pop(top_ticket_index)
					break
		#sends unused tickets to the back of the deck
		for ticket in reject_tickets:
			for top_ticket_index in range(max_top_of_deck_range):
				top_ticket = self.game.tickets[top_ticket_index]
				if top_ticket['index'] == ticket['index']:
					self.game.tickets.append(self.game.tickets.pop(top_ticket_index))
		return 0 

	def finished_ticket(self, ticket, mark_ticket=False):
		"""
		checks to see if a ticket is finished by the player
		if mark_ticket=True, this will move the player's ticket 
		from unfinished to finished when appropriate
		"""
		ticket_complete = check_tracks_for_completion(self.tracks, ticket)
		if mark_ticket and ticket_complete:
			for i in range(len(self.tickets)):
				owned_ticket = self.uncompleted_tickets[i]
				if owned_ticket['index']==ticket['index']:
					self.completed_tickets.append(self.uncompleted_tickets.pop(i))
					self.negative_ticket_points -= ticket['points']
					self.positive_ticket_points += ticket['points']

	def calculate_longest_track(self):
		self.longest_track_length = calculate_longest_track(self.tracks)
		return self.longest_track_length

	#this one has a second phase to decide what to do if 
	#the first decision isn't to take a rainbow (or only one ticket is left)
	def select_trains(self):
		pass

	def build_tracks(self, segment):
		pass

	def calculate_train_cost(self, segment):
		"""
		determines possible costs of segment, taking rainbow cars into account
		shows options for gray/double tracks
		also checks if any particular color is occupied
		"""
		possible_costs1 = []
		possible_costs2 = []
		length = segment['length']
		color1 = segment['color1']
		color2 = segment['color2']
		occupied1 = segment['occupied1']
		occupied2 = segment['occupied2']
		#check to see if player already as trains on track
		if self.order in [occupied1, occupied2]:
			return [ [], [] ]
		#check to see which ones are already occupied
		if occupied1 <> None:
			pass

		if occupied2 <> None:
			pass

	#take turn
	def take_turn(self):
		self.update_points()

	##check if possible
	def can_build(self):
		pass

	def can_select_trains(self):
		pass

	def can_select_tickets(self):
		pass

	##look at probabilities
	def calc_build_choices(self):
		pass

	#this is done before action is made
	def calc_should_select_ticket_choices(self):
		pass

	#this is done after action is made
	def calc_select_ticket_choices(self, tickets, initial_selection=False):
		pass

	def calc_select_train_choices(self):
		pass

	#used for loading/saving player states
	def save_state(self):
		pass

	def load_state(self):
		pass

	def update_points(self, check_longest_track=False):
		self.points = self.track_points + self.positive_ticket_points - self.negative_ticket_points
		if check_longest_track and self.has_longest_track:
			self.points += 10
		return 0 

#a couple of recursive path-navigating algorithms
#the first isn't efficient, but I don't think it's a huge issue for the given map sizes
def check_tracks_for_completion(tracks, ticket, 
	unexplored_paths='default', starting_city=None):
	#checks if city is destination
	if starting_city <> None:
		if starting_city==ticket[1]:
			return True
	#default behavior for first call
	if unexplored_paths == 'default':
		unexplored_paths = deepcopy(tracks)
	if isinstance(ticket, dict):
		ticket = list(ticket['cities'])
	if not starting_city:
		starting_city = ticket[0]
	#general part of algorithm
	#check all cities that have connections to starting city
	connecting_cities = CITY_CITY_CONNECTIONS[starting_city]
	#for each of these, check which ones exist in unexplored_paths
	#then, temporarily pop off this track and then 
	valid_connections = [city for city in connecting_cities if frozenset([starting_city, city]) in unexplored_paths]
	for city in valid_connections:
		elem_to_remove = frozenset([city, starting_city])
		unexplored_paths.remove(elem_to_remove)
		if check_tracks_for_completion(tracks, ticket, 
			unexplored_paths=unexplored_paths,
			starting_city=city):
			return True
		unexplored_paths.add(elem_to_remove)
	return False 

def calculate_longest_track(tracks,  
	unexplored_paths='default', starting_city=None):
	if unexplored_paths == 'default':
		unexplored_paths = deepcopy(tracks)
	if starting_city==None:
		#iterate over all cities
		#check which cities are valid, then make recursive call for each
		valid_starting_cities = set()
		#if no paths exist, longest path=0
		if len(tracks)==0:
			return 0
		for track in tracks:
			for city in track:
				valid_starting_cities.add(city)
		#iterates over all cities and returns the maximum length of a path that begins
		#at one of those cities
		path_lengths = []
		for city in valid_starting_cities:
			path_lengths.append(calculate_longest_track(tracks,
				unexplored_paths, starting_city=city))
		return max(path_lengths)
	else:
		#iterate over connecting cities
		#general part of algorithm
		#check all cities that have connections to starting city
		connecting_cities = CITY_CITY_CONNECTIONS[starting_city].keys()
		#for each of these, check which ones exist in unexplored_paths
		#then, temporarily pop off this track and then 
		valid_connections = [city for city in connecting_cities if frozenset([starting_city, city]) in unexplored_paths]
		#if no valid connections, then terminates path
		if len(valid_connections)==0:
			return 0
		#for each
		connection_lengths = [None for _ in valid_connections]
		for i, city in enumerate(valid_connections):
			#removes and re-adds the path element from the set being passed so that it 
			#doesn't spend too much time using copy/deepcopy
			elem_to_remove = frozenset([city, starting_city])
			unexplored_paths.remove(elem_to_remove)
			max_sublength = calculate_longest_track(tracks, unexplored_paths, starting_city=city)
			#second index of the CITY_CITY_CONNECTIONS object is the length (first is a numeric ID for the card)
			connection_lengths[i] = max_sublength + CITY_CITY_CONNECTIONS[starting_city][city][1]
			unexplored_paths.add(elem_to_remove)
		return max(connection_lengths)
	
	
