from constants import *
from player import Player 
from copy import copy, deepcopy
from random import shuffle

#TODO:
"""
* add player ownership aspect of track lines for self.tracks/city_city_connections
* add calculations of longest track
* add calculations (in player.py) of which tickets are completed
  ** also determine if ticket is already completed when searching for new one
"""

class Game(object):
	def __init__(self, pre_existing_players=None, 
		config=DEFAULT_GAME_CONFIG, n_players=None):
		if not pre_existing_players:
			self.players = [Player(id = i) for i in range(n_players)]
		else:
			self.players = pre_existing_players 
		for i, player in enumerate(self.players):
			player.order = i 
			player.game = self 

		self.trains = copy(TRACK_DECK)
		shuffle(self.trains)
		self.discarded_trains = []
		self.face_up_trains = []
		#organize tracks
		self.tracks = deepcopy(TRACKS)
		for track in self.tracks:
			#these will refer to player order numbers when they become
			#occupied
			track['occupied1'] = None
			track['occupied2'] = None 
		self.city_city_connections = deepcopy(CITY_CITY_CONNECTIONS)

		if config['ticket_versions'] == 'base':
			self.tickets = deepcopy(BASE_TICKETS)
		elif config['ticket_versions'] == 'bigcities':
			self.tickets = deepcopy(BIGCITIES_TICKETS)
		else:
			self.tickets = deepcopy(BASE_TICKETS + BIGCITIES_TICKETS)
		shuffle(self.tickets)

		self.longest_track_player = None
		#controls for the end of the game
		self.last_round = False
		self.final_countdown = len(self.players)
		self.end_game = False
		#some things used to check at end of turn
		self.tiles_updated = False

	def run(self):
		self.pass_first_cards()
		self.give_random_cards_to_players()
		self.lay_out_cards()
		while self.final_countdown > 0:
			pass

	def pass_first_cards(self):
		pass

	def give_random_cards_to_players(self, n_cards=4):
		for player in self.players:
			player.trains[self.trains.pop()] += 1
		return 0 

	def shuffle_discarded_trains_into_deck(self):
		self.trains = self.discarded_trains + self.trains 
		self.discarded_trains = []
		shuffle(self.trains)
		return 0 

	def lay_out_cards(self):
		"""
		will fill out 
		"""
		while len(self.face_up_trains) < 5 and len(self.trains) > 0:
			self.face_up_trains.append(self.trains.pop())
		return 0

	def calculate_longest_track(self):
		maxlen = 0
		best_players = []
		for player in players:
			player.has_longest_track = False
			track_length = player.calculate_longest_track()
			if track_length > maxlen:
				maxlen = track_length
				best_players = [player]
			elif track_length == maxlen:
				best_players.append(player)
		for player in best_players:
			player.has_longest_track = True 
		return maxlen

	#used for saving/loading game states
	def save_state(self):
		pass

	def load_state(self):
		pass
