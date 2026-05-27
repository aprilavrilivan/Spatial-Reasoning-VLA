'''
This module defines a custom FSM class for Group 15 Spatial VLM project.
'''

import re

class SpatialVLMFSM:
    def __init__(self):
        self.states = ['INIT', 'DRIVETONEARESTBENCH', 'PICKUP', 'DRIVETONEARESTSTOP',
                  'VIEWANIMALS', 'END']
        self.current_state = 'INIT' # initial state
        self.observations = {
            'CountPeople': None,
            'CountPeopleRaw': None,
            'ClosestToFurthestBenches': [],
            'ClosestBench': None,
            'ClosestBenchWithPerson': -1,
            'ArrivedAtBench': None,
            'ArrivedAtAnimalsAroundStopSigns': None,
            'CountPersonAtClosestBench': None,
            'CountPersonAtClosestBenchRaw': None,
            'ClosestStopSign': None,
            'ClosestToFurthestStopSigns': [],
            'all_zoos_visited': False,
            'waiting_time_exceeded': None,
        }
        # in state transitions, include guard conditions as needed
        self.state_transitions = {
            'INIT': ('CountPeople', {True: 'DRIVETONEARESTBENCH', False: 'DRIVETONEARESTSTOP'}),
            'DRIVETONEARESTBENCH': ('ArrivedAtBench', {True: 'PICKUP', False: 'DRIVETONEARESTBENCH'}),
            'PICKUP': ('CountPersonAtClosestBench', {True: 'PICKUP', False: 'INIT'}),
            'DRIVETONEARESTSTOP': ('ArrivedAtAnimalsAroundStopSigns', {True: 'VIEWANIMALS', False: 'DRIVETONEARESTSTOP'}),
            'VIEWANIMALS': (('waiting_time_exceeded', 'all_zoos_visited'), 
                            {(True, True): 'END', (False, True): 'VIEWANIMALS', 
                             (False, False): 'VIEWANIMALS', (True, False): 'DRIVETONEARESTSTOP'}),
        }

        # self.question_dict = {
        #     'people_waiting': 'Are there one or more people in the image? Respond with \'Yes\' or \'No\'.',

        #     'occupied_benches': 'Each bench in the image has a visible number label beside it (e.g., 1, 2, 3, ...). '
        #                         'Use these printed numbers as the bench IDs. List the benches in order from closest to '
        #                         'furthest from the clock, separated by commas. For example, \'2, 1, 4, 3\'.',

        #     'target_bench': 'Each bench in the image has a visible number label beside it (e.g., 1, 2, 3, ...). '
        #                     'Use these printed numbers as the bench IDs. Which bench is closest to the clock that '
        #                     'has at least one person at it? Answer the bench ID. If no benches have people, respond with \'0\'.', 

        #     'at_bench': 'Each bench in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these '
        #                 f'printed numbers as the bench IDs.  Is the clock close to bench number {self.observations["target_bench"]}? '
        #                 'Respond with \'Yes\' or \'No\'.',

        #     'at_stop': 'Each stop sign in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these '
        #                'printed numbers as the stop sign IDs. For each stop sign, consider all animals that are spatially '
        #                'closest to that stop sign. Is the clock close to the animals around stop sign number '
        #                f'{self.observations["target_zoo"]}? Respond with \'Yes\' or \'No\'.',

        #     'people_waiting_current_bench': 'Are there people at the bench closest to the clock? Respond with \'Yes\' or \'No\'.',

        #     'target_zoo': '', # shouldn't be asked directly (we keep track of it here)

        #     'zoos_to_visit': 'Each stop sign in the image has a visible number label beside it (e.g., 1, 2, 3, ...). '
        #                      'Use these printed numbers as the stop sign IDs. List the stop signs in order from closest to '
        #                      'furthest from the clock, separated by commas. For example, \'2, 1, 4, 3\'.', 

        #     'all_zoos_visited': '', # shouldn't be asked directly (we keep track of it here)

        #     'waiting_time_exceeded': '', # shouldn't be asked directly (we keep track of it here)
            
        # }
        self.question_dict = {
            'CountPeople': "How many people are currently visible in this scene? Respond with only an integer.",
            'CountAnimals': "How many animals are currently visible in this scene? Respond with only an integer.",
            'CountPeopleAtBench': (
                "Each bench in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the bench IDs. "
                "How many people are currently at bench #{bench_number}? Respond with only an integer."
            ),
            'CountAnimalsAtStopSign': (
                "Each stop sign in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the stop sign IDs. "
                "How many animals are currently around stop sign #{stop_sign_number}? Respond with only an integer."
            ),
            'ListBenchesWithAtLeastKPeople': (
                "Each bench in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the bench IDs. "
                "List the IDs of all benches that have at least {k} people in ascending order, separated by commas. "
                "If none, respond with '0'."
            ),
            'ListStopSignsWithAtLeastKAnimals': (
                "Each stop sign in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the stop sign IDs. "
                "List the IDs of all stop signs that have at least {k} animals around them in ascending order, separated by commas. "
                "If none, respond with '0'."
            ),
            'ArrivedAtBench': (
                "Each bench in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the bench IDs. "
                "Is the clock close enough to be considered arrived at bench number {bench_number}? Respond with 'Yes' or 'No'."
            ),
            'ArrivedAtAnimalsAroundStopSigns': (
                "Each stop sign in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the stop sign IDs. "
                "For each stop sign, consider all animals that are spatially closest to that stop sign. "
                "Is the clock close enough to be considered arrived at at least one of the animals around stop sign number {stop_sign_number}? Respond with 'Yes' or 'No'."
            ),
            'ClosestBench': (
                "Each bench in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the bench IDs. "
                "Which bench is closest to the clock? "
                "Answer with the bench ID."
            ),
            'ClosestStopSign': (
                "Each stop sign in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the stop sign IDs. "
                "Which stop sign is closest to the clock? "
                "Answer with its ID."
            ),
            'PairwiseCloserBench': (
                "Each bench in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the bench IDs. "
                "Which is closer to the clock, bench #{bench_i} or bench #{bench_j}? Respond with only the bench number."
            ),
            'PairwiseCloserStopSign': (
                "Each stop sign in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the stop sign IDs. "
                "Which is closer to the clock, stop sign #{stop_i} or stop sign #{stop_j}? Respond with only the stop sign number."
            ),
            'ClosestToFurthestBenches': (
                "Each bench in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the bench IDs. "
                "List the benches in order from closest to furthest from the clock, separated by commas. "
                "For example, '2, 1, 4, 3'. "
            ),
            'ClosestToFurthestStopSigns': (
                "Each stop sign in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the stop sign IDs. "
                "List the stop signs in order from closest to furthest from the clock, separated by commas. "
                "For example, '2, 1, 4, 3'. "
            ),
            'GeometricDirectionToBench': (
                "Each bench in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the bench IDs. "
                "What is the relative direction of bench #{bench_number} to the clock? "
                "Answer with exactly one of: 'North', 'South', 'East', 'West', "
                "'Northeast', 'Northwest', 'Southeast', or 'Southwest'. "
            ),
            'GeometricDirectionToStopSign': (
                "Each stop sign in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the stop sign IDs. "
                "What is the relative direction of stop sign #{stop_sign_number} to the clock? "
                "Answer with exactly one of: 'North', 'South', 'East', 'West', "
                "'Northeast', 'Northwest', 'Southeast', or 'Southwest'. "
            ),
            'AvoidObstacleToReachBench': (
                "Each bench in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the bench IDs. "
                "A red circle is placed in front of the clock to indicate its current heading direction. "
                "The clock is currently facing bench {bench_number} and wants to reach that bench. "
                "Ignore the people already at bench {bench_number}. "
                "If no other object blocks the straight path between the clock and bench {bench_number}, answer 'keep straight'. "
                "Otherwise, answer 'turn left' or 'turn right' to avoid the first blocking object along that path."
            ),
            'AvoidObstacleToReachStopSign': (
                "Each stop sign in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the stop sign IDs. "
                "A red circle is placed in front of the clock to indicate its current heading direction. "
                "The clock is currently facing stop sign {stop_sign_number} and wants to reach that stop sign. "
                "Ignore the animals already grouped with stop sign {stop_sign_number}. "
                "If no other object blocks the straight path between the clock and stop sign {stop_sign_number}, answer 'keep straight'. "
                "Otherwise, answer 'turn left' or 'turn right' to avoid the first blocking object along that path."
            ),
            'BusHeadingDirection': (
                "A red circle is placed in front of the clock in the image to indicate its current heading direction. "
                "Based on the position of the red circle relative to the clock, in which direction is the clock currently heading? "
                "Answer with exactly one of: 'North', 'South', 'East', 'West', "
                "'Northeast', 'Northwest', 'Southeast', or 'Southwest'."
            ),
            'TurnDirectionToBench': (
                "Each bench in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the bench IDs. "
                "A red circle is placed in front of the clock to indicate its current heading direction. "
                "To face bench #{bench_number}, should the clock turn left, turn right, or is it already facing that bench? "
                "Answer with exactly one of: 'turn left', 'turn right', or 'already facing'."
            ),
            'TurnDirectionToStopSign': (
                "Each stop sign in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the stop sign IDs. "
                "A red circle is placed in front of the clock to indicate its current heading direction. "
                "To face stop sign #{stop_sign_number}, should the clock turn left, turn right, or is it already facing that stop sign? "
                "Answer with exactly one of: 'turn left', 'turn right', or 'already facing'."
            ),
            'BenchRelativeToHeading': (
                "Each bench in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the bench IDs. "
                "A red circle is placed in front of the clock to indicate its current heading direction. "
                "Where is bench #{bench_number} relative to the clock's current heading direction? "
                "Answer with exactly one of: 'front', 'front-right', 'right', 'back-right', 'back', 'back-left', 'left', or 'front-left'."
            ),
            'StopSignRelativeToHeading': (
                "Each stop sign in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the stop sign IDs. "
                "A red circle is placed in front of the clock to indicate its current heading direction. "
                "Where is stop sign #{stop_sign_number} relative to the clock's current heading direction? "
                "Answer with exactly one of: 'front', 'front-right', 'right', 'back-right', 'back', 'back-left', 'left', or 'front-left'."
            ),
            'CountPersonAtClosestBench': "How many people are at the bench closest to the clock? Respond with only an integer.",
            'ClosestBenchWithPerson': (
                "Each bench in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these printed numbers as the bench IDs. "
                "Which bench is closest to the clock that has at least one person at it? Answer with the bench ID. "
                "If no benches have people, respond with '0'. "
            ),
            'AvoidObstacleToReachClosestBench': (
                "A red circle is placed in front of the clock to indicate its current heading direction. "
                "The clock is currently facing the closest bench and wants to reach that bench. "
                "Ignore the people already at that bench. "
                "If no other object blocks the straight path between the clock and the closest bench, answer 'keep straight'. "
                "Otherwise, answer 'turn left' or 'turn right' to avoid the first blocking object along that path."
            ),
            'AvoidObstacleToReachClosestStopSign': (
                "A red circle is placed in front of the clock to indicate its current heading direction. "
                "The clock is currently facing the closest stop sign and wants to reach that stop sign. "
                "Ignore the animals already grouped with that stop sign. "
                "If no other object blocks the straight path between the clock and the closest stop sign, answer 'keep straight'. "
                "Otherwise, answer 'turn left' or 'turn right' to avoid the first blocking object along that path."
            ),
            'DirectionToClosestBench': (
                "What is the relative direction of the closest bench to the clock? "
                "Answer with exactly one of: 'North', 'South', 'East', 'West', "
                "'Northeast', 'Northwest', 'Southeast', or 'Southwest'."
            ),
            'DirectionToClosestStopSign': (
                "What is the relative direction of the closest stop sign to the clock? "
                "Answer with exactly one of: 'North', 'South', 'East', 'West', "
                "'Northeast', 'Northwest', 'Southeast', or 'Southwest'."
            ),
        }
    
    def override_states(self, new_state):
        self.current_state = new_state
        

    # update observations based on VLM responses
    # takes a dict of observation_key: VLM_response.
    # Doesn't have to include all observations, only those that were queried
    def update_observations(self, vlm_observations):
        for key, value in vlm_observations.items():
            # parse VLM response based on expected type
            value = self.parse_VLM_response(key, value)

            if key in self.observations:
                self.observations[key] = value
        #I dont think we need this anymore, since we have specific questions for these.
        # update observation dependent questions
        # self.question_dict['ArrivedAtBench'] = ('Each bench in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these '
        #                 f'printed numbers as the bench IDs.  Is the clock close to bench number {self.observations["target_bench"]}? '
        #                 'Respond with \'Yes\' or \'No\'.')
        # self.question_dict['ArrivedAtAnimalsAroundStopSigns'] = ('Each stop sign in the image has a visible number label beside it (e.g., 1, 2, 3, ...). Use these '
        #                'printed numbers as the stop sign IDs. For each stop sign, consider all animals that are spatially '
        #                'closest to that stop sign. Is the clock close to the animals around stop sign number '
        #                f'{self.observations["ClosestStopSign"]}? Respond with \'Yes\' or \'No\'.')

        # manually set zoos to visit
        #self.observations['ClosestToFurthestBenches'] = [1] # I dont understand this to be honest

        # after updating observations, check for possible state transition
        self._do_transition()

    # internal method to check/apply transitions based on current observations
    def _do_transition(self):
        prev_state = self.current_state
        transition = self.state_transitions.get(self.current_state)
        if transition:
            obs_key, conditions = transition
            if isinstance(obs_key, tuple):
                obs_values = tuple(self.observations[key] for key in obs_key)
                new_state = conditions.get(obs_values)
            else:
                obs_value = self.observations[obs_key]
                new_state = conditions.get(obs_value)

            if new_state and new_state in self.states:
                self.current_state = new_state

        # we can do some manual observation updates
        # if prev_state == 'DRIVETONEARESTBENCH' and self.current_state == 'PICKUP':
            # self.observations['occupied_benches'].pop(0)  # remove the bench we're at
            # if len(self.observations['occupied_benches']) == 0:
            #     self.observations['people_waiting'] = False

        if prev_state == 'DRIVETONEARESTSTOP' and self.current_state == 'VIEWANIMALS':
            if self.observations['ClosestToFurthestStopSigns']:
                self.observations['ClosestToFurthestStopSigns'].pop(0)  # remove the zoo we're at
            if len(self.observations['ClosestToFurthestStopSigns']) == 0:
                self.observations['all_zoos_visited'] = True

        # shouldn't have to reset at_bench/at_stop since the target will be updated and requeried from VLM, but just in case.
        elif prev_state == 'INIT' and self.current_state == 'DRIVETONEARESTBENCH':
            # reset at_bench observation
            self.observations['ArrivedAtBench'] = False
        elif prev_state in ('INIT', 'VIEWANIMALS') and self.current_state == 'DRIVETONEARESTSTOP':
            # reset at_stop and waiting_time_exceeded observations
            self.observations['ArrivedAtAnimalsAroundStopSigns'] = False
            self.observations['waiting_time_exceeded'] = False
            
        # some observations can be derived from others
        self.observations['ClosestBench'] = self.observations['ClosestToFurthestBenches'][0] if self.observations['ClosestToFurthestBenches'] else None
        self.observations['ClosestStopSign'] = self.observations['ClosestToFurthestStopSigns'][0] if self.observations['ClosestToFurthestStopSigns'] else None
        self.observations['all_zoos_visited'] = len(self.observations['ClosestToFurthestStopSigns']) == 0

    # convert VLM response to appropriate type/format
    def parse_VLM_response(self, obs_key, response):
        response = response.strip().lower()
        if obs_key in ['ArrivedAtBench', 'ArrivedAtAnimalsAroundStopSigns', 'waiting_time_exceeded', 'all_zoos_visited']:
            if response in ['yes', 'true', '1']:
                return True
            elif response in ['no', 'false', '0']:
                return False
        elif obs_key == 'CountPeople':
            try:
                count = int(re.search(r"-?\d+", response).group(0))
                self.observations['CountPeopleRaw'] = count
                return count > 0
            except:
                return False
        elif obs_key == 'CountPersonAtClosestBench':
            try:
                count = int(re.search(r"-?\d+", response).group(0))
                self.observations['CountPersonAtClosestBenchRaw'] = count
                return count > 0
            except:
                return False
        elif obs_key in ['ClosestToFurthestBenches', 'ClosestToFurthestStopSigns']:
            try:
                return [int(item) for item in re.findall(r"-?\d+", response)]
            except:
                return []
        elif obs_key in ['ClosestBench', 'ClosestStopSign', 'ClosestBenchWithPerson', 'CountPeopleAtBench', 'CountAnimalsAtStopSign']:
            try:
                return int(re.search(r"-?\d+", response).group(0))
            except:
                return -1
        return response

    def get_current_state(self):
        return self.current_state
    
    # get relevant observation keys for next transition
    def get_relevant_observation_keys(self):
        transition = self.state_transitions.get(self.current_state)
        if transition:
            obs_key, _ = transition
            if isinstance(obs_key, tuple):
                return list(obs_key)
            else:
                keys = [obs_key]
                if self.current_state == 'INIT':
                    keys.append('ClosestBenchWithPerson')
                    keys.append('ClosestToFurthestStopSigns')
                return keys
        return []
    
    # get relevant questions for next transition
    def get_relevant_questions(self, keys=None):
        if keys==None:
            keys = self.get_relevant_observation_keys()

        if 'all_zoos_visited' in keys:
            keys.remove('all_zoos_visited') # don't need to query VLM for this
        if 'waiting_time_exceeded' in keys:
            keys.remove('waiting_time_exceeded')
        

        return {key: self.question_dict[key] for key in keys if key in self.question_dict}
    
    def get_init_keys(self):
        return ['CountPeople', 'ClosestBenchWithPerson', 'ClosestToFurthestStopSigns']
    
    # get target bench or zoo, return tuple (type, id)
    def get_target(self):
        if self.current_state in ['DRIVETONEARESTBENCH', 'PICKUP']:
            target = self.observations.get('ClosestBenchWithPerson')
            if not target or target < 0:
                target = self.observations.get('ClosestBench')
            return ('bench', target)
        elif self.current_state in ['DRIVETONEARESTSTOP', 'VIEWANIMALS']:
            return ('stop sign', self.observations['ClosestStopSign'])
        return (None, None)
