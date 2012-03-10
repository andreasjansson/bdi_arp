# bdi_arp
# Copyright (C) 2012   <andreas@jansson.me.uk>
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at
# your option) any later version.
# 
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# 

# This is a multi-agent algorithmic composition system, informally based
# on the Belief Desire Intention architecture.
#
# The system is made up of a number of agents, all playing evolving
# arpeggios. The agents listen to eachother and try to sound good
# as a group, whilst constantly trying to push the piece forward.
# 
# Agents have two conflicting desires, a desire for consonance and
# a desire for chord progression. The desire for chord progression
# is gradually increasing, while the desire for consonance is decreasing.
# When a chord change actually comes about, the agent shifts mood
# back to consonance.
#
# On each time step, the agent generates a random set of small changes
# to its arpeggio, and evaluates these with regard to its currently
# dominant desire, the overall chord that the "orchestra" is currently
# playing, and its knowledge music theory (represented in the chord_templates
# and progression_scores variables).
#
# If a consonant arpeggio is chosen, the agent's mood changes slightly
# towards progression. If an arpeggio in a new chord is chosen, the
# agent has a limited amount of time to convince the other agents to
# play this chord. During this time the desire is fixed to be consonant
# with this new chord. If the agent fails to impose this new chord
# on the others, it reverts back to its normal behaviour, with an
# initial desire for consonance.
#
# Run the current arrangement simply by `python agent.py` (or
# `python2 agent.py`). This requires python-portmidi
# (https://github.com/grantma/python-portmidi), and the portmidi
# (available in the repos on most Linux distros). Run pmdefaults
# to assign a MIDI output device.

import numpy as np
import math
import random
import time
import itertools
from pprint import pprint

debugging = True

class Agent:

    uid = 0

    def __init__(self,
                 memory_length,
                 chord_templates,
                 progression_scores,
                 pitch_bounds,
                 length_bounds,
                 allow_pause = False,
                 density = .1,
                 length_factor = 19,
                 chord_change_timeout = 10,
                 max_desire = 20,
                 random_action_count = 10):

        # how many time steps back does the agent remember
        self.memory_length = memory_length

        # a set of profiles for matching heard notes to chords
        self.chord_templates = chord_templates

        # a (chordsXchords) matrix of chord progression likelihoods
        self.progression_scores = progression_scores

        # upper and lower pitch for the agent
        self.pitch_bounds = pitch_bounds

        # upper and lower arpeggio length for the agent
        self.length_bounds = length_bounds

        # does the agent allow pauses
        self.allow_pause = allow_pause

        # if it allows pauses, how likely are these pauses?
        # (pretty nonscientific, arbitrary unit. usually between 0 and 1)
        self.density = density

        # the higher the number, the more biased the agent is towards long
        # arpeggios (within the bounds)
        self.length_factor = length_factor

        # how long will the agent try to convince other agents of a new chord
        self.chord_change_timeout = chord_change_timeout

        # effectively this variable determines how quickly the agent shifts moods
        self.max_desire = max_desire

        # how many random changes will be generated and evaluated on each iteration
        self.random_action_count = random_action_count

        self.consonance_desire = self.max_desire
        self.progression_desire = 0
        self.desired_chord = None
        self.chord_change_ticker = 0
        self.heard_notes = np.zeros((12, self.memory_length))
        self.arpeggio = self.get_initial_arpeggio()
        self.uid = Agent.uid
        Agent.uid += 1

    def hear(self, note):
        """
        Hear another agent's note.
        """
        if note is not None:
            self.heard_notes[note % 12, 0] += 1

    def get_note(self):
        """
        Generate and return a new note.
        """
        profile = self.get_heard_profile()
        chord = self.get_chords_for_profile(profile)[0] # assuming there's enough data to avoid ambiguity

        self.update_arpeggio(profile, chord)
        self.update_chord_change_ticker(chord)
        self.update_desire()

        note = self.arpeggio[0]
        if note == -1:
            return None
        return note

    def get_state_string(self):
        """
        A string describing what the agent currently is
        thinking and doing. Useful for debugging.
        """
        profile = self.get_heard_profile()
        chord = self.get_chords_for_profile(profile)[0] # assuming there's enough data to avoid ambiguity
        return "%s:\t  Ch: %s;\tDesCh: %s;\tChChngT: %d;\tConsDes: %d;\tProgDes: %d;\tArp: %s" % \
            (self.uid, chord_to_string(chord), chord_to_string(self.desired_chord), self.chord_change_ticker,
             self.consonance_desire, self.progression_desire, arp_to_string(self.arpeggio))

    def forget(self):
        """
        Rotate memory.
        """
        self.heard_notes = np.roll(self.heard_notes, 1, axis = 1)
        self.heard_notes[:,0] = 0
        
    def update_arpeggio(self, profile, chord):
        """
        Generate changes to the arpeggio, evaluate them,
        update the arpeggio and rotate it.
        """
        potential_actions = self.generate_random_actions()
        potential_actions = [NoChangeAction()] + potential_actions

        scores_for_actions = np.zeros(len(potential_actions))
        arpeggios = []
        for i, action in enumerate(potential_actions):
            arpeggio = action.execute(np.copy(self.arpeggio))
            arpeggios.append(arpeggio)
            scores_for_actions[i] = self.evaluate_arpeggio(arpeggio, profile, chord)

        best_i = np.where(scores_for_actions == max(scores_for_actions))[0]
        best_i = random.sample(best_i, 1)[0]
        self.arpeggio = arpeggios[best_i]
        self.arpeggio = np.roll(self.arpeggio, -1)

    def update_chord_change_ticker(self, current_chord):
        """
        If the new arpeggio is a chord change, set the chord change countdown.
        Otherwise, decrement this countdown.
        """
        new_chord,_ = self.get_best_progression(current_chord, self.arpeggio)
        arp_profile = self.get_arpeggio_profile(self.arpeggio)
        new_chord_is_better = sum(self.chord_templates[new_chord,:] * arp_profile) > \
            sum(self.chord_templates[current_chord,:] * arp_profile)

        chord_changed = new_chord != current_chord and new_chord != self.desired_chord \
            and new_chord_is_better and self.progression_desire > self.consonance_desire
        if chord_changed:
            self.desired_chord = new_chord
            self.chord_change_ticker = self.chord_change_timeout
        elif self.desired_chord is not None:
            self.chord_change_ticker -= 1
            if self.chord_change_ticker == 0:
                self.desired_chord = None

    def update_desire(self):
        """
        Mood shifts.
        """
        if self.desired_chord is not None:
            self.progression_desire = 0
            self.consonance_desire = self.max_desire
        elif self.consonance_desire > 0: #1: # >1 instead of >0 to retain some randomness
            self.progression_desire += 1
            self.consonance_desire -= 1

    def evaluate_arpeggio(self, arpeggio, profile, chord):
        """
        Return a score for how well an arpeggio fits with
        the current desires and the history of heard notes.
        """
        orig_arpeggio = arpeggio
        if self.allow_pause:
            arpeggio = filter(lambda x: x != -1, arpeggio)
        pause_count = len(orig_arpeggio) - len(arpeggio)

        _, progression_score = self.get_best_progression(chord, arpeggio)

        ps = progression_score * 100 # arbitrary factor TODO: put in constructor
        cs = self.consonance_score(chord, arpeggio)
        pd = self.progression_desire
        cd = self.consonance_desire
        score = (ps * pd + cs * cd) + (len(orig_arpeggio) - pause_count * self.density) * self.length_factor

        return score

    def consonance_score(self, current_chord, arpeggio):
        """
        How well does this chord fit with the current chord
        (or the chord we want everyone to change to).
        """
        arp_profile = self.get_arpeggio_profile(arpeggio)
        chord = current_chord if self.desired_chord is None else self.desired_chord
        chord_profile = self.chord_templates[chord,:]
        return np.sum(chord_profile * arp_profile)

    def get_best_progression(self, current_chord, arpeggio):
        """
        Calculates what is the best chord progression we can make
        given the current chord and a particular arpeggio.
        """
        arp_profile = self.get_arpeggio_profile(self.arpeggio)
        potential_arp_chords = self.get_chords_for_profile(arp_profile)
        scores = self.progression_scores[current_chord][potential_arp_chords]
        best_score = np.max(scores)
        best_index = np.where(scores == best_score)
        best_chord = random.sample(potential_arp_chords[best_index], 1)[0]
        return (best_chord, best_score)

    def generate_random_actions(self):
        """
        An action is a way to change the arpeggio. This function
        returns a bunch of them.
        """
        actions = [AddAction, RemoveAction, ShiftAction]
        if self.allow_pause:
            actions += [SilentAction]
        return [random.choice(actions)(self.pitch_bounds, self.length_bounds)
                for i in range(self.random_action_count)]

    def get_heard_profile(self):
        """
        Returns a chromagram of what has been heard in the past
        little while.
        """
        recency_bias = np.arange(self.memory_length, 0, -1). \
            reshape(1, self.memory_length).repeat(12, 0)
        profile = (self.heard_notes * recency_bias).sum(axis = 1)
        #profile = (self.heard_notes).sum(axis = 1)
        return normalise(profile)

    def get_chords_for_profile(self, profile):
        scores = (profile * self.chord_templates).sum(axis = 1)
        return np.concatenate(np.where(scores == np.max(scores)))

    def get_arpeggio_profile(self, arpeggio):
        if self.allow_pause:
            arpeggio = filter(lambda x: x != -1, arpeggio)

        if len(arpeggio) == 0: # all pauses, return the currently heard profile
            return self.get_heard_profile()

        arpeggio = np.tile(arpeggio, int(math.ceil(16.0 / len(arpeggio))))[0:16] # pad to 16 elements for fairness
        profile = np.bincount(arpeggio % 12, minlength = 12)
        return profile

    def get_initial_arpeggio(self):
        sample_array = range(self.pitch_bounds[0], self.pitch_bounds[1])
        if self.allow_pause:
            sample_array += [-1]
        sample_count = random.randint(self.length_bounds[0], self.length_bounds[1])
        return np.array(random.sample(sample_array, sample_count))


class Action:
    def __init__(self, pitch_bounds = None, length_bounds = None):
        self.pitch_bounds = pitch_bounds
        self.length_bounds = length_bounds

class NoChangeAction(Action):
    def execute(self, arpeggio):
        return arpeggio

class AddAction(Action):
    """
    Add a note to the arpeggio.
    """
    def execute(self, arpeggio):
        if len(arpeggio) == self.length_bounds[1]:
            return arpeggio
        index = np.random.randint(0, len(arpeggio) + 1)
        possible_notes = np.setdiff1d(np.arange(
                self.pitch_bounds[0], self.pitch_bounds[1]), arpeggio)
        note = random.sample(possible_notes, 1)[0]
        return np.insert(arpeggio, index, note)

class RemoveAction(Action):
    """
    Remove a note from the arpeggio.
    """
    def execute(self, arpeggio):
        if len(arpeggio) == self.length_bounds[0]:
            return arpeggio
        index = np.random.randint(0, len(arpeggio))
        return np.delete(arpeggio, index)

class ShiftAction(Action):
    """
    Move a note in the arpeggio.
    """
    def execute(self, arpeggio):
        index = np.random.randint(0, len(arpeggio))
        possible_notes = np.setdiff1d(np.arange(
                self.pitch_bounds[0], self.pitch_bounds[1]), arpeggio)
        note = random.sample(possible_notes, 1)[0]
        arpeggio[index] = note
        return arpeggio

class SilentAction(Action):
    """
    Mute a note in the arpeggio.
    """
    def execute(self, arpeggio):
        index = np.random.randint(0, len(arpeggio))
        arpeggio[index] = -1
        return arpeggio


# This is the actual composition:

def get_my_simple_chord_templates():
    minor = np.array([1.0, -0.5, 0.3, 1.0, -0.5, 0.2, -0.5, 1.0, -0.5, 0.1, 0.5, -0.5])
    seventh = np.array([1.0, -0.5, 0.2, -0.5, 1.0, 0.2, -0.5, 1.0, -0.5, 0.1, 0.6, -0.5])
    major = np.array([1.0, -0.5, 0.2, -0.5, 1.0, 0.3, -0.5, 1.0, -0.5, 0.1, -0.5, 0.5])

    minor_chunk = get_chromatic_chunk(minor)
    seventh_chunk = get_chromatic_chunk(seventh)
    major_chunk = get_chromatic_chunk(major)

    return np.vstack((minor_chunk, seventh_chunk, major_chunk))

def get_chromatic_chunk(profile):
    m = profile
    for i in np.arange(1, 12):
        shifted = np.roll(profile, i)
        m = np.vstack((m, shifted))
    return m

def get_my_progression_scores():
    m = np.zeros((36, 36))
#    m += .01 # for a wee bit of randomness

    def add_entries(fr0m, to, value):
        from_offset, from_note = fr0m
        to_offset, to_note = to
        for i in np.arange(12):
            m[from_offset + (from_note + i) % 12,
              to_offset + (to_note + i) % 12] = value

    add_entries((24, 0), (0, 2),  1)
    add_entries((24, 0), (0, 4),  .7)
    add_entries((24, 0), (24, 5), 1)
    add_entries((24, 0), (12, 7), .7)
    add_entries((24, 0), (0, 9),  .6)
    add_entries((24, 0), (12, 4), .6)
    add_entries((0, 2),  (24, 0), .5)
    add_entries((0, 2),  (0, 4),  .5)
    add_entries((0, 2),  (24, 5), .6)
    add_entries((0, 2),  (12, 7), 1)
    add_entries((0, 2),  (0, 9),  .6)
    add_entries((0, 2),  (12, 4), .2)
    add_entries((0, 4),  (24, 0), .5)
    add_entries((0, 4),  (0, 2),  .7)
    add_entries((0, 4),  (24, 5), .4)
    add_entries((0, 4),  (12, 7), .2)
    add_entries((0, 4),  (0, 9),  .8)
    add_entries((0, 4),  (12, 4), .7)
    add_entries((24, 5), (24, 0), .8)
    add_entries((24, 5), (0, 2),  .8)
    add_entries((24, 5), (0, 4),  .3)
    add_entries((24, 5), (12, 7), .9)
    add_entries((24, 5), (0, 9),  .5)
    add_entries((24, 5), (12, 4), .5)
    add_entries((12, 7), (24, 0), 1)
    add_entries((12, 7), (0, 2),  .4)
    add_entries((12, 7), (0, 4),  .2)
    add_entries((12, 7), (24, 5), .5)
    add_entries((12, 7), (0, 9),  .4)
    add_entries((12, 7), (12, 4), .2)
    add_entries((0, 9),  (24, 0), .5)
    add_entries((0, 9),  (0, 2),  .7)
    add_entries((0, 9),  (0, 4),  .7)
    add_entries((0, 9),  (24, 5), .6)
    add_entries((0, 9),  (12, 7), .4)
    add_entries((0, 9),  (12, 4), .7)
    add_entries((12, 4), (24, 0), .3)
    add_entries((12, 4), (0, 2),  .2)
    add_entries((12, 4), (0, 4),  .3)
    add_entries((12, 4), (24, 5), .8)
    add_entries((12, 4), (12, 7), .2)
    add_entries((12, 4), (0, 9),  1)

    return m


# A couple of helper functions:

def normalise(profile):
    mx = np.max(profile)
    if mx > 0:
        return profile / mx
    else:
        return profile

note_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
             'F#', 'G', 'G#','A', 'A#', 'B']

def arp_to_string(arp):
    return map(lambda n: note_names[n % 12] if n >= 0 else '-', arp)

def chord_to_string(chord):
    if chord is None:
        return "None"

    offset = int(chord / 12)
    note = chord % 12
    suffixes = ['m', '7', '']
    return note_names[note] + suffixes[offset]

def get_midi_out():
    import pypm
    pypm.Initialize()
    dev = pypm.GetDefaultOutputDeviceID()
    midi_out = pypm.Output(dev)
    return midi_out


class Arrangement:
    """
    An on-or-off type arrangement.
    """

    def __init__(self, keyframes):
        self.keyframes = keyframes
        self.time = 0

    def step(self):
        self.time += 1

    def set_active(self, agents):
        for agent in agents:
            agent.active = agent.uid in self.current_keyframe()

    def finished(self):
        return len(self.current_keyframe()) == 0

    def current_keyframe(self):
        # lazy brute force
        for k in reversed(sorted(self.keyframes.iterkeys())):
            if self.time >= k:
                return self.keyframes[k]


def main():

    midi_out = get_midi_out()

    agents = [Agent(memory_length = 8,
                    chord_templates = get_my_simple_chord_templates(),
                    progression_scores = get_my_progression_scores(),
                    pitch_bounds = (60, 80),
                    length_bounds = (3, 5),
                    allow_pause = True),
              Agent(memory_length = 8,
                    chord_templates = get_my_simple_chord_templates(),
                    progression_scores = get_my_progression_scores(),
                    pitch_bounds = (60, 80),
                    length_bounds = (3, 6),
                    allow_pause = True),
              Agent(memory_length = 7,
                    chord_templates = get_my_simple_chord_templates(),
                    progression_scores = get_my_progression_scores(),
                    pitch_bounds = (37, 50),
                    length_bounds = (3, 5)),
              Agent(memory_length = 6,
                    chord_templates = get_my_simple_chord_templates(),
                    progression_scores = get_my_progression_scores(),
                    pitch_bounds = (70, 90),
                    length_bounds = (3, 9),
                    allow_pause = True,
                    density = 0,
                    length_factor = 0,
                    chord_change_timeout = 40),
              Agent(memory_length = 7,
                    chord_templates = get_my_simple_chord_templates(),
                    progression_scores = get_my_progression_scores(),
                    pitch_bounds = (60, 90),
                    length_bounds = (7, 15),
                    allow_pause = True,
                    density = 0.05,
                    length_factor = 1,
                    chord_change_timeout = 40)
              ]

    arrangement = Arrangement({0: [2, 3], 40: [0, 1, 2], 200: [0, 1, 3], 300: [0, 1, 2, 3],
                               440: [0, 1, 2], 600: [0, 1, 2, 4], 1000: [2, 4], 1200: [4], 1250: []})

    while True:

        arrangement.step()
        if arrangement.finished():
            break

        arrangement.set_active(agents)

        notes = [None] * len(agents)
        for i, agent in enumerate(agents):
            note = agent.get_note() # even if inactive, agents hum along silently
            if agent.active:
                notes[i] = note

        if debugging:
            print("\nIteration %d:" % (arrangement.time))
            for agent in agents:
                print(agent.get_state_string())

        for i, note in enumerate(notes):
            if note is not None:
                midi_out.WriteShort(0x90 + i, note, 100)
                for agent in agents:
                    agent.hear(note)

        time.sleep(0.2)
        for i, note in enumerate(notes):
            if note is not None:
                midi_out.WriteShort(0x80 + i, note, 80)

        for agent in agents:
            agent.forget()
                

if __name__ == '__main__':
    main()
