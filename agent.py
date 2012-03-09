import numpy as np
import math
import random
from pprint import pprint

class Agent:

    def __init__(self,
                 memory_length,
                 chord_templates,
                 progression_scores,
                 pitch_bounds,
                 length_bounds):

        self.memory_length = memory_length
        self.chord_templates = chord_templates
        self.progression_scores = progression_scores
        self.pitch_bounds = pitch_bounds
        self.length_bounds = length_bounds

        self.max_desire = 10
        self.consonance_desire = self.max_desire
        self.progression_desire = 0
        self.random_action_count = 10 # the higher the more likely it is to find a "good" action, so more likely to change
        self.heard_notes = np.zeros((12, self.memory_length))
        self.arpeggio = self.get_initial_arpeggio()

    def hear(self, note):
        self.heard_notes[note % 12, 0] += 1

    def get_note(self):
        profile = self.get_heard_profile()
        chord_changed = self.update_arpeggio(profile)
        self.update_desire(chord_changed)
        return self.arpeggio[0]

    def forget(self):
        self.heard_notes = np.roll(self.heard_notes, 1, axis = 1)
        
    def update_arpeggio(self, profile):
        potential_actions = self.generate_random_actions()
        if len(self.arpeggio) >= self.length_bounds[0]:
            potential_actions = [NoChangeAction()] + potential_actions

        max_score = -1
        best_arpeggio = None
        chord_changed = False
        for action in potential_actions:
            arpeggio = action.execute(np.copy(self.arpeggio))
            score, is_chord_change = self.evaluate_arpeggio(arpeggio, profile)
            if score >= max_score: # `>=` means that "real" actions win over NoChangeAction (since it's at index 0)
                max_score = score
                best_arpeggio = arpeggio
                chord_changed = is_chord_change

        self.arpeggio = best_arpeggio
        self.arpeggio = np.roll(self.arpeggio, 1)

        return chord_changed

    def update_desire(self, chord_changed):
        if chord_changed:
            self.progression_desire = 0
            self.consonance_desire = self.max_desire
        elif self.consonance_desire > 0:
            self.progression_desire += 1
            self.consonance_desire -= 1

    def evaluate_arpeggio(self, arpeggio, profile):
        chord = self.get_chords_for_profile(profile)[0] # assuming there's enough data to avoid ambiguity
        
        arp_profile = self.get_arpeggio_profile(arpeggio)
        potential_arp_chords = self.get_chords_for_profile(arp_profile)

        ps = self.progression_score(chord, potential_arp_chords)
        cs = self.consonance_score(profile, arp_profile)
        pd = self.progression_desire
        cd = self.consonance_desire
        score = ps * pd + cs * cd
        is_chord_change = ps > cs

        # print("ps: %f, cs: %f, pd: %f, cd: %f" % (ps, cs, pd, cd))

        return (score, is_chord_change)

    def progression_score(self, from_chord, to_chords):
        # the best possible score
        return np.max(self.progression_scores[from_chord][to_chords])

    def consonance_score(self, chord, profile):
        return math.sqrt(np.sum(chord * profile))

    def generate_random_actions(self):
        actions = [AddAction, RemoveAction, ShiftAction]
        return [random.choice(actions)(self.pitch_bounds, self.length_bounds)
                for i in range(self.random_action_count)]

    def get_heard_profile(self):
        recency_bias = np.arange(self.memory_length, 0, -1). \
            reshape(1, self.memory_length).repeat(12, 0)
        profile = (self.heard_notes * recency_bias).sum(axis = 1)
        return normalise(profile)

    def get_chords_for_profile(self, profile):
        scores = (profile * self.chord_templates).sum(axis = 1)
        return np.concatenate(np.where(scores == np.max(scores)))

    def get_arpeggio_profile(self, arpeggio):
        profile = np.bincount(arpeggio % 12, minlength = 12)
        return normalise(profile)

    def get_initial_arpeggio(self):
        return np.array(random.sample(range(self.pitch_bounds[0], self.pitch_bounds[1]),
                                      random.randint(self.length_bounds[0], self.length_bounds[1])))


class DesireDirection:
    towards_consonance, towards_progression = range(2)


class Action:
    def __init__(self, pitch_bounds = None, length_bounds = None):
        self.pitch_bounds = pitch_bounds
        self.length_bounds = length_bounds

class NoChangeAction(Action):
    def execute(self, arpeggio):
        return arpeggio

class AddAction(Action):
    def execute(self, arpeggio):
        if len(arpeggio) == self.length_bounds[1]:
            return arpeggio
        index = np.random.randint(0, len(arpeggio) + 1)
        possible_notes = np.setdiff1d(np.arange(
                self.pitch_bounds[0], self.pitch_bounds[1]), arpeggio)
        note = random.sample(possible_notes, 1)[0]
        return np.insert(arpeggio, index, note)

class RemoveAction(Action):
    def execute(self, arpeggio):
        if len(arpeggio) == self.length_bounds[0]:
            return arpeggio
        index = np.random.randint(0, len(arpeggio))
        return np.delete(arpeggio, index)

class ShiftAction(Action):
    def execute(self, arpeggio):
        index = np.random.randint(0, len(arpeggio))
        possible_notes = np.setdiff1d(np.arange(
                self.pitch_bounds[0], self.pitch_bounds[1]), arpeggio)
        note = random.sample(possible_notes, 1)[0]
        arpeggio[index] = note
        return arpeggio


# this is the composition:

def get_ii_v_i_chord_templates():
    ii = np.array([1, 0, .7, .9, 0, .6, 0, .8, .3, .6, .8, 0])
    v = np.array([1, .4, .1, .6, .9, .1, .7, .1, .6, .1, .8, 0])
    i = np.array([1, 0, .7, 0, .9, .5, .2, .7, .1, .7, 0, .8])

    def get_chromatic_chunk(profile):
        m = profile
        for i in np.arange(1, 12):
            shifted = np.roll(profile, i)
            m = np.vstack((m, shifted))
        return m

    ii_chunk = get_chromatic_chunk(ii)
    v_chunk = get_chromatic_chunk(v)
    i_chunk = get_chromatic_chunk(i)

    return np.vstack((ii_chunk, v_chunk, i_chunk))

def get_ii_v_i_progression_scores():
    m = np.zeros((36, 36))

    def add_entries(fr0m, to, value):
        from_offset, from_note = fr0m
        to_offset, to_note = to
        for i in np.arange(12):
            m[from_offset + (from_note + i) % 12,
              to_offset + (to_note + i) % 12] = value

    add_entries((0, 2), (12, 7), 1)
    add_entries((12, 7), (24, 0), 1)
    add_entries((24, 0), (0, 2), 1)
    add_entries((24, 0), (0, 0), .3)

    return m

def get_krumhansl_chord_templates():
    pass

def get_krumhansl_progression_scores():
    pass

def normalise(profile):
    mx = np.max(profile)
    if mx > 0:
        return profile / mx
    else:
        return profile


def main():
    agents = [Agent(memory_length = 8,
                    chord_templates = get_ii_v_i_chord_templates(),
                    progression_scores = get_ii_v_i_progression_scores(),
                    pitch_bounds = (40, 60),
                    length_bounds = (3, 5)),
              Agent(memory_length = 8,
                    chord_templates = get_ii_v_i_chord_templates(),
                    progression_scores = get_ii_v_i_progression_scores(),
                    pitch_bounds = (60, 80),
                    length_bounds = (4, 7)),
              Agent(memory_length = 8,
                    chord_templates = get_ii_v_i_chord_templates(),
                    progression_scores = get_ii_v_i_progression_scores(),
                    pitch_bounds = (80, 100),
                    length_bounds = (5, 9))]

    for i in np.arange(100):

        notes = []
        for agent in agents:
            notes.append(agent.get_note())

        # print("%d\t%d\t%d" % (notes[0], notes[1], notes[2]))

        for note in notes:
            for agent in agents:
                agent.hear(note)

        for agent in agents:
            # print("arp: " + str(agent.arpeggio))
            agent.forget()
                

if __name__ == '__main__':
    main()
