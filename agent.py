import numpy as np
import random

class Agent:

    def __init__(self,
                 memory_length,
                 chord_templates,
                 progression_scores,
                 pitch_bounds,
                 length_bounds):

        self.consonance_desire = 10
        self.progression_desire = 0
        self.random_action_count = 5 # the higher the more likely it is to find a "good" action, so more likely to change
        self.pitch_bounds = pitch_bounds
        self.length_bounds = length_bounds
        self.arpeggio = np.empty(0, dtype=int)
        self.desire_direction = DesireDirection.towards_progression
        self.heard_notes = np.zeros((12, self.memory_length))

    # TODO: hear own note!
    def hear(self, note):
        self.heard_notes[note, 0] += 1

    def get_note(self):
        profile = self.get_heard_profile()
        chord = self.get_believed_chord(profile)
        self.update_arpeggio(chord)
        self.update_desire()
        return self.arpeggio[0]

    def forget(self):
        self.heard_notes = np.roll(self.heard_notes, 1, axis = 1)
        
    def update_arpeggio(self, chord):
        potential_actions = self.generate_random_actions()
        if len(self.arpeggio) >= length_bounds[0]:
            potential_actions = np.concatenate(NoChangeAction(self.arpeggio), potential_actions)

        max_score = -1
        best_arpeggio = None
        for action in potential_actions:
            arpeggio = action.execute(self.arpeggio)
            score = self.evaluate_arpeggio(arpeggio, chord)
            if score >= max_score: # `>=` means that "real" actions win over NoChangeAction (since it's at index 0)
                max_score = score
                best_arpeggio = arpeggio

        self.arpeggio = best_arpeggio
        np.roll(self.arpeggio, 1)

    def evaluate_arpeggio(self, arpeggio, chord):
        # TODO:
        #   ps = chord progression score based on the new arpeggio (and heard history minus own notes?)
        #   cs = consonance based on new arpeggio and history
        #   pd = progression_desire
        #   cd = consonance_desire
        #   return ps * pd + cs * cd

    def generate_random_actions(self):
        actions = [AddAction, RemoveAction, ShiftAction]
        return [random.choice(actions)(pitch_bounds, length_bounds)
                for i in range(self.random_action_count)]

    def update_desire(self):
        if self.desire_direction == DesireDirection.towards_consonance:
            self.consonance_desire += 1
            self.progression_desire -= 1
        else
            self.consonance_desire -= 1
            self.progression_desire += 1
        if self.consonance_desire == 0:
            self.desire_direction = DesireDirection.towards_consonance
        else if self.progression_desire == 0:
            self.desire_direction = DesireDirection.towards_progression

    def get_heard_profile(self):
        recency_bias = np.arange(self.memory_length, 0, -1). \
            reshape(1, self.memory_length).repeat(12, 0)
        return np.matrix.sum(self.heard_notes * recency_bias)

    def get_believed_chord(self, profile):
        max_score = -1
        best_i = -1
        for i, row in enumerate(self.chord_templates):
            score = sum(profile * row)
            if score > max_score:
                max_score = score
                best_i = i
        return best_i

    

class DesireDirection:
    towards_consonance, towards_progression = range(2)


class Action:
    def __init__(self, pitch_bounds, length_bounds):
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
        note = np.random.randint(self.pitch_bounds[0], self.pitch_bounds[1])
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
        note = np.random.randint(self.pitch_bounds[0], self.pitch_bounds[1])
        arpeggio[index] = note
        return arpeggio

