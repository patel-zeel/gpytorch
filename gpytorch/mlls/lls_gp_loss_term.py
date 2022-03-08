#!/usr/bin/env python3

from .added_loss_term import AddedLossTerm


class LLSGPLossTerm(AddedLossTerm):
    def __init__(self, term):
        self.term = term

    def loss(self, *params):
        return self.term
