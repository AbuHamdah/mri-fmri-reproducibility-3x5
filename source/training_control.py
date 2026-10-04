"""Validation-only learning-rate scheduling and early stopping for new runs."""
import math


class ValidationControl:
    def __init__(self, patience=10, lr_patience=4, min_delta=1e-4, factor=0.5, min_lr=1e-7):
        self.patience, self.lr_patience = patience, lr_patience
        self.min_delta, self.factor, self.min_lr = min_delta, factor, min_lr
        self.best = math.inf
        self.wait = self.lr_wait = 0

    def observe(self, validation_loss, learning_rate):
        if not math.isfinite(validation_loss):
            raise ValueError('Nonfinite validation loss')
        improved = validation_loss < self.best - self.min_delta
        if improved:
            self.best = validation_loss
            self.wait = self.lr_wait = 0
        else:
            self.wait += 1
            self.lr_wait += 1
        next_lr = learning_rate
        if self.lr_wait >= self.lr_patience:
            next_lr = max(self.min_lr, learning_rate * self.factor)
            self.lr_wait = 0
        return improved, next_lr, self.wait >= self.patience


def dropout_configuration(model):
    result = []
    def visit(layer, prefix):
        name = prefix + layer.name
        for key in ('rate', 'dropout', 'recurrent_dropout'):
            if hasattr(layer, key):
                result.append({'layer': name, 'parameter': key, 'value': float(getattr(layer, key))})
        for child in getattr(layer, 'layers', []):
            visit(child, name + '/')
        for key in ('layer', 'forward_layer', 'backward_layer'):
            child = getattr(layer, key, None)
            if child is not None:
                visit(child, name + '/')
    visit(model, '')
    return result
