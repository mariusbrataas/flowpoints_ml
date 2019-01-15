import React from 'react';

import { getFlowCodeName } from './CommonTools'



export function getFitStep() {
  var msg = '\n    def fit_step(self, train_loader):'
  msg += '\n        # Preparations for fit step'
  msg += '\n        self.train_loss = 0 # Resetting training loss'
  msg += '\n        self.train()        # Switching to autograd'
  msg += '\n        # Looping through data'
  msg += '\n        for x, y in train_loader:'
  msg += '\n            # Use CUDA?'
  msg += '\n            if self.use_cuda:'
  msg += '\n                x, y = x.cuda(), y.cuda()        # Moving tensors to GPU'
  msg += '\n            # Performing calculations'
  msg += '\n            self.forward(x)                      # Forward pass'
  msg += '\n            loss = self.criterion(self.state, y) # Calculating loss'
  msg += '\n            self.train_loss += loss.item()       # Adding to epoch loss'
  msg += '\n            loss.backward()                      # Backward pass'
  msg += '\n            self.optimizer.step()                # Optimizing weights'
  msg += '\n            self.optimizer.zero_grad()           # Clearing gradients'
  msg += '\n        # Adding loss to history'
  msg += '\n        self.train_loss_hist.append(self.train_loss / len(train_loader))'
  return msg
}

export function getValidationStep() {
  var msg = '\n    def validation_step(self, validation_loader):'
  msg += '\n        # Preparations for validation step'
  msg += '\n        self.valid_loss = 0 # Resetting validation loss'
  msg += '\n        # Switching off autograd'
  msg += '\n        with torch.no_grad():'
  msg += '\n            # Looping through data'
  msg += '\n            for x, y in validation_loader:'
  msg += '\n                # Use CUDA?'
  msg += '\n                if self.use_cuda:'
  msg += '\n                    x, y = x.cuda(), y.cuda()        # Moving tensors to GPU'
  msg += '\n                # Performing calculations'
  msg += '\n                self.forward(x)                      # Forward pass'
  msg += '\n                loss = self.criterion(self.state, y) # Calculating loss'
  msg += '\n                self.valid_loss += loss.item()       # Adding to epoch loss'
  msg += '\n        # Adding loss to history'
  msg += '\n        self.valid_loss_hist.append(self.valid_loss / len(validation_loader))'
  return msg
}
