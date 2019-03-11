import React, { Component } from 'react';

import { getPointName } from '../../FlowOrder.js';


function Initials(link) {
  var msg = "'''"
  msg += "\nCreated with https://mariusbrataas.github.io/flowpoints_ml/"
  if (link) {
    msg += "\n\nLink to model:\n" + link
  }
  msg += "\n\nLICENSE:"
  msg += "\nhttps://github.com/mariusbrataas/flowpoints_ml/blob/master/LICENSE"
  msg += "\n'''"
  return msg
}


function Imports() {
  var msg = "# Importing torch tools"
  msg += "\nimport torch"
  msg += "\nfrom torch import nn, optim, cuda"
  msg += "\n\n"
  msg += "\n# Importing other libraries"
  msg += "\nimport numpy as np"
  msg += "\nimport matplotlib.pyplot as plt"
  msg += "\nimport time"
  return msg
}


function Constructor(flowpoints, order) {

  // Basics
  var msg = "# Model"
  msg += "\nclass neural_net(nn.Module):"
  msg += "\n\n\n    def __init__(self, optimizer=optim.SGD, alpha=0.01, criterion=nn.CrossEntropyLoss(), use_cuda=None):"
  msg += "\n\n        # Basics"
  msg += "\n        super(neural_net, self).__init__()"
  msg += "\n\n        # Settings"
  msg += "\n        self.optim_type = optimizer"
  msg += "\n        self.optimizer  = None"
  msg += "\n        self.alpha      = alpha"
  msg += "\n        self.criterion  = criterion"
  msg += "\n\n        # Use CUDA?"
  msg += "\n        self.use_cuda = use_cuda if (use_cuda != None and cuda.is_available()) else cuda.is_available()"
  msg += "\n\n        # Current loss and loss history"
  msg += "\n        self.train_loss      = 0"
  msg += "\n        self.valid_loss      = 0"
  msg += "\n        self.train_loss_hist = []"
  msg += "\n        self.valid_loss_hist = []"
  msg += "\n\n        # State"
  msg += "\n        self.state = None"

  // Initializing modules
  msg += "\n\n        # Adding all modules"
  order.map(key => {
    var p = flowpoints[key];
    if (p.specs.title !== 'Input') {

      // Getting longest param name
      var max_l = 0;
      Object.keys(p.specs.params).map(param_key => {max_l = Math.max(param_key.length, max_l)})

      // Init object
      msg += "\n        self." + getPointName(flowpoints, key) + " = " + p.specs.ref + '('

      // Adding arguments
      Object.keys(p.specs.params).map(param_key => {
        msg += "\n            " + param_key
        for (var i = 0; i < max_l - param_key.length; i++) msg += ' '
        msg += ' = '
        if (p.specs.params[param_key].type === 'tuple') {
          msg += '('
          p.specs.params[param_key].current.map(val => {msg += (val === '' ? p.specs.params[param_key].min : val) + ','})
          msg = msg.substring(0, msg.length - 1) + '),'
        } else if (p.specs.params[param_key].type === 'select') {
          msg += "'" + (p.specs.params[param_key].current === '' ? p.specs.params[param_key].min : p.specs.params[param_key].current) + "'" + ','
        } else {
          msg += (p.specs.params[param_key].current === '' ? p.specs.params[param_key].min : p.specs.params[param_key].current) + ','
        }
      })
      if (max_l > 0) msg += '\n        '
      msg += ')'
    }
  })

  // Startup routines
  msg += "\n\n        # Running startup routines"
  msg += "\n        self.startup_routines()"

  // Returning
  return msg

}



function StartupRoutines() {
  var msg = "    def startup_routines(self):"
  msg += "\n        self.optimizer = self.optim_type(self.parameters(), lr=self.alpha)"
  msg += "\n        if self.use_cuda:"
  msg += "\n            self.cuda()"
  return msg
}



function Predict() {
  var msg = "    def predict(self, x):"
  msg += "\n        "
  msg += "\n        # Switching off auto-grad"
  msg += "\n        with torch.no_grad():"
  msg += "\n            "
  msg += "\n            # Use CUDA?"
  msg += "\n            if self.use_cuda:"
  msg += "\n                x = x.cuda()"
  msg += "\n            "
  msg += "\n            # Running inference"
  msg += "\n            return self.forward(x)"
  return msg
}



function Forward(lib, states, order, inputs) {
  var msg = "    def forward(self, "

  // Adding inputs
  inputs.map(key => { msg += states[key] + ', ' })
  if (inputs.length > 0) msg = msg.substring(0, msg.length - 2)
  msg += '):'

  // Forwarding
  var steplib = []
  order.map(key => {
    const p = lib[key];
    if (!(p.specTitle === 'Input')) {
      var stepmsg = "        " + states[key] + ' = '
      stepmsg += 'self.' + getPointName(lib, key) + '('
      Object.keys(p.inputs).map(inp_key => { stepmsg += states[inp_key] + ' + ' })
      stepmsg = stepmsg.substring(0, stepmsg.length - 3) + ')'
      steplib.push({msg:stepmsg, title:p.specTitle})
    }
  })

  // Adding steps
  var maxL = 0
  steplib.map(step => {maxL = Math.max(maxL, step.msg.length)})
  steplib.map(step => {
    msg += '\n' + step.msg
    Array.from(Array(maxL + 1 - step.msg.length).keys()).map(idx => {msg += ' '})
    msg += '# ' + step.title
  })

  // Adding return
  msg += '\n        return self.state'

  // Returning
  return msg

}


function FitStep() {
  var msg = "    def fit_step(self, train_loader):"
  msg += '\n\n        # Preparations for fit step'
  msg += '\n        self.train_loss = 0 # Resetting training loss'
  msg += '\n        self.train()        # Switching to autograd'
  msg += '\n\n        # Looping through data'
  msg += '\n        for x, y in train_loader:'
  msg += '\n\n            # Use CUDA?'
  msg += '\n            if self.use_cuda:'
  msg += '\n                x, y = x.cuda(), y.cuda()        # Moving tensors to GPU'
  msg += '\n\n            # Performing calculations'
  msg += '\n            self.forward(x)                      # Forward pass'
  msg += '\n            loss = self.criterion(self.state, y) # Calculating loss'
  msg += '\n            self.train_loss += loss.item()       # Adding to epoch loss'
  msg += '\n            loss.backward()                      # Backward pass'
  msg += '\n            self.optimizer.step()                # Optimizing weights'
  msg += '\n            self.optimizer.zero_grad()           # Clearing gradients'
  msg += '\n\n        # Adding loss to history'
  msg += '\n        self.train_loss_hist.append(self.train_loss / len(train_loader))'

  // Returning
  return msg
}


function ValidationStep() {
  var msg = "    def validation_step(self, validation_loader):"
  msg += '\n\n        # Preparations for validation step'
  msg += '\n        self.valid_loss = 0 # Resetting validation loss'
  msg += '\n\n        # Swithing off autograd'
  msg += '\n        with torch.no_grad():'
  msg += '\n\n            # Looping through data'
  msg += '\n            for x, y in validation_loader:'
  msg += '\n\n                # Use CUDA?'
  msg += '\n                if self.use_cuda:'
  msg += '\n                    x, y = x.cuda(), y.cuda()        # Moving tensors to GPU'
  msg += '\n\n                # Performing calculations'
  msg += '\n                self.forward(x)                      # Forward pass'
  msg += '\n                loss = self.criterion(self.state, y) # Calculating loss'
  msg += '\n                self.valid_loss += loss.item()       # Adding to epoch loss'
  msg += '\n\n        # Adding loss to history'
  msg += '\n        self.valid_loss_hist.append(self.valid_loss / len(validation_loader))'

  // Returning
  return msg
}


function Fit() {
  var msg = '    def fit(self, train_loader, validation_loader=None, epochs=10, show_progress=True, save_best=False):'
  msg += '\n\n        # Helpers'
  msg += '\n\n        best_validation = 1e5'
  msg += '\n\n        # Possibly prepping progress message'
  msg += '\n        if show_progress:'
  msg += '\n            epoch_l = max(len(str(epochs)), 2)'
  msg += "\n            print('Training model...')"
  msg += "\n            print('%sEpoch   Training loss   Validation loss   Duration   Time remaining' % ''.rjust(2 * epoch_l - 4, ' '))"
  msg += '\n            t = time.time()'
  msg += '\n\n        # Looping through epochs'
  msg += '\n        for epoch in range(epochs):'
  msg += '\n            self.fit_step(train_loader)                 # Optimizing weights'
  msg += '\n            if validation_loader != None:               # Perform validation?'
  msg += '\n                self.validation_step(validation_loader) # Calculating validation loss'
  msg += '\n\n            # Possibly printing progress'
  msg += '\n            if show_progress:'
  msg += '\n                eta_s = (time.time() - t) * (epochs - epoch)'
  msg += "\n                eta = '%sm %ss' % (round(eta_s / 60), 60 - round(eta_s % 60))"
  msg += "\n                print('%s/%s' % (str(epoch + 1).rjust(epoch_l, ' '), str(epochs).ljust(epoch_l, ' ')),"
  msg += "\n                    '| %s' % str(round(self.train_loss_hist[-1], 8)).ljust(13, ' '),"
  msg += "\n                    '| %s' % str(round(self.valid_loss_hist[-1], 8)).ljust(15, ' '),"
  msg += "\n                    '| %ss' % str(round(time.time() - t, 3)).rjust(7, ' '),"
  msg += "\n                    '| %s' % eta.ljust(14, ' '))"
  msg += '\n                t = time.time()'
  msg += '\n\n            # Possibly saving model'
  msg += '\n            if save_best:'
  msg += '\n                if self.valid_loss_hist[-1] < best_validation:'
  msg += "\n                    self.save('best_validation')"
  msg += "\n                    best_validation = self.valid_loss_hist[-1]"
  msg += '\n\n        # Switching to eval'
  msg += '\n        self.eval()'

  // Returning
  return msg
}


function PlotHist() {
  var msg = '    def plot_hist(self):'
  msg += "\n\n        # Adding plots"
  msg += "\n        plt.plot(self.train_loss_hist, color='blue', label='Training loss')"
  msg += "\n        plt.plot(self.valid_loss_hist, color='springgreen', label='Validation loss')"
  msg += "\n\n        # Axis labels"
  msg += "\n        plt.xlabel('Epoch')"
  msg += "\n        plt.ylabel('Loss')"
  msg += "\n        plt.legend(loc='upper right')"
  msg += "\n\n        # Displaying plot"
  msg += "\n        plt.show()"

  // Returning
  return msg
}


function SaveLoad(lib, order) {
  var msg = "    def save(self, name='model.pth'):"
  msg += "\n        if not '.pth' in name: name += '.pth'"
  msg += "\n        torch.save({"
  order.map(key => {
    const point = lib[key]
    if (!(point.specTitle === 'Input')) {
      const pointcode = getPointName(lib, key)
      msg += "\n            '" + pointcode + "': self." + pointcode + ','
    }
  })
  msg += "\n            'train_loss':      self.train_loss,"
  msg += "\n            'valid_loss':      self.valid_loss,"
  msg += "\n            'train_loss_hist': self.train_loss_hist,"
  msg += "\n            'valid_loss_hist': self.valid_loss_hist,"
  msg += "\n            'optim_type':      self.optim_type,"
  msg += "\n            'alpha':           self.alpha,"
  msg += "\n            'criterion':       self.criterion,"
  msg += "\n            'use_cuda':        self.use_cuda"
  msg += "\n        }, name)"
  msg += "\n\n\n    @staticmethod"
  msg += "\n    def load(name='model.pth'):"
  msg += "\n        if not '.pth' in name: name += '.pth'"
  msg += "\n        checkpoint = torch.load(name)"
  msg += "\n        model = neural_net("
  msg += "\n            optimizer = checkpoint['optim_type'],"
  msg += "\n            alpha     = checkpoint['alpha'],"
  msg += "\n            criterion = checkpoint['criterion'],"
  msg += "\n            use_cuda  = checkpoint['use_cuda']"
  msg += "\n        )"
  order.map(key => {
    const point = lib[key]
    if (!(point.specTitle === 'Input')) {
      const pointcode = getPointName(lib, key)
      msg += "\n        model." + pointcode + " = checkpoint['" + pointcode + "']"
    }
  })
  msg += "\n        model.train_loss      = checkpoint['train_loss']"
  msg += "\n        model.valid_loss      = checkpoint['valid_loss']"
  msg += "\n        model.train_loss_hist = checkpoint['train_loss_hist']"
  msg += "\n        model.valid_loss_hist = checkpoint['valid_loss_hist']"
  msg += '\n        model.startup_routines()'
  msg += "\n        return model"

  // Returning
  return msg
}


function Flatten() {
  var msg = '# Helper class for flattening'
  msg += '\nclass Flatten(nn.Module):'
  msg += '\n    def __init__(self):'
  msg += '\n        super(Flatten, self).__init__()'
  msg += '\n    def forward(self, x):'
  msg += '\n        return x.view(x.shape[0], -1)'
  return msg
}


export function PyTorchParser(flowpoints, lib, link) {

  // Add flatten?
  var addflatten = false;
  Object.keys(flowpoints).map(key => {
    if (flowpoints[key].specs.title === 'Flatten') addflatten = true;
  })

  // Helpers
  const order = lib.order;
  const states = lib.states;
  const inputs = lib.inputs;

  // Initials
  var msg = Initials(link)
  msg += '\n\n\n' + Imports();

  // Add flatten?
  if (addflatten) msg += '\n\n\n' + Flatten()

  // Model
  msg += '\n\n\n' + Constructor(flowpoints, order, {});
  msg += '\n\n\n' + StartupRoutines();
  msg += '\n\n\n' + Predict();
  msg += '\n\n\n' + Forward(lib.lib, states, order, inputs);
  msg += '\n\n\n' + FitStep();
  msg += '\n\n\n' + ValidationStep();
  msg += '\n\n\n' + Fit();
  msg += '\n\n\n' + PlotHist();
  msg += '\n\n\n' + SaveLoad(lib.lib, order);

  // Returning
  return msg

}
