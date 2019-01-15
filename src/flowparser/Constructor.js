import React from 'react';

import { getFlowCodeName } from './CommonTools'

function getFlatten() {
  var msg = '\n# Helper class for flattening'
  msg += '\nclass Flatten(nn.Module):'
  msg += '\n    def __init__(self):'
  msg += '\n        super(Flatten, self).__init__()'
  msg += '\n    def forward(self, x):'
  msg += '\n        return x.view(x.shape[0], -1)'
  return msg
}

function classBasics() {
  var msg = '\n# Main class'
  msg += '\nclass neural_net(nn.Module):'
  msg += '\n\n    def __init__(self, optimizer=optim.SGD, learn_rate=0.01, criterion=nn.CrossEntropyLoss(), use_cuda=None):'
  msg += '\n        super(neural_net, self).__init__()'
  msg += '\n        # Settings'
  msg += '\n        self.optim_type = optimizer'
  msg += '\n        self.optimizer = None'
  msg += '\n        self.learn_rate = learn_rate'
  msg += '\n        self.criterion = criterion'
  msg += '\n        # Use CUDA?'
  msg += '\n        self.use_cuda = use_cuda if (use_cuda != None and cuda.is_available()) else cuda.is_available()'
  msg += '\n        # Current loss and loss history'
  msg += '\n        self.train_loss = 0'
  msg += '\n        self.valid_loss = 0'
  msg += '\n        self.train_loss_hist = []'
  msg += '\n        self.valid_loss_hist = []'
  return msg
}

function initStates(order, statenames) {
  var msg = '        # States'
  msg += '\n        self.state = None'
  var initialized = []
  order.map(key => {
    if (statenames[key].includes('self.state_') && !initialized.includes(statenames[key])) {
      msg += '\n        ' + statenames[key] + ' = None'
      initialized.push(statenames[key])
    }
  })
  return msg
}

function parseInitArgs(config) {
  var msg = ''
  var longestKey = 0
  Object.keys(config.params).map(configkey => {
    longestKey = Math.max(longestKey, configkey.length)
  })
  longestKey += 1
  Object.keys(config.params).map(configkey => {
    const conf = config.params[configkey]
    msg += ',\n            ' + configkey
    for (var i = 0; i < longestKey - configkey.length; i++) {msg += ' '}
    msg += '= '
    if (conf.type.includes('tuple')) {
      msg += '('
      conf.current.map(val => {
        msg += val.toString() + ','
      })
      msg = msg.substring(0, msg.length - 1) + ')'
    } else {
      if (conf.type.includes('int') && conf.current < 0) {
        msg += 'None'
      } else {
        msg += conf.current
      }
    }
  })
  return '(' + msg.substring(1) + (longestKey > 1 ? '\n        )' : ')')
}

function initFlowpoints(flowpoints, order) {
  var msg = '        # Layers'
  order.map(key => {
    const point = flowpoints[key]
    if (!point.flowtype.includes('input')) {
      msg += '\n        self.' + getFlowCodeName(flowpoints, key) + ' = '
      var config = null
      if (point.flowtype in point.layertypes) {
        msg += point.layertypes[point.flowtype].ref
        config = point.layertypes[point.flowtype]
      } else {
        msg += point.activationtypes[point.flowtype].ref
        config = point.activationtypes[point.flowtype]
      }
      msg += parseInitArgs(config)
    }
  })
  return msg
}

function getRoutines() {
  var msg = '        # Running startup routines'
  msg += '\n        self.startup_routines()'
  msg += '\n\n    def startup_routines(self):'
  msg += '\n        self.optimizer = self.optim_type(self.parameters(), lr=self.learn_rate)'
  msg += '\n        if self.use_cuda:'
  msg += '\n            self.cuda()'
  return msg
}


export function getConstructor(flowpoints, order, inps, statenames) {
  var msg = ''
  var addFlatten = false
  order.map(key => {
    if (flowpoints[key].flowtype.includes('flatten')) {
      addFlatten = true
    }
  })
  if (addFlatten) {
    msg += getFlatten() + '\n'
  }
  msg += classBasics()
  msg += '\n' + initStates(order, statenames)
  msg += '\n' + initFlowpoints(flowpoints, order)
  msg += '\n' + getRoutines()
  return msg
}
