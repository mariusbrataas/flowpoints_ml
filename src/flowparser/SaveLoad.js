import React, { Component } from 'react';

import { getFlowCodeName } from './CommonTools'

export function getSaveLoad(flowpoints, order) {
  var msg = "\n    def save(self, name='model.pth'):"
  msg += "\n        if not '.pth' in name: name += '.pth'"
  msg += "\n        torch.save({"
  order.map(key => {
    const point = flowpoints[key]
    if (!point.flowtype.includes('input')) {
      const pointcode = getFlowCodeName(flowpoints, key)
      msg += "\n            '" + pointcode + "': self." + pointcode + ','
    }
  })
  msg += "\n            'train_loss': self.train_loss,"
  msg += "\n            'valid_loss': self.valid_loss,"
  msg += "\n            'train_loss_hist': self.train_loss_hist,"
  msg += "\n            'valid_loss_hist': self.valid_loss_hist,"
  msg += "\n            'optim_type': self.optim_type,"
  msg += "\n            'learn_rate': self.learn_rate,"
  msg += "\n            'criterion': self.criterion,"
  msg += "\n            'use_cuda': self.use_cuda"
  msg += "\n        }, name)"
  msg += "\n\n    @staticmethod"
  msg += "\n    def load(name='model.pth'):"
  msg += "\n        if not '.pth' in name: name += '.pth'"
  msg += "\n        checkpoint = torch.load(name)"
  msg += "\n        model = neural_net("
  msg += "\n            optimizer  = checkpoint['optim_type'],"
  msg += "\n            learn_rate = checkpoint['learn_rate'],"
  msg += "\n            criterion  = checkpoint['criterion'],"
  msg += "\n            use_cuda   = checkpoint['use_cuda']"
  msg += "\n        )"
  order.map(key => {
    const point = flowpoints[key]
    if (!point.flowtype.includes('input')) {
      const pointcode = getFlowCodeName(flowpoints, key)
      msg += "\n        model." + pointcode + " = checkpoint['" + pointcode + "']"
    }
  })
  msg += "\n        model.train_loss = checkpoint['train_loss']"
  msg += "\n        model.valid_loss = checkpoint['valid_loss']"
  msg += "\n        model.train_loss_hist = checkpoint['train_loss_hist']"
  msg += "\n        model.valid_loss_hist = checkpoint['valid_loss_hist']"
  msg += '\n        model.startup_routines()'
  msg += "\n        return model"
  return msg
}
