import React from 'react';

import { getFlowCodeName } from './CommonTools'

export function getForward(flowpoints, order, inps, statenames) {
  var last = ''
  var msg = '\n    def forward(self'
  inps.map(key => {
    msg += ', ' + statenames[key]
  })
  msg += '):'
  order.map(key => {
    if (!flowpoints[key].flowtype.includes('input')) {
      const point = flowpoints[key]
      last = statenames[key]
      msg += '\n        ' + statenames[key] + ' = ' + 'self.' + getFlowCodeName(flowpoints, key) + '('
      point.inputs.map((inpkey, index) => {
        if (index > 0) {
          msg += ' + '
        }
        msg += statenames[inpkey]
      })
      msg += ') # '
      if (point.flowtype in point.layertypes) {
        msg += point.layertypes[point.flowtype].ref.replace('nn.', '')
      } else {
        msg += point.activationtypes[point.flowtype].ref.replace('nn.', '')
      }
    }
  })
  msg += '\n        # Updating state and returning'
  msg += '\n        self.state = ' + last
  msg += '\n        return self.state'
  return msg
}
