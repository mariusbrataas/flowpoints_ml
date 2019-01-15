import React from 'react';

import { getFlowCodeName, isDownStream } from './CommonTools'

function getLastUser(testorder, targetkey, order) {
  var maxIndex = -1
  order.map((testkey, index) => {
    if (testorder[testkey].inputs.includes(targetkey)) {
      maxIndex = Math.max(maxIndex, index)
    }
  })
  return order[maxIndex]
}

export function getStateNames(flowpoints, order) {
  var statenames = {}
  var init_states = []
  order.map(key => {
    const point = flowpoints[key]
    if (!(key in statenames)){
      statenames[key] = getFlowCodeName(flowpoints, key)
      if (!flowpoints[key].flowtype.includes('input')) {
        if (!init_states.includes(statenames[key])) {
          init_states.push(key)
          statenames[key] = 'self.state_' + statenames[key]
        }
      }
    }
    const lastUser = getLastUser(flowpoints, key, order)
    if (lastUser) {
      statenames[lastUser] = statenames[key]
    }
  })
  return statenames
}
