import React from 'react';

import { getFlowCodeName, isDownStream } from './CommonTools'


function recgetOrdered(testorder, order, myKey, depth) {
  myKey = parseInt(myKey)
  var myData = testorder[myKey]
  myData.visited = true
  if (!myData.finished) {
    myData.ready = true
    myData.inputs.map(testkey => {
      if (!testorder[testkey].ready) {
        if (!isDownStream(testorder, testkey, myKey, order)) {
          myData.ready = false
        }
      }
    })
    if (myData.ready && depth < 500) {
      if (!order.includes(myKey)) {
        order.push(myKey)
      }
      myData.finished = true
      myData.outputs.map(testkey => {
        order = recgetOrdered(testorder, order, testkey, depth + 1)
      })
    }
  }
  return order
}

export function getOrdered(flowpoints, inps) {
  var order = []
  var testorder = {}
  Object.keys(flowpoints).map((key) => {
    key = parseInt(key)
    if (flowpoints[key].flowtype.includes('input')) {
      order.push(key)
    }
    testorder[key] = {
      visited: false,
      ready: false,
      finished: false,
      name: flowpoints[key].name,
      outputs: flowpoints[key].outputs,
      inputs: flowpoints[key].inputs
    }
  })
  inps.map(testkey => {
    testorder[testkey].visited = true
    testorder[testkey].ready = true
  })
  return recgetOrdered(testorder, order, inps[0] ? inps[0] : Object.keys(flowpoints)[0], 0)
}

export function getInputs(flowpoints) {
  var inps = []
  Object.keys(flowpoints).map(testkey => {
    if (flowpoints[testkey].flowtype.includes('input')) {
      inps.push(testkey)
    }
  })
  return inps
}
