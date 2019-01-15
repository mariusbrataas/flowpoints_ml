import React from 'react';

export function breakUpLines(msg, addition) {
  // Pun intended
  addition.split('\n').map(val => {
    msg.push(val)
    msg.push(<br />)
  })
}

export function getFlowCodeName(flowpoints, key) {
  return flowpoints[key].name ? flowpoints[key].name : 'flowpoint_'.concat(key)
}

function recIsDownStream(targetkey, myKey, testorder, depth, order, maxdepth) {
  if (order.includes(myKey)) {return false}
  if (targetkey === myKey) {return true}
  testorder[myKey].visited = true
  var isDone = false
  if (depth < maxdepth) {
    testorder[myKey].outputs.map(testkey => {
      if (!isDone && !testorder[testkey].visited) {
        isDone = recIsDownStream(targetkey, testkey, testorder, depth + 1, order, maxdepth)
      }
    })
  }
  return isDone
}

export function isDownStream(flowpoints, targetkey, myKey, order) {
  var testorder = {}
  Object.keys(flowpoints).map((key) => {
    testorder[key] = {
      visited: false,
      outputs: flowpoints[key].outputs,
      inputs: flowpoints[key].inputs
    }
  })
  return recIsDownStream(targetkey, myKey, testorder, 0, order, Object.keys(flowpoints).length)
}
