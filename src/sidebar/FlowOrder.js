import React from 'react';


export function getPointName(lib, key) {
  const p = lib[key]
  return (p.name === '' ? ('p_' + key) : p.name).replace(/ /g, '_')
}


function prepPoints(flowpoints) {

  // Helpers
  var n_inputs = 0
  var lib = {}

  // Adding all keys to lib
  Object.keys(flowpoints).map(key => {
    const p = flowpoints[key];
    lib[key] = {
      specTitle: p.specs.title,
      inputs: {},
      outputs: {},
      name: p.name,
      ref: p.specs.ref,
      params: p.specs.params,
      index: null,
      visited: false,
      downstreamVisited: false,
      ready: false,
      statename: null,
      needState: false,
    }
    if (p.specs.title === 'Input') {
      lib[key].index = n_inputs;
      lib[key].ready = true;
      lib[key].statename = (p.name === '' ? ('input_' + n_inputs) : p.name);
      n_inputs = n_inputs + 1
    }
  })

  // Adding connection pointers
  Object.keys(flowpoints).map(key => {
    var p = lib[key];
    Object.keys(flowpoints[key].outputs).map(out_key => {
      p.outputs[out_key] = lib[out_key]
      if (!(out_key in lib)) {
        console.log(out_key, lib)
      } else {
        lib[out_key].inputs[key] = lib[key]
      }
    })
  })

  // Returning
  return lib

}



function recursive_isDownStream(lib, targetKey, myKey, depth, maxDepth) {

  // Returning if matching
  if (targetKey === myKey) return true

  // Marking self as visited on this run
  lib[myKey].downstreamVisited = true;

  // Helper
  var isDone = false;

  // Testing outputs
  if (depth < maxDepth) {
    lib[myKey].outputs.map(out_key => {
      if (!isDone && !lib[out_key].downstreamVisited) {
        isDone = recursive_isDownStream(lib, targetKey, out_key, depth + 1, maxDepth);
      }
    })
  }

  // Returning
  return isDone

}


function isDownStream(lib, targetKey, myKey) {

  // Prepping lib
  Object.keys(lib).map(key => {
    lib[key].downstreamVisited = false;
  })

  // Testing
  const isDone = recursive_isDownStream(lib, targetKey, myKey, 0, 100)

  // Returning
  return isDone

}


function isReady(lib, myKey) {
  var ready = true;
  Object.keys(lib[myKey].inputs).map(key => {
    if (!lib[key].ready) ready = false
  })
  return ready
}


function recursive_getOrder(lib, myKey, order, depth, maxDepth) {

  // Helper
  var me = lib[myKey]

  if (me) {
    // Visited before? -> Return
    if (me.visited) return order

    // All inputs ready?
    me.ready = isReady(lib, myKey)

    // Visiting and continuing recursion downstream
    if (me.ready) {
      me.visited = true;
      if (!order.includes(myKey)) order.push(myKey)
      Object.keys(me.outputs).map(out_key => {
        order = recursive_getOrder(lib, out_key, order, depth + 1, maxDepth)
      })
    }
  }

  // Returning
  return order

}


function getLastUser(lib, order, target_key) {
  var maxIndex = -1;
  order.map((test_key, index) => {
    if (target_key in lib[test_key].inputs) maxIndex = Math.max(maxIndex, index);
  })
  return order[maxIndex]
}


function getStateNames(lib, order) {
  var states = {};
  order.map((key, idx) => {
    const p = lib[key];
    if (!(key in states)) {
      states[key] = getPointName(lib, key) + (p.specTitle === 'Input' ? '' : '_state');
    }
    const lastUser = getLastUser(lib, order, key);
    if (lastUser) {
      states[lastUser] = states[key]
    }
  })
  states[order[order.length - 1]] = 'self.state'
  return states
}


export function FlowOrder(flowpoints) {

  // Helpers
  var order = []
  var inputs = {}
  var states = {}

  // Prepping
  var lib = prepPoints(flowpoints)

  // Getting inputs
  Object.keys(lib).map(key => {
    if (lib[key].ready) {
      order.push(key)
      inputs[key] = lib[key];
    }
  })

  // Testing inputs and downstream
  Object.keys(inputs).map(inp_key => {
    order = recursive_getOrder(lib, inp_key, order, 0, 100)
  })

  // Inputs always have their own states
  var states = getStateNames(lib, order);

  // Returning
  return { lib:lib, order:order, states:states, inputs:Object.keys(inputs) }

}
