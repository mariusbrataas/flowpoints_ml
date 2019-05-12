export function getPointName(flowpoints, key) {
  let p = flowpoints[key];
  return (p.name === '' ? ('p_' + key) : p.name).replace(/ /g, '_')
}


export function dent(indent, n) {
  var msg = ''
  Array.from(Array(n).keys()).map(idx => msg += indent)
  return msg
}


export function FormatInitParams(parameters, indentation, indent, padding) {
  if (padding === undefined) padding = true
  // Getting longest parameter name
  var max_l = 0;
  Object.keys(parameters).map(p_key => {
    if (parameters[p_key].value !== 'None') max_l = Math.max(p_key.length, max_l)
  })
  var msg = ''
  if (padding) msg += '('
  Object.keys(parameters).map(p_key => {
    let param = parameters[p_key];

    if (param.value !== 'None') {
      // Adding argument
      msg += '\n' + dent(indent, indentation + 1) + p_key

      // Aligning with other arguments
      for (var i = 0; i < max_l - p_key.length; i++) msg += ' '
      msg += ' = '

      // Adding argument values
      if (param.istuple) {
        // Tuples are put in between parantheses
        msg += '(';
        param.value.map(val => msg += (val === '' ? (param.min === Infinity ? 0 : param.min) : val) + ',');
        msg = msg.substring(0, msg.length - 1) + ')';
      } else if (param.type === 'select') {
        msg += "'" + param.value + "'";
      } else if (param.type === 'bool') {
        msg += (param.value === true ? 'True' : 'False')
      } else if (param.type === 'string') {
        msg += '"' + param.value + '"'
      } else {
        msg += (param.value === '' ? (param.min === Infinity ? 0 : param.min) : param.value)
      }
      msg += ','
    }
  })
  if (padding) {
    if (max_l > 0) {
      msg += '\n' + dent(indent, 1)
    }
    msg += ')'
  }
  return msg
}


export function FormatParamInputs(flowpoints, input_keys) {
  var msg = ''
  input_keys.map(key => msg += getPointName(flowpoints, key) + ', ')
  if (input_keys.length > 0) msg = msg.substring(0, msg.length - 2)
  return msg
}


function getDummyLibrary(flowpoints) {
  var dummies = {}
  Object.keys(flowpoints).map(key => {
    var dummy = {
      base_ref: flowpoints[key].base_ref,
      name: flowpoints[key].name,
      inputs: [],
      outputs: [],
      ordered: false,
      ready: false,
      downStreamTested: false,
      is_recurrent: false
    }
    Object.keys(flowpoints[key].outputs).map(out_key => {
      dummy.outputs.push(out_key)
    })
    if (flowpoints[key].base_ref === 'Input') {
      dummy.ready = true;
    }
    dummies[key] = dummy;
  })

  // Adding corresponding inputs
  Object.keys(dummies).map(key => {
    dummies[key].outputs.map(out_key => {
      dummies[out_key].inputs.push(key)
    })
  })

  // Returning
  return dummies

}


function getInputs(flowpoints) {
  var inps = [];
  Object.keys(flowpoints).map(key => {
    if (flowpoints[key].base_ref === 'Input') inps.push(key)
  })
  return inps
}


function recIsDownStream(dummies, target_key, test_key) {

  // Returning if matching
  if (target_key === test_key) return true;

  // Continue only if havent been visited before
  var found_downstream = false
  if (!dummies[test_key].downStreamTested) {

    // Marking self as visited on this run
    dummies[test_key].downStreamTested = true;

    // Looking deeper down
    dummies[test_key].outputs.map(out_key => {
      if (!found_downstream) {
        found_downstream = recIsDownStream(dummies, target_key, out_key)
      }
    })
  }

  // Returning
  return found_downstream

}


function isDownStream(dummies, target_key, test_key) {

  // Resetting downstream helpers
  Object.keys(dummies).map(r_key => dummies[r_key].downStreamTested = false)

  // Running checks
  return recIsDownStream(dummies, target_key, test_key)

}


function isReady(dummies, key) {
  var ready = true;
  dummies[key].inputs.map(test_key => {
    if ((!dummies[test_key].ready && !isDownStream(dummies, test_key, key))) {
      ready = false;
    }
  })
  return ready
}


function recGetOrder(order, dummies, key) {

  // Helper
  var dummy = dummies[key];

  // Checking dummy, going deeper
  if (dummy) {

    // Visited before? -> Return
    if (dummy.ordered) return order

    // Dummy ready?
    if (isReady(dummies, key)) {

      // Making sure dummy is ready
      dummy.ready = true;

      // Making sure dummy is added to order
      dummy.ordered = true;
      if (!order.includes(key)) order.push(key)

      // Going deeper
      dummy.outputs.sort().map(out_key => order = recGetOrder(order, dummies, out_key))

    }

  }

  // Returning
  return order

}


function getLastUser(order, dummies, target_key) {
  var max_index = -1;
  order.map((test_key, idx) => {
    if (dummies[test_key].inputs.includes(target_key)) max_index = Math.max(max_index, idx)
  })
  return order[max_index]
}


function getStateNames(order, dummies) {
  var states = {};

  // Adding all states
  order.map((key, idx) => {
    const point = dummies[key];

    // Adding to states
    if (!(key in states)) {
      states[key] = getPointName(dummies, key) + (point.base_ref === 'Input' ? '' : '_state');
    }

    // Setting state of last user
    const last_user = getLastUser(order, dummies, key);
    if (last_user !== null) {
      states[last_user] = states[key];
    }

  })

  // Setting state of last node to default
  states[order[order.length - 1]] = 'self.state'

  // Returning
  return states

}


export function FlowOrder(state) {

  // Init helpers
  var inps = getInputs(state.flowpoints);
  var dummies = getDummyLibrary(state.flowpoints);
  var order = [];
  var init_states = [];
  var outs = [];

  // Adding input flowpoints to beginning of order
  inps.map(key => order.push(key));

  // Recursively mapping tree
  inps.map(key => order = recGetOrder(order, dummies, key));

  // Flagging flowpoints with recurrent connections and getting init states
  order.map((key, idx) => {
    var is_recurrent = false;
    dummies[key].outputs.map(out_key => {
      if (idx > order.indexOf(out_key) && !is_recurrent) is_recurrent = true;
    })
    if (is_recurrent || state.flowpoints[key].is_output) {
      dummies[key].is_recurrent = true;
      init_states.push(key)
    }
  })

  // Outputs
  order.map(key => {
    if (state.flowpoints[key].is_output) outs.push(key)
  })
  if (outs.length === 0) outs = [order[order.length - 1]]

  // State names
  var states = getStateNames(order, dummies);

  // Returning
  return {order, inps, states, dummies, init_states, outs}

}