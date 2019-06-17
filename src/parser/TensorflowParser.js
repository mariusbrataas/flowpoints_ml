import { getPointName, FormatParamInputs, dent, FormatInitParams } from "./FlowOrder";


function TensorFlowImports() {
  var msg = '# Importing TensorFlow tools'
  msg += '\nimport tensorflow as tf'
  return msg
}


function Constructor(state, order, inps, dummies, indent) {

  var flowpoints = state.flowpoints;
  var environment = state.environment;

  // Basics
  var msg = '# Model'
  msg += "\ndef NeuralNet(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']):"

  // Initializing inputs
  msg += '\n\n' + dent(indent, 1) + '# Initializing inputs'
  order.map(key => {
    let point = flowpoints[key]
    let current_point_name = getPointName(flowpoints, key);

    if (inps.includes(key)) {
      let content = point.content;
      msg += '\n' + dent(indent, 1) + current_point_name + ' = tf.keras.layers.Input(('
      Object.keys(content.dimensions).sort().map(idx => msg += '' + content.dimensions[idx] + ',')
      msg += "), name='" + current_point_name + "')"
    }
  })

  // Initializing layers
  var current_point_name = '';
  var current_point_key = '';
  var outputs = [];
  msg += '\n\n' + dent(indent, 1) + '# Initializing layers'
  order.map(key => {
    let point = flowpoints[key]
    current_point_name = getPointName(flowpoints, key);
    current_point_key = key

    if (point.base_ref !== 'Input') {
      const content = point.content[environment.library.toLowerCase()];

      if (content) {
        const rawparams = content.parameters;
        var parameters = { name:{type:'string', value:current_point_name} }
        Object.keys(rawparams).map(raw_p_key => {
          if (raw_p_key !== 'extras') parameters[raw_p_key] = rawparams[raw_p_key]
        })
        //parameters['name'] = {type:'string', value:current_point_name};
        let inputs = dummies[key].inputs;

        /*
        Need an add-operation first?
        If a node got multiple inputs and is not "Concatenate", those inputs should be added together.
        */ 
        if (inputs.length > 1) {

          if (point.base_ref === 'Concatenate') {

            // Adding inputs
            msg += '\n' + dent(indent, 1) + current_point_name + ' = tf.keras.layers.Concatenate(axis=' + point.concat_dim + ')('
            msg += '\n' + dent(indent, 2) + '['
            inputs.map(inp_key => {
              msg += '\n' + dent(indent, 3) + getPointName(flowpoints, inp_key) + ','
            })
            msg = msg.slice(0, -1)
            msg += '\n' + dent(indent, 2) + ']'
            msg += '\n' + dent(indent, 1) + ')'

          } else {

            if (point.concat_inputs) {

              // Concat name
              var concatname = "'concat_"
              inputs.map(inp_key => concatname += getPointName(flowpoints, inp_key) + '_')
              concatname = concatname.slice(0, -1) + "'"

              // Creating layer
              msg += '\n' + dent(indent, 1) + current_point_name + ' = tf.keras.layers.' + content.reference
              msg += FormatInitParams(parameters, 1, indent)
              msg += '('
              msg += '\n' + dent(indent, 2) + "tf.keras.layers.Concatenate(name=" + concatname + ", axis=" + point.concat_dim + ")(["
              inputs.map(inp_key => {
                msg += '\n' + dent(indent, 3) + getPointName(flowpoints, inp_key) + ','
              })
              msg = msg.slice(0, -1)
              msg += '\n' + dent(indent, 2) + '])'
              msg += '\n' + dent(indent, 1) + ')'

            } else {

              // Adder name
              var addname = "'add_"
              inputs.map(inp_key => addname += getPointName(flowpoints, inp_key) + '_')
              addname = addname.slice(0, -1) + "'"

              // Creating layer
              msg += '\n' + dent(indent, 1) + current_point_name + ' = tf.keras.layers.' + content.reference
              msg += FormatInitParams(parameters, 1, indent)
              msg += '('
              msg += '\n' + dent(indent, 2) + "tf.keras.layers.Add(name=" + addname + ")(["
              inputs.map(inp_key => {
                msg += '\n' + dent(indent, 3) + getPointName(flowpoints, inp_key) + ','
              })
              msg = msg.slice(0, -1)
              msg += '\n' + dent(indent, 2) + '])'
              msg += '\n' + dent(indent, 1) + ')'

            }

          }

        } else {

          // Creating layer
          msg += '\n' + dent(indent, 1) + current_point_name + ' = tf.keras.layers.' + content.reference
          msg += FormatInitParams(parameters, 1, indent)
          msg += '(' + getPointName(flowpoints, inputs[0]) + ')'

        }
        if (point.is_output) outputs.push(current_point_key)
      } else {
        msg += '\n\nCOULD NOT ADD ' + current_point_name + ' (' + point.base_ref + ')!'
        msg += '\nThe layertype is not available in the the currently selected library.\n\n'
      }
    }
  })

  // Fixing outputs?
  if (outputs.length === 0 && current_point_key !== '') outputs.push(current_point_key)

  // Creating model
  msg += '\n\n' + dent(indent, 1) + '# Creating model'
  msg += '\n' + dent(indent, 1) + '_model = tf.keras.models.Model('
  if (inps.length > 0) {
    msg += '\n' + dent(indent, 2) + 'inputs  = [' + FormatParamInputs(dummies, inps) + '],'
  }
  if (outputs.length > 0) {
    msg += '\n' + dent(indent, 2) + 'outputs = [' + FormatParamInputs(dummies, outputs) + ']'
  }
  if (state.settings.modelID !== '' && state.settings.modelID !== null) {
    msg += ','
    msg += '\n' + dent(indent, 2) + "name    = 'flowpoints.io/?p=" + state.settings.modelID + "'"
  }
  msg += '\n' + dent(indent, 1) + ')'

  // Compiling model
  msg += '\n\n' + dent(indent, 1) + '# Compiling model'
  msg += '\n' + dent(indent, 1) + "_model.compile("
  msg += '\n' + dent(indent, 2) + "optimizer = optimizer,"
  msg += '\n' + dent(indent, 2) + "loss      = loss,"
  msg += '\n' + dent(indent, 2) + "metrics   = metrics"
  msg += '\n' + dent(indent, 1) + ')'

  // Returning model
  msg += '\n\n' + dent(indent, 1) + '# Returning model'
  msg += '\n' + dent(indent, 1) + 'return _model'

  // Returning
  return msg

}



export function TensorFlowParser(state, order, inps, states, dummies, indent) {

  var flowpoints = state.flowpoints;

  // Imports
  var msg = TensorFlowImports()

  // Adding all code
  msg += '\n\n\n' + Constructor(state, order, inps, dummies, indent)

  // Returning
  return msg
}