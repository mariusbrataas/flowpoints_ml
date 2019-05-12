import { FlowOrder } from "./FlowOrder";
import { PyTorchParser } from "./PytorchParser";
import { TensorFlowParser } from "./TensorflowParser";


function ReplaceAll(str, search, replacement) {
  var newstr = ''
  str.split(search).map(val => {newstr += val + replacement})
  return newstr.substring(0, newstr.length - replacement.length)
}


const code_parsers = {
  pytorch: PyTorchParser,
  tensorflow: TensorFlowParser
}


function DefaultMessage(state) {
  var msg = "";
  msg += "'''\n";
  msg += "Created using flowpoints.io\n\n";
  if (state.settings.modelID) {
    msg += "Link to model:\n";
    msg += "https://mariusbrataas.github.io/flowpoints_ml/?p=" + state.settings.modelID + "\n\n";
  }
  msg += "LICENSE:\n"
  msg += "https://github.com/mariusbrataas/flowpoints_ml/blob/master/LICENSE"
  if (state.environment.notes !== '') {
    var notes = state.environment.notes;
    notes = ReplaceAll(notes, "'", '')
    msg += '\n\nNOTES:\n' + notes
  }
  msg += "\n'''\n\n\n"
  return msg
}


export function Parser(state) {

  if (Object.keys(state.flowpoints).length > 0) {

    // Order and inputs
    const tmp = FlowOrder(state)
    const flow_order = tmp.order;
    const inps = tmp.inps;
    const states = tmp.states;
    const init_states = tmp.init_states;
    const outs = tmp.outs;
    var dummies = tmp.dummies;

    // Init msg with default text
    var msg = DefaultMessage(state)

    // Parsing
    const lib = state.environment.library.toLowerCase()
    if (lib in code_parsers) {
      msg += code_parsers[lib](state, flow_order, inps, states, dummies, '    ', init_states, outs)
    }

    // Returning
    return {msg, order:flow_order, dummies}

  }

  return {msg:'', order:[], dummies:{}}

}