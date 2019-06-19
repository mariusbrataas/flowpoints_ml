import { getPointName, FormatParamInputs, dent } from "./FlowOrder";


function getStateName(key, flowpoints, states, init_states) {
  if (init_states.includes(key)) {
    return 'self.' + getPointName(flowpoints, key) + '_state'
  }
  return states[key]
}


function getOutputTargetName(key, flowpoints) {
  return getPointName(flowpoints, key) + '_target'
}


function PyTorchImports(indent) {
  var msg = '# Importing PyTorch tools'
  msg += '\nimport torch'
  msg += '\nfrom torch import nn, optim, cuda'
  msg += '\n\n\n# Importing other libraries'
  msg += '\nimport numpy as np'
  msg += '\nimport matplotlib.pyplot as plt'
  msg += '\nimport time'
  return msg
}


function Flatten(indent) {
  var msg = '# Helper class for flattening'
  msg += '\nclass Flatten(nn.Module):'
  msg += '\n' + dent(indent, 1) + 'def __init__(self):'
  msg += '\n' + dent(indent, 2) + 'super(Flatten, self).__init__()'
  msg += '\n' + dent(indent, 1) + 'def forward(self, x):'
  msg += '\n' + dent(indent, 2) + 'return x.view(x.shape[0], -1)'
  return msg
}



function Constructor(state, order, indent, dummies, states, init_states, got_hidden_states, library, modelID) {

  var flowpoints = state.flowpoints;
  var environment = state.environment;
  
  // Basics
  var msg = '# Model'
  msg += '\nclass NeuralNet(nn.Module):'
  msg += '\n\n\n' + dent(indent, 1) + 'def __init__(self, optimizer=optim.SGD, alpha=0.001, criterion=nn.CrossEntropyLoss(), use_cuda=None):'
  msg += '\n\n' + dent(indent, 2) + '# Basics'
  msg += '\n' + dent(indent, 2) + 'super(NeuralNet, self).__init__()'
  msg += '\n' + dent(indent, 2) + 'self.name        = ' + (modelID ? ("'" + modelID + "'") : "'model'")
  msg += '\n' + dent(indent, 2) + 'self.batch_first = ' + (environment.batch_first ? 'True' : 'False')
  msg += '\n' + dent(indent, 2) + 'self.__call__    = self.predict'
  msg += '\n\n' + dent(indent, 2) + '# Settings'
  msg += '\n' + dent(indent, 2) + 'self.optim_type = optimizer'
  msg += '\n' + dent(indent, 2) + 'self.optimizer  = None'
  msg += '\n' + dent(indent, 2) + 'self.alpha      = alpha'
  msg += '\n' + dent(indent, 2) + 'self.criterion  = criterion'
  msg += '\n\n' + dent(indent, 2) + '# Use CUDA?'
  msg += '\n' + dent(indent, 2) + 'self.use_cuda = use_cuda if (use_cuda != None and cuda.is_available()) else cuda.is_available()'
  msg += '\n\n' + dent(indent, 2) + '# Current loss and loss history'
  msg += '\n' + dent(indent, 2) + 'self.train_loss = 0'
  msg += '\n' + dent(indent, 2) + 'self.valid_loss = 0'
  msg += '\n' + dent(indent, 2) + 'self.train_loss_hist = []'
  msg += '\n' + dent(indent, 2) + 'self.valid_loss_hist = []'

  // Prep aligning recurrent states
  var max_l = 0;
  order.map(key => {
    let dummy = dummies[key];
    if (dummy.is_recurrent) {
      max_l = Math.max(max_l, getStateName(key, flowpoints, states, init_states).length);
    }
  })

  // Initializing recurrent states
  if (max_l > 0) {
    var state_msg = dent(indent, 2) + '# Initializing states with placeholder tensors';
    order.map(key => {
      let dummy = dummies[key];
      if (dummy.is_recurrent) {

        // Init state
        state_msg += '\n' + dent(indent, 2) + getStateName(key, flowpoints, states, init_states)

        // Aligning equals sign
        for (var i = 0; i < max_l - getStateName(key, flowpoints, states, init_states).length; i++) state_msg += ' '
        state_msg += ' = torch.tensor([0], dtype=torch.float64)'

      }
    })
    msg += '\n\n' + state_msg
  }

  // Initializing all layers
  msg += '\n\n' + dent(indent, 2) + '# Initializing all layers'
  order.map(key => {
    let point = flowpoints[key];
    
    if (point.base_ref !== 'Input') {
      if (flowpoints[key].content[library]) {
        let content = point.content[environment.library.toLowerCase()];
        let parameters = content.parameters

        // Getting longest parameter name
        var max_l = 0;
        Object.keys(parameters).map(p_key => {
          if (p_key !== 'extras') max_l = Math.max(p_key.length, max_l)
        })

        // Init object
        msg += '\n' + dent(indent, 2) + 'self.' + getPointName(flowpoints, key) + ' = ' + (point.base_ref === 'Flatten' ? '' : 'nn.') + content.reference + '(';

        // Adding arguments
        Object.keys(parameters).map(p_key => {

          if (p_key !== 'extras') {

            let param = parameters[p_key];
          
            // Adding argument
            msg += '\n' + dent(indent, 3) + p_key

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
              if (p_key === 'batch_first') {
                msg += 'self.batch_first'
              } else {
                msg += (param.value === true ? 'True' : 'False')
              }
            } else {
              msg += (param.value === '' ? 'None' : param.value)
              //msg += (param.value === '' ? (param.min === Infinity ? 0 : param.min) : param.value)
            }
            msg += ','

          }

        })
        if (max_l > 0) msg += '\n' + dent(indent, 2)
        msg += ')'
      }
    }
  })

  // Startup routines
  msg += '\n\n' + dent(indent, 2) + '# Running startup routines'
  msg += '\n' + dent(indent, 2) + 'self.startup_routines()'
  if (got_hidden_states) msg += '\n' + dent(indent, 2) + 'self.reset_hidden_states()'

  // Returning
  return msg

}


function StartupRoutines(indent) {
  var msg = dent(indent, 1) + 'def startup_routines(self):'
  msg += '\n' + dent(indent, 2) + 'self.optimizer = self.optim_type(self.parameters(), lr=self.alpha)'
  msg += '\n' + dent(indent, 2) + 'if self.use_cuda:'
  msg += '\n' + dent(indent, 3) + 'self.cuda()'
  return msg
}


function ResetHidden(flowpoints, order, inps, states, dummies, indent, init_states, library) {
  var msg = dent(indent, 1) + 'def reset_hidden_states(self, sample=None):'
  msg += '\n' + dent(indent, 2) + 'if type(sample) == torch.Tensor:'
  msg += '\n' + dent(indent, 3) + 'batch_size = sample.shape[0 if self.batch_first else 1]'
  order.map(key => {
    if (flowpoints[key].base_ref !== 'Input') {
      if (flowpoints[key].content[library]) {
        const parameters = flowpoints[key].content[library].parameters;
        if (parameters.extras.gothidden) {
          if (parameters['hidden_size'] && parameters['num_layers']) {
            msg += '\n' + dent(indent, 3) + 'self.' + getPointName(flowpoints, key) + '_hidden = '
            const hidden = '\n' + dent(indent, 4) + 'torch.zeros((' + parameters['num_layers'].value + ', batch_size, ' + parameters['hidden_size'].value + '), device=sample.device)'
            msg += '(' + hidden + ', ' + hidden + '\n' + dent(indent, 3) + ')'
          }
        }
      }
    }
  })
  msg += '\n' + dent(indent, 2) + 'else:'
  order.map(key => {
    if (flowpoints[key].base_ref !== 'Input') {
      if (flowpoints[key].content[library]) {
        if (flowpoints[key].content[library].parameters.extras.gothidden) {
          msg += '\n' + dent(indent, 3) + 'self.' + getPointName(flowpoints, key) + '_hidden = None'
        }
      }
    }
  })
  return msg
}


function Predict(flowpoints, inps, indent, got_hidden_states) {
  var msg = dent(indent, 1) + 'def predict(self, '

  // Adding all inputs
  const formated_inputs = FormatParamInputs(flowpoints, inps);
  msg += formated_inputs + (got_hidden_states ? ', reset_hidden_states=False' : '') + '):'

  // Running preds
  msg += '\n\n' + dent(indent, 2) + '# Switching off autograd'
  msg += '\n' + dent(indent, 2) + 'with torch.no_grad():'
  msg += '\n\n' + dent(indent, 3) + '# Use CUDA?'
  msg += '\n' + dent(indent, 3) + 'if self.use_cuda:'
  inps.map(key => {
    const p_name = getPointName(flowpoints, key);
    msg += '\n' + dent(indent, 4) + p_name + ' = ' + p_name + '.cuda()'
  })
  if (got_hidden_states) {
    msg += '\n\n' + dent(indent, 3) + '# Reset hidden states?'
    msg += '\n' + dent(indent, 3) + 'if reset_hidden_states:'
    msg += '\n' + dent(indent, 4) + 'self.reset_hidden_states(' + getPointName(flowpoints, inps[0]) + ')'
  }
  msg += '\n\n' + dent(indent, 3) + '# Running inference'
  msg += '\n' + dent(indent, 3) + 'return self.forward(' + formated_inputs + ')'

  // Returning
  return msg
}


function Forward(flowpoints, order, inps, states, dummies, indent, init_states, got_hidden_states, library) {
  var msg = dent(indent, 1) + 'def forward(self, '

  // Adding all inputs
  const formated_inputs = FormatParamInputs(dummies, inps);
  msg += formated_inputs + '):'

  // Forwarding
  var step_lib = [];
  var outputs = [];
  order.map(key => {
    const point = dummies[key];
    if (point.base_ref !== 'Input') {
      if (flowpoints[key].content[library]) {
        var step_msg = dent(indent, 2) + getStateName(key, flowpoints, states, init_states)
        if (flowpoints[key].content.pytorch.parameters.extras.gothidden) step_msg += ', self.' + getPointName(flowpoints, key) + '_hidden'
        step_msg += ' = self.' + getPointName(dummies, key) + '(';
        if (point.inputs.length > 1) {
          if (flowpoints[key].concat_inputs) {
            //torch.cat(tensors, dim=0
            step_msg += 'torch.cat(['
            point.inputs.map(inp_key => step_msg += getStateName(inp_key, flowpoints, states, init_states) + ', ')
            step_msg = step_msg.substring(0, step_msg.length -2) + '], dim=' + flowpoints[key].concat_dim + ')'
          } else {
            point.inputs.map(inp_key => step_msg += getStateName(inp_key, flowpoints, states, init_states) + ' + ');
            step_msg = step_msg.substring(0, step_msg.length - 3);
          }
        } else {
          step_msg += getStateName(point.inputs[0], flowpoints, states, init_states)
        }
        if (flowpoints[key].content.pytorch.parameters.extras.gothidden) step_msg += ', tuple([_.data for _ in self.' + getPointName(flowpoints, key) + '_hidden]) if self.' + getPointName(flowpoints, key) + '_hidden else None'
        step_msg += ')'
        step_lib.push({ msg:step_msg, title:point.base_ref });
        if (flowpoints[key].contiguous) {
          var reshape_msg = '';
          reshape_msg += dent(indent, 2) + getStateName(key, flowpoints, states, init_states) + ' = ' + getStateName(key, flowpoints, states, init_states) + '.contiguous()'
          if (flowpoints[key].reshape_ndims > 0) {
            reshape_msg += '.view('
            flowpoints[key].reshape_dims.map(val => {
              reshape_msg += val + ', '
            })
            reshape_msg = reshape_msg.substring(0, reshape_msg.length - 2) + ')'
          }
          step_lib.push({ msg:reshape_msg, title:'Tensor transforms' });
        } else if (flowpoints[key].reshape_ndims > 0) {
          var reshape_msg = '';
          reshape_msg += dent(indent, 2) + getStateName(key, flowpoints, states, init_states) + ' = ' + getStateName(key, flowpoints, states, init_states)
          reshape_msg += '.view('
          flowpoints[key].reshape_dims.map(val => {
            reshape_msg += val + ', '
          })
          reshape_msg = reshape_msg.substring(0, reshape_msg.length - 2) + ')'
          step_lib.push({ msg:reshape_msg, title:'Tensor transforms' });
        }
        if (flowpoints[key].is_output) outputs.push(key);
      }
    }
  })

  // Max length
  var max_l = 0;
  step_lib.map(step => max_l = Math.max(max_l, step.msg.length > 50 ? 0 : step.msg.length))

  // Adding steps
  step_lib.map(step => {
    
    // Adding step
    msg += '\n' + step.msg;

    // Aligning comment with others
    for (var i = 0; i < max_l - step.msg.length; i++) msg += ' ';
    msg += ' # ' + step.title;
  })

  // Adding return
  if (outputs.length === 0) outputs = [order[order.length-1]];
  msg += '\n' + dent(indent, 2) + 'return'
  outputs.map(key => {
    msg += ' ' + getStateName(key, flowpoints, states, init_states) + ','
  })
  msg = msg.slice(0, -1)

  // Returning
  return msg

}


function FitStep(dummies, inps, indent, outs, flowpoints, states, init_states, got_hidden_states, library) {
  const formated_inputs = FormatParamInputs(dummies, inps);
  var msg = dent(indent, 1) + 'def fit_step(self, training_loader'
  if (got_hidden_states) msg += ', reset_hidden_states=False'
  msg += '):'
  msg += '\n\n' + dent(indent, 2) + '# Preparations for fit step'
  if (got_hidden_states) {
    msg += '\n' + dent(indent, 2) + 'self.train_loss = 0        # Resetting training loss'
    msg += '\n' + dent(indent, 2) + 'self.train()               # Switching to autograd'
  } else {
    msg += '\n' + dent(indent, 2) + 'self.train_loss = 0 # Resetting training loss'
    msg += '\n' + dent(indent, 2) + 'self.train()        # Switching to autograd'
  }
  msg += '\n\n' + dent(indent, 2) + '# Looping through data'
  msg += '\n' + dent(indent, 2) + 'for idx, (' + formated_inputs + ','
  outs.map((key, idx) => {
    msg += ' ' + getOutputTargetName(key, flowpoints) + ','
  })
  msg = msg.slice(0, -1)
  msg += ') in enumerate(training_loader):'
  msg += '\n\n' + dent(indent, 3) + '# Use CUDA?'
  msg += '\n' + dent(indent, 3) + 'if self.use_cuda:'
  inps.map(key => {
    const p_name = getPointName(dummies, key);
    msg += '\n' + dent(indent, 4) + '' + p_name + ' = ' + p_name + '.cuda()'
  })
  outs.map(key => {
    const p_name = getOutputTargetName(key, flowpoints);
    msg += '\n' + dent(indent, 4) + '' + p_name + ' = ' + p_name + '.cuda()'
  })
  if (got_hidden_states) {
    msg += '\n\n' + dent(indent, 3) + '# Reset hidden states?'
    msg += '\n' + dent(indent, 3) + 'if reset_hidden_states or idx == 0:'
    msg += '\n' + dent(indent, 4) + 'self.reset_hidden_states(' + getPointName(flowpoints, inps[0]) + ')'
  }
  msg += '\n\n' + dent(indent, 3) + '# Forward pass'
  msg += '\n' + dent(indent, 3) + 'self.forward(' + formated_inputs + ')'

  // Loss
  msg += '\n\n' + dent(indent, 3) + '# Calculating loss'
  outs.map((key, idx) => {
    msg += '\n' + dent(indent, 3) + 'loss ' + (idx === 0 ? '= ' : '+= ')
    msg += 'self.criterion(' + getStateName(key, flowpoints, states, init_states) + ', '
    msg += getOutputTargetName(key, flowpoints)
    msg += ')'
  })
  msg += '\n' + dent(indent, 3) + 'self.train_loss += loss.item() # Adding to epoch loss'
  
  // Backward pass and optimization
  msg += '\n\n' + dent(indent, 3) + '# Backward pass and optimization'
  msg += '\n' + dent(indent, 3) + 'loss.backward()            # Backward pass'
  msg += '\n' + dent(indent, 3) + 'self.optimizer.step()      # Optimizing weights'
  msg += '\n' + dent(indent, 3) + 'self.optimizer.zero_grad() # Clearing gradients'
  msg += '\n\n' + dent(indent, 2) + '# Adding loss to history'
  msg += '\n' + dent(indent, 2) + 'self.train_loss_hist.append(self.train_loss / len(training_loader))'

  // Returning
  return msg

}


function ValidationStep(dummies, inps, indent, outs, flowpoints, states, init_states, got_hidden_states, library) {
  const formated_inputs = FormatParamInputs(dummies, inps);
  var msg = dent(indent, 1) + 'def validation_step(self, validation_loader'
  if (got_hidden_states) msg += ', reset_hidden_states=False'
  msg += '):'
  msg += '\n\n' + dent(indent, 2) + '# Preparations for validation step'
  msg += '\n' + dent(indent, 2) + 'self.valid_loss = 0 # Resetting validation loss'
  msg += '\n\n' + dent(indent, 2) + '# Switching off autograd'
  msg += '\n' + dent(indent, 2) + 'with torch.no_grad():'
  msg += '\n\n' + dent(indent, 3) + '# Looping through data'
  msg += '\n' + dent(indent, 3) + 'for idx, (' + formated_inputs + ','
  outs.map((key, idx) => {
    msg += ' ' + getOutputTargetName(key, flowpoints) + ','
  })
  msg = msg.slice(0, -1)
  msg += ') in enumerate(validation_loader):'
  msg += '\n\n' + dent(indent, 4) + '# Use CUDA?'
  msg += '\n' + dent(indent, 4) + 'if self.use_cuda:'
  inps.map(key => {
    const p_name = getPointName(dummies, key);
    msg += '\n' + dent(indent, 5) + p_name + ' = ' + p_name + '.cuda()'
  })
  outs.map(key => {
    const p_name = getOutputTargetName(key, flowpoints);
    msg += '\n' + dent(indent, 5) + '' + p_name + ' = ' + p_name + '.cuda()'
  })
  if (got_hidden_states) {
    msg += '\n\n' + dent(indent, 4) + '# Reset hidden states?'
    msg += '\n' + dent(indent, 4) + 'if reset_hidden_states or idx == 0:'
    msg += '\n' + dent(indent, 5) + 'self.reset_hidden_states(' + getPointName(flowpoints, inps[0]) + ')'
  }
  msg += '\n\n' + dent(indent, 4) + '# Forward pass'
  msg += '\n' + dent(indent, 4) + 'self.forward(' + formated_inputs + ')'

  // Loss
  msg += '\n\n' + dent(indent, 4) + '# Calculating loss'
  outs.map((key, idx) => {
    msg += '\n' + dent(indent, 4) + 'loss ' + (idx === 0 ? '= ' : '+= ')
    msg += 'self.criterion(' + getStateName(key, flowpoints, states, init_states) + ', '
    msg += getOutputTargetName(key, flowpoints)
    msg += ')'
  })
  msg += '\n' + dent(indent, 4) + 'self.valid_loss += loss.item() # Adding to epoch loss'
  msg += '\n\n' + dent(indent, 3) + '# Adding loss to history'
  msg += '\n' + dent(indent, 3) + 'self.valid_loss_hist.append(self.valid_loss / len(validation_loader))'

  // Returning
  return msg

}


function Fit(indent, got_hidden_states) {
  var msg = dent(indent, 1) + 'def fit(self, training_loader, validation_loader=None, epochs=10, show_progress=True, save_best=False'
  if (got_hidden_states) msg += ', reset_hidden_states=False'
  msg += '):'
  msg += '\n\n' + dent(indent, 2) + '# Helpers'
  msg += '\n' + dent(indent, 2) + 'best_validation = 1e5'
  msg += '\n\n' + dent(indent, 2) + '# Possibly printing progress'
  msg += '\n' + dent(indent, 2) + 'if show_progress:'
  msg += '\n' + dent(indent, 3) + 'epoch_l = max(len(str(epochs)), 2)' //HERE
  msg += '\n' + dent(indent, 3) + "msg = '%sEpoch   Training loss' % ''.rjust(2 * epoch_l - 4, ' ')"
  msg += '\n' + dent(indent, 3) + "msg += ('   Validation loss' if validation_loader else '') + '   Time remaining'"
  msg += '\n' + dent(indent, 3) + 'print(msg)'
  msg += '\n' + dent(indent, 3) + 't = time.time()'
  msg += '\n\n' + dent(indent, 2) + '# Looping through epochs'
  msg += '\n' + dent(indent, 2) + 'for epoch in range(epochs):'
  msg += '\n\n' + dent(indent, 3) + '# Fit step'
  msg += '\n' + dent(indent, 3) + 'self.fit_step(training_loader' + (got_hidden_states ? ', reset_hidden_states=reset_hidden_states' : '') + ')'
  msg += '\n\n' + dent(indent, 3) + '# Validation step'
  msg += '\n' + dent(indent, 3) + 'if validation_loader:'
  msg += '\n' + dent(indent, 4) + 'self.validation_step(validation_loader' + (got_hidden_states ? ', reset_hidden_states=reset_hidden_states' : '') + ')'
  msg += '\n\n' + dent(indent, 3) + '# Possibly printing progress'
  msg += '\n' + dent(indent, 3) + 'if show_progress:'
  msg += '\n' + dent(indent, 4) + 'eta_s = ((time.time() - t) / (epoch + 1)) * (epochs - epoch - 1)'
  msg += '\n' + dent(indent, 4) + "msg = '%s/%s' % (str(epoch + 1).rjust(epoch_l, ' '), str(epochs).ljust(epoch_l, ' '))"
  msg += '\n' + dent(indent, 4) + "msg += ' | %s' % str(round(self.train_loss_hist[-1], 9)).ljust(13, ' ')"
  msg += '\n' + dent(indent, 4) + "if validation_loader: msg += ' | %s' % str(round(self.valid_loss_hist[-1], 9)).ljust(15, ' ')"
  msg += '\n' + dent(indent, 4) + "msg += ' | '"
  msg += '\n' + dent(indent, 4) + "msg += '%sh ' % round(eta_s / 3600) if eta_s > 3600 else ''"
  msg += '\n' + dent(indent, 4) + "msg += '%sm ' % round(eta_s % 3600 / 60) if eta_s > 60 else ''"
  msg += '\n' + dent(indent, 4) + "msg += '%ss ' % round(eta_s % 60)"
  msg += '\n' + dent(indent, 4) + 'print(msg)'
  msg += '\n\n' + dent(indent, 3) + '# Possibly saving model'
  msg += '\n' + dent(indent, 3) + 'if save_best:'
  msg += '\n' + dent(indent, 4) + 'if self.valid_loss_hist[-1] < best_validation:'
  msg += '\n' + dent(indent, 5) + "self.save('best_validation')"
  msg += '\n' + dent(indent, 5) + 'best_validation = self.valid_loss_hist[-1]'
  if (got_hidden_states) {
    msg += '\n\n' + dent(indent, 2) + '# Resetting hidden states'
    msg += '\n' + dent(indent, 2) + 'self.reset_hidden_states()'
  }
  msg += '\n\n' + dent(indent, 2) + '# Switching to eval'
  msg += '\n' + dent(indent, 2) + 'self.eval()'

  // Returning
  return msg
}


function PlotHist(indent) {
  var msg = dent(indent, 1) + 'def plot_hist(self):'
  msg += '\n\n' + dent(indent, 2) + '# Adding plots'
  msg += '\n' + dent(indent, 2) + "plt.plot(self.train_loss_hist, color='blue', label='Training loss')"
  msg += '\n' + dent(indent, 2) + "plt.plot(self.valid_loss_hist, color='springgreen', label='Validation loss')"
  msg += '\n\n' + dent(indent, 2) + '# Axis labels'
  msg += '\n' + dent(indent, 2) + "plt.xlabel('Epoch')"
  msg += '\n' + dent(indent, 2) + "plt.ylabel('Loss')"
  msg += '\n' + dent(indent, 2) + "plt.legend(loc='upper right')"
  msg += '\n\n' + dent(indent, 2) + '# Displaying plot'
  msg += '\n' + dent(indent, 2) + 'plt.show()'

  // Returning
  return msg
}


function SaveLoad(flowpoints, dummies, order, indent, library) {
  var msg = dent(indent, 1) + "def save(self, name=None):"
  msg += '\n' + dent(indent, 2) + "if not name: name = self.name + '_%sepochs' % len(self.train_loss_hist)"
  msg += '\n' + dent(indent, 2) + "if not '.pth' in name: name += '.pth'"
  msg += "\n" + dent(indent, 2) + "torch.save({"
  order.map(key => {
    const point = dummies[key]
    if (point.base_ref !== 'Input') {
      if (flowpoints[key].content[library]) {
        const pointcode = getPointName(dummies, key)
        msg += "\n" + dent(indent, 3) + "'" + pointcode + "': self." + pointcode + ','
      }
    }
  })
  msg += "\n" + dent(indent, 3) + "'name':            self.name,"
  msg += "\n" + dent(indent, 3) + "'train_loss':      self.train_loss,"
  msg += "\n" + dent(indent, 3) + "'valid_loss':      self.valid_loss,"
  msg += "\n" + dent(indent, 3) + "'train_loss_hist': self.train_loss_hist,"
  msg += "\n" + dent(indent, 3) + "'valid_loss_hist': self.valid_loss_hist,"
  msg += "\n" + dent(indent, 3) + "'optim_type':      self.optim_type,"
  msg += "\n" + dent(indent, 3) + "'alpha':           self.alpha,"
  msg += "\n" + dent(indent, 3) + "'criterion':       self.criterion,"
  msg += "\n" + dent(indent, 3) + "'use_cuda':        self.use_cuda"
  msg += "\n" + dent(indent, 2) + "}, name)"
  msg += "\n\n\n" + dent(indent, 1) + "@staticmethod"
  msg += "\n" + dent(indent, 1) + "def load(name='model'):"
  msg += "\n" + dent(indent, 2) + "if not '.pth' in name: name += '.pth'"
  msg += "\n" + dent(indent, 2) + "checkpoint = torch.load(name)"
  msg += "\n" + dent(indent, 2) + "model = NeuralNet("
  msg += "\n" + dent(indent, 3) + "optimizer = checkpoint['optim_type'],"
  msg += "\n" + dent(indent, 3) + "alpha     = checkpoint['alpha'],"
  msg += "\n" + dent(indent, 3) + "criterion = checkpoint['criterion'],"
  msg += "\n" + dent(indent, 3) + "use_cuda  = checkpoint['use_cuda']"
  msg += "\n" + dent(indent, 2) + ")"
  order.map(key => {
    const point = dummies[key]
    if (point.base_ref !== 'Input') {
      if (flowpoints[key].content[library]) {
        const pointcode = getPointName(dummies, key)
        msg += "\n" + dent(indent, 2) + "model." + pointcode + " = checkpoint['" + pointcode + "']"
      }
    }
  })
  msg += "\n" + dent(indent, 2) + "model.name = checkpoint['name']"
  msg += "\n" + dent(indent, 2) + "model.train_loss = checkpoint['train_loss']"
  msg += "\n" + dent(indent, 2) + "model.valid_loss = checkpoint['valid_loss']"
  msg += "\n" + dent(indent, 2) + "model.train_loss_hist = checkpoint['train_loss_hist']"
  msg += "\n" + dent(indent, 2) + "model.valid_loss_hist = checkpoint['valid_loss_hist']"
  msg += "\n" + dent(indent, 2) + "model.startup_routines()"
  msg += "\n" + dent(indent, 2) + "return model"

  // Returning
  return msg
}


function InitModel(indent) {
  var msg = '# Initialize model?'
  msg += "\nif __name__ == '__main__':"
  msg += '\n' + dent(indent, 1) + 'model = NeuralNet('
  msg += '\n' + dent(indent, 2) + 'optimizer = optim.Adam,'
  msg += '\n' + dent(indent, 2) + 'alpha     = 0.001,'
  msg += '\n' + dent(indent, 2) + 'criterion = nn.CrossEntropyLoss(),'
  msg += '\n' + dent(indent, 2) + 'use_cuda  = True'
  msg += '\n' + dent(indent, 1) + ')'
  return msg
}


export function PyTorchParser(state, order, inps, states, dummies, indent, init_states, outs) {

  var flowpoints = state.flowpoints;
  const library = state.environment.library.toLowerCase()

  // Imports
  var msg = PyTorchImports(indent)

  // Add flattening helper class if needed
  var need_flattening = false;
  Object.keys(flowpoints).map(key => {
    if (flowpoints[key].base_ref === 'Flatten') need_flattening = true;
  })
  if (need_flattening) msg += '\n\n\n' + Flatten(indent)

  // Need hidden states?
  var got_hidden_states = false
  order.map(key => {
    if (flowpoints[key].base_ref !== 'Input') {
      if (flowpoints[key].content[library]) {
        if (flowpoints[key].content[library].parameters.extras.gothidden) {
          got_hidden_states = true
        }
      }
    }
  })

  // Adding all code
  msg += '\n\n\n' + Constructor(state, order, indent, dummies, states, init_states, got_hidden_states, library, state.settings.modelID);
  msg += '\n\n\n' + StartupRoutines(indent);
  if (got_hidden_states) msg += '\n\n\n' + ResetHidden(flowpoints, order, inps, states, dummies, indent, init_states, library);
  msg += '\n\n\n' + Predict(flowpoints, inps, indent, got_hidden_states);
  msg += '\n\n\n' + Forward(flowpoints, order, inps, states, dummies, indent, init_states, got_hidden_states, library);
  msg += '\n\n\n' + FitStep(dummies, inps, indent, outs, flowpoints, states, init_states, got_hidden_states, library);
  msg += '\n\n\n' + ValidationStep(dummies, inps, indent, outs, flowpoints, states, init_states, got_hidden_states, library);
  msg += '\n\n\n' + Fit(indent, got_hidden_states);
  msg += '\n\n\n' + PlotHist(indent);
  msg += '\n\n\n' + SaveLoad(flowpoints, dummies, order, indent, library);
  msg += '\n\n\n' + InitModel(indent)

  // Returning
  return msg
}