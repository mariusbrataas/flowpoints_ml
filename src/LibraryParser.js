import { MainLibrary } from "./MainLibrary";


// Cascaded simplifying and de-simplifying

// Parameters
function SimplifyParameters(parameters) {
  var new_parameters = [];
  var keys = [];
  Object.keys(parameters).map(p_key => {
    if (p_key !== 'extras') keys.push(p_key)
  })
  keys.sort().map((p_key, idx) => {
    new_parameters.push(parameters[p_key].value)
  })
  new_parameters.push(parameters.extras)
  return new_parameters
}
function DeSimplifyParameters(parameters, recipe) {
  recipe = JSON.parse(JSON.stringify(recipe));
  const recipe_copy = JSON.parse(JSON.stringify(recipe))
  var new_param_lib = {};
  var keys = [];
  Object.keys(recipe).map(p_key => {
    if (p_key !== 'extras') keys.push(p_key)
  })
  keys.map(p_key => new_param_lib[p_key] = null)
  keys.sort().map((p_key, idx) => {
    new_param_lib[p_key] = recipe[p_key];
    new_param_lib[p_key].value = parameters[idx];
    if (new_param_lib[p_key].value === undefined) {
      new_param_lib[p_key].value = recipe_copy[p_key].value
    }
  })
  if (parameters[keys.length]) {
    new_param_lib.extras = parameters[keys.length]
  } else {
    new_param_lib.extras = recipe.extras
  }
  return new_param_lib
}

// Content
function SimplifyContent(content) {
  var new_content = {};
  Object.keys(content).sort().map((library_key, idx) => {
    new_content[library_key] = [
      content[library_key].reference,
      SimplifyParameters(content[library_key].parameters)
    ]
  })
  return new_content
}
function DeSimplifyContent(content, recipe) {
  recipe = JSON.parse(JSON.stringify(recipe));
  var new_content_lib = {}
  Object.keys(recipe).sort().map((library_key, idx) => {
    if (library_key in content) {
      new_content_lib[library_key] = {
        reference: content[library_key][0],
        parameters: DeSimplifyParameters(content[library_key][1], recipe[library_key].parameters)
      }
    } else {
      new_content_lib[library_key] = recipe[library_key]
    }
  })
  return new_content_lib
}

// Outputs
function SimplifyOutputs(outputs) {
  var new_outputs = [];
  Object.keys(outputs).sort().map(key => {
    new_outputs.push(key)
  })
  return new_outputs
}
function DeSimplifyOutputs(outputs) {
  var new_outputs_lib = {};
  outputs.map(key => {
    new_outputs_lib[key] = {}
  })
  return new_outputs_lib
}

// Position
function SimplifyPosition(pos) {
  return [
    pos.x,
    pos.y
  ]
}
function DeSimplifyPosition(pos) {
  return {
    x: pos[0],
    y: pos[1]
  }
}

// Entire flowpoint
function SimplifyFlowpoint(flowpoint) {
  var new_flowpoint = {
    ref: flowpoint.base_ref,
    name: flowpoint.name,
    isout: flowpoint.is_output,
    out: SimplifyOutputs(flowpoint.outputs),
    pos: SimplifyPosition(flowpoint.pos),
    concat_inputs: flowpoint.concat_inputs,
    concat_dim: flowpoint.concat_dim,
    cont: {},
    contig: flowpoint.contiguous,
    re_ndims: flowpoint.reshape_ndims,
    re_dims: flowpoint.reshape_dims
  }
  if (flowpoint.base_ref === 'Input') {
    new_flowpoint.cont = flowpoint.content;
  } else {
    new_flowpoint.cont = SimplifyContent(flowpoint.content)
  }
  return new_flowpoint
}
function DeSimplifyFlowpoint(flowpoint, getEmptyFlowpointContent) {
  const recipe = getEmptyFlowpointContent(flowpoint.ref);
  var new_flowpoint_lib = {
    base_ref: flowpoint.ref,
    name: flowpoint.name,
    is_output: flowpoint.isout,
    outputs: DeSimplifyOutputs(flowpoint.out),
    pos: DeSimplifyPosition(flowpoint.pos),
    concat_inputs: flowpoint.concat_inputs || false,
    concat_dim: flowpoint.concat_dim || 0,
    output_shape: [],
    content: {},
    contiguous: flowpoint.contig || false,
    reshape_ndims: flowpoint.re_ndims || 0,
    reshape_dims: flowpoint.re_dims || []
  };
  if (flowpoint.ref === 'Input') {
    new_flowpoint_lib.content = flowpoint.cont;
  } else {
    new_flowpoint_lib.content = DeSimplifyContent(flowpoint.cont, recipe, flowpoint.ref === 'Softmax')
  }
  return new_flowpoint_lib
}

// Fix tuples
function FixTuples(flowpoints) {
  Object.keys(flowpoints).map(key => {
    var point = flowpoints[key];
    if (point.base_ref !== 'Input') {
      Object.keys(point.content).map(lib_key => {
        var content = flowpoints[key].content[lib_key];
        Object.keys(content.parameters).map(param_key => {
          var param = flowpoints[key].content[lib_key].parameters[param_key];
          if (param.istuple && !Array.isArray(param.value)) {

            if (point.base_ref.toLowerCase().includes('1d')) {
              param.value = [param.value];
            } else if (point.base_ref.toLowerCase().includes('2d')) {
              param.value = [param.value, param.value]
            } else if (point.base_ref.toLowerCase().includes('3d')) {
              param.value = [param.value, param.value, param.value]
            }

            if (!Array.isArray(param.value)) {
              var sample_tuple = null;
              Object.keys(content).map(param_key_2 => {
                const p2 = content[param_key_2];
                if (p2.istuple && Array.isArray(p2.value)) sample_tuple = p2.value.map(val => 1 * val)
              })
            }

            if (!Array.isArray(param.value)) console.log('Could not fix', point)

          }
        })
      })
    }
  })
  return flowpoints
}

// Flowpoints (like plural)
function SimplifyFlowpoints(flowpoints) {
  var new_flowpoints = {};
  Object.keys(flowpoints).sort().map((key, idx) => {
    new_flowpoints[key] = SimplifyFlowpoint(flowpoints[key])
  })
  return new_flowpoints
}
function DeSimplifyFlowpoints(flowpoints, getEmptyFlowpointContent) {
  var new_flowpoints_lib = {};
  Object.keys(flowpoints).sort().map((key, idx) => {
    new_flowpoints_lib[key] = DeSimplifyFlowpoint(flowpoints[key], getEmptyFlowpointContent)
  })
  return FixTuples(new_flowpoints_lib)
}

// Environment
function SimplifyEnvironment(env) {
  return [
    env.library,
    env.notes,
    env.batch_first,
    env.modelname,
    env.include_training,
    env.include_saveload,
    env.include_predict
  ]
}
function DeSimplifyEnvironment(env, main_env) {
  var new_env_lib = JSON.parse(JSON.stringify(main_env));
  new_env_lib.library = '' + env[0]
  new_env_lib.notes = '' + env[1]
  new_env_lib.batch_first = env[2] || false
  new_env_lib.modelname = env[3] || 'NeuralNet'
  new_env_lib.include_training = env[4] || false
  new_env_lib.include_saveload = env[5] || false
  new_env_lib.include_predict = env[6] || false
  new_env_lib.autoparams = main_env.autoparams
  return new_env_lib
}

// Visual
function SimplifyVisual(vis) {
  return [
    vis.darkTheme,
    vis.theme,
    vis.background,
    vis.variant,
    vis.drawerOpen,
    vis.showShape,
    vis.showName,
    vis.snap
  ]
}
function DeSimplifyVisual(vis, main_vis) {
  var new_vis_lib = JSON.parse(JSON.stringify(main_vis));
  new_vis_lib.darkTheme = vis[0]
  new_vis_lib.theme = vis[1]
  new_vis_lib.background = vis[2]
  new_vis_lib.variant = vis[3]
  new_vis_lib.drawerOpen = vis[4]
  new_vis_lib.showShape = vis[5]
  new_vis_lib.showName = vis[6]
  new_vis_lib.snap = vis[7]
  return new_vis_lib
}

// Settings
function SimplifySettings(settings) {
  return [
    settings.modelID,
    settings.count,
    settings.lastPos
  ]
}
function DeSimplifySettings(settings, main_set) {
  var new_set_lib = JSON.parse(JSON.stringify(main_set));
  new_set_lib.modelID = settings[0]
  new_set_lib.count = settings[1]
  new_set_lib.lastPos = settings[2]
  return new_set_lib
}

// Try to parse json
function TryParseJSON(msg) {
  try {
    return JSON.parse(msg)
  } catch(err) {}
  return msg
}

// Full
function Simplify(state) {
  return {
    flowpoints: SimplifyFlowpoints(state.flowpoints),
    environment: SimplifyEnvironment(state.environment),
    visual: SimplifyVisual(state.visual),
    settings: SimplifySettings(state.settings),
  }
}
function DeSimplify(state, getEmptyFlowpointContent, main_lib) {
  state = TryParseJSON(state)
  //var main_lib = MainLibrary();
  return {
    flowpoints: DeSimplifyFlowpoints(state.flowpoints, getEmptyFlowpointContent),
    environment: DeSimplifyEnvironment(state.environment, main_lib.environment),
    visual: DeSimplifyVisual(state.visual, main_lib.visual),
    settings: DeSimplifySettings(state.settings, main_lib.settings),
    notification: main_lib.notification
  }
}


export function Library2String(state) {

  return JSON.stringify(Simplify(state))

}


export function String2Library(msg, getEmptyFlowpointContent, main_lib) {
  
  return DeSimplify(msg, getEmptyFlowpointContent, main_lib)

}