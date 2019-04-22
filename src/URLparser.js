import React, { Component } from 'react';
import { PyTorchModules } from './libraries/pytorch.js';

/*
const lvls = {
  0: '%24',
  1: '#',
  2: '%2C',
  3: '%'
}*/
/*
const lvls = {
  0: 'MAIS',
  1: 'FISK',
  2: 'OST',
  3: 'KORN'
}*/

export const themes = [
  'red',
  'pink',,
  'purple',
  'deep-purple',
  'indigo',
  'blue',
  'light-blue',
  'green',
  'light-green',
  'lime',
  'yellow',
  'amber',
  'orange',
  'deep-orange',
  'brown',
  'grey',
  'blue-grey',
  'black',
  'white'
]

// mais+og-fisk_og.ost,og(korn)

const lvls = {
  0: '$',
  1: '(',
  2: ')',
  3: '-'
}
/*
const lvls = {
  0: '.',
  1: ',',
  2: '-',
  3: '_'
}
*/
function num2string(num) {
  return num.toString(36)
}

function string2num(str) {
  return parseInt(str, 36)
}

function encodeParams(params) {
  var msg = ''
  Object.keys(params).sort().map(pkey => {
    const param = params[pkey]
    if (param.type === 'bool') {
      msg += (param.current === 'True' ? 't' : 'f')
    } else if (param.type === 'tuple') {
      param.current.map(pval => {
        msg += num2string(pval) + lvls[3]
      })
      msg = msg.substring(0, msg.length - lvls[3].length)
    } else if (param.type === 'select') {
      msg += param.options.indexOf(param.current)
    } else if (param.type === 'int') {
      msg += num2string(param.current)
    } else {
      msg += param.current
    }
    msg += lvls[2]
  })
  return msg.substring(0, msg.length - lvls[2].length)
}
//https://mariusbrataas.github.io/flowpoints_ml/?p=loadPDF_3_41e2f4_4164f4_0##5#5#0#1aa%2C2aa%2#t%2C1%2C1%2_1##p#5#0##t%2C1%2C1%2_2##p#c#0##t%2C1%2C1%2
//https://mariusbrataas.github.io/flowpoints_ml/?p=loadPDF_3_41e2f4_4164f4_0##5#5#0#1aa%2C2aa%2#t%2C1%2C1%2_1##p#5#0##t%2C1%2C1%2_2##p#c#0##t%2C1%2C1%2

function decodeParams(msg, specs) {
  var params = specs.params;
  const queries = msg.split(lvls[2])
  Object.keys(params).sort().map((pkey, idx) => {
    var param = params[pkey]
    const q = queries[idx]
    if (param.type === 'bool') {
      param.current = (q === 't' ? 'True' : 'False')
    } else if (param.type === 'tuple') {
      q.split(lvls[3]).map((pval, pidx) => {
        param.current[pidx] = string2num(pval)
      })
    } else if (param.type === 'select') {
      param.current = param.options[q]
    } else if (param.type === 'int') {
      param.current = string2num(q)
    } else {
      param.current = q
    }
  })
  specs.params = params
  return specs
}

function encodeOutputs(outputs) {
  var msg = ''
  Object.keys(outputs).map(out_key => {
    msg += num2string(out_key) + outputs[out_key].output[0] + outputs[out_key].input[0] + lvls[2]
  })
  if (Object.keys(outputs).length > 0) {
    msg = msg.substring(0, msg.length - lvls[2].length)
  }
  return msg
}

function decodeOutputs(msg) {
  const pos2title = {
    a: 'auto',
    t: 'top',
    l: 'left',
    r: 'right',
    b: 'bottom'
  }
  var queries = msg.split(lvls[2])
  var outputs = {}
  queries.map(query => {
    if (query !== '') {
      outputs[query.substring(0, query.length - 2)] = {
        output: pos2title[query[query.length - 2]],
        input: pos2title[query[query.length - 1]]
      }
    }
  })
  return outputs
}

function encodePoint(p, key) {
  /*
  0: key
  1: name
  2: x
  3: y
  4: id
  5: outputs
  6: params
  */
  var msg = '' + key
  msg += lvls[1] + p.name
  msg += lvls[1] + num2string(Math.round(p.pos.x/10))
  msg += lvls[1] + num2string(Math.round(p.pos.y/10))
  msg += lvls[1] + (p.specs.title === 'Input' ? '' : num2string(p.specs.id))
  msg += lvls[1] + encodeOutputs(p.outputs)
  msg += lvls[1] + encodeParams(p.specs.params)
  return msg
}

function decodePoint(msg, id2key, getModules) {
  const queries = msg.split(lvls[1])

  // Decoding some basics
  var p = {
    name: queries[1],
    output_shape: [],
    pos: {
      x: string2num(queries[2]) * 10,
      y: string2num(queries[3]) * 10,
    },
    specs: decodeParams(queries[6], getModules()[queries[4] === '' ? 'input' : id2key[string2num(queries[4])]]),
    outputs: decodeOutputs(queries[5]),
    isHover: false
  }
  return { key:queries[0], p:p }
}

export function parseToQuery(flowpoints, theme, variant, background, count, darktheme, showNames, library, showSidebar) {
  const variant2key = {
    'paper': 0,
    'outlined': 1,
    'filled':2
  }
  // LibraryTheme_Count_Incolor_Outcolor_point_point_
  var msg = '' + library[0]
  msg += num2string(themes.indexOf(theme))
  msg += variant2key[variant]
  msg += num2string(themes.indexOf(background))
  msg += (darktheme ? 'D' : 'L')
  msg += (showNames ? 'T' : 'F')
  msg += (showSidebar ? 'T' : 'F')
  msg += lvls[0] + num2string(count)
  Object.keys(flowpoints).map(key => {
    msg += lvls[0] + encodePoint(flowpoints[key], key)
  })
  return msg
}


export function parseFromQuery(query) {
  query = query.replace(/%20/g, ' ')
  const queries = query.split(lvls[0])

  // Helpers
  const key2lib = {
    P: {t:'PyTorch', modules:PyTorchModules}
  }
  const key2variant = {
    0: 'paper',
    1: 'outlined',
    2: 'filled'
  }

  // Creating lib
  var lib = {
    library: key2lib[query[0]].t,
    getModules: key2lib[query[0]].modules,
    darktheme: (query[4] === 'D' ? true : false),
    showName: (query[5] === 'T' ? true : false),
    showSidebar: (query[6] === 'T' ? true : false),
    count: string2num(queries[0]),
    theme: themes[string2num(query[1])],
    variant: key2variant[query[2]],
    background: themes[string2num(query[3])]
  }

  // Id to module key
  var id2key = {}
  var modules = lib.getModules()
  Object.keys(modules).map(mod_key => {
    id2key[modules[mod_key].id] = mod_key
  })

  // Parsing all points
  var flowpoints = {}
  queries.splice(2, queries.length).map(point_query => {
    const newpoint = decodePoint(point_query, id2key, lib.getModules);
    flowpoints[newpoint.key] = newpoint.p
  })
  lib['flowpoints'] = flowpoints;

  // Fixing inputs n_dims
  Object.keys(flowpoints).map(key => {
    if (flowpoints[key].specs.title === 'Input') {
      flowpoints[key].specs.params.n_dims.current = flowpoints[key].specs.params.dimensions.current.length;
    }
  })

  // Returning
  return lib

}
