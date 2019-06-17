import React, { Component } from 'react';
import { Autosuggest, SelectContainer, TextFieldContainer, SwitchContainer, SelectContainerTooltips } from './FrontHelpers';
import { IconButton, TextField, MenuItem, Paper, FormControl, Chip } from '@material-ui/core';
import DeleteIcon from '@material-ui/icons/Delete';

import { Tooltip } from '@material-ui/core';



function fieldChanger(refresh, updateFlowpoints, key, value) {
  var state = refresh();
  var environment = state.environment;
  var flowpoints = state.flowpoints;
  const selected = state.settings.selected;
  flowpoints[selected].content[environment.library.toLowerCase()].parameters[key].value = value;
  updateFlowpoints(flowpoints)
}

function fieldChanger_input(refresh, updateFlowpoints, key, value) {
  var state = refresh()
  var environment = state.environment;
  var flowpoints = state.flowpoints;
  const selected = state.settings.selected;
  flowpoints[selected].content[key].value = value;
  updateFlowpoints(flowpoints)
}

function fieldChanger_base(refresh, updateFlowpoints, key, value) {
  var state = refresh()
  var environment = state.environment;
  var flowpoints = state.flowpoints;
  const selected = state.settings.selected;
  flowpoints[selected][key] = value;
  updateFlowpoints(flowpoints)
}


const TransformsSection = props => {

  // Point
  var point = props.state.flowpoints[props.state.settings.selected]

  // Container for bools
  var bools = [];
  var ints = [];
  var tuples = [];

  // Adding bools
  bools.push(
    <SwitchContainer
      label='contiguous'
      value={point.contiguous}
      onChange={val => {
        fieldChanger_base(
          props.refresh,
          props.updateFlowpoints,
          'contiguous',
          val
        )
      }}/>
  )

  // Adding ints
  ints.push(
    <TextFieldContainer
      label='Reshape dimensions'
      value={point.reshape_ndims}
      type='number'
      variant='outlined'
      margin='dense'
      style={{
        width: 150,
        paddingRight: 5
      }}
      onChange={val => {
        fieldChanger_base(
          props.refresh,
          props.updateFlowpoints,
          'reshape_ndims',
          val
        )
        if (val) {
          var new_dims = [];
          Array.from(Array(parseInt(val)).keys()).map(idx => {
            new_dims.push(point.reshape_dims[idx] ? point.reshape_dims[idx] : 0)
          })
          fieldChanger_base(
            props.refresh,
            props.updateFlowpoints,
            'reshape_dims',
            new_dims
          )
        }
      }}/>
  )

  // Tuples
  Array.from(Array(parseInt(point.reshape_ndims ? point.reshape_ndims : 0)).keys()).map(idx => {
    tuples.push(
      <TextFieldContainer
        label={'reshape dim ' + idx}
        value={point.reshape_dims[idx]}
        type='number'
        variant='outlined'
        margin='dense'
        style={{
          width: 150,
          paddingRight: 5
        }}
        onChange={val => {
          var new_value = point.reshape_dims;
          new_value[idx] = val
          fieldChanger_base(
            props.refresh,
            props.updateFlowpoints,
            'reshape_dims',
            new_value
          )
        }}/>
    )
  })

  // Returning
  return (
    <div>
  
      <h3 style={{marginTop:30}}>Tensor transforms</h3>

      <div style={{paddingTop:0, marginTop:0, marginLeft:-5, paddingBottom:10}}>{ bools }</div>
      <div>{ ints }</div>
      <div>{ tuples }</div>

    </div>
  )

}


const ParametersSection = props => {

  // Point
  var point = props.state.flowpoints[props.state.settings.selected];

  // Creating all fields
  var fields = {
    int: [],
    float: [],
    select: [],
    tuple: [],
    bool: [],
    string: [],
    unknown: []
  }

  if (point.base_ref !== 'Input') {

    var parameters = point.content[props.state.environment.library.toLowerCase()].parameters;

    // Adding fields
    Object.keys(parameters).map(p_key => {

      if (p_key !== 'extras') {

        const parameter = parameters[p_key]
        if (parameter.istuple) {
          fields.tuple.push(
            <div style={{paddingTop:15}}><h5 style={{margin:0}}>{p_key}</h5></div>
          )
          parameter.value.map((value, idx) => {
            fields.tuple.push(
              <TextFieldContainer
                label={p_key + ' ' + idx}
                value={value}
                type='number'
                variant='outlined'
                margin='dense'
                style={{
                  width: 150,
                  paddingRight: 5
                }}
                onChange={val => {
                  var new_value = parameter.value;
                  new_value[idx] = val;
                  fieldChanger(
                    props.refresh,
                    props.updateFlowpoints,
                    p_key,
                    new_value
                  )
                }}/>
            )
          })
        } else {

          switch(parameter.type) {
            case 'int':
              fields.int.push(
                <TextFieldContainer
                  label={p_key}
                  value={parameter.value}
                  type='number'
                  variant='outlined'
                  margin='dense'
                  style={{
                    width: 150,
                    paddingRight: 5
                  }}
                  onChange={val => {
                    fieldChanger(
                      props.refresh,
                      props.updateFlowpoints,
                      p_key,
                      val
                    )
                  }}/>
              )
              break;
            
            case 'float':
              fields.float.push(
                <TextFieldContainer
                  label={p_key}
                  value={parameter.value}
                  type='number'
                  variant='outlined'
                  margin='dense'
                  style={{
                    width: 150,
                    paddingRight: 5
                  }}
                  onChange={val => {
                    fieldChanger(
                      props.refresh,
                      props.updateFlowpoints,
                      p_key,
                      val
                    )
                  }}/>
              )
              break;
            
            case 'string':
            fields.string.push(
              <TextFieldContainer
                label={p_key}
                value={parameter.value}
                type='text'
                variant='outlined'
                margin='dense'
                style={{
                  width: 150,
                  paddingRight: 5
                }}
                onChange={val => {
                  fieldChanger(
                    props.refresh,
                    props.updateFlowpoints,
                    p_key,
                    val
                  )
                }}/>
            )
            break;
            
            case 'bool':
              if (p_key === 'batch_first') {
                fields.bool.push(
                  <SwitchContainer
                    label={p_key}
                    value={props.state.environment.batch_first}
                    onChange={val => {}}/>
                )
              } else {
                fields.bool.push(
                  <SwitchContainer
                    label={p_key}
                    value={parameter.value}
                    onChange={val => {
                      fieldChanger(
                        props.refresh,
                        props.updateFlowpoints,
                        p_key,
                        val
                      )
                    }}/>
                )
              }
            break;
          
          case null:
          fields.unknown.push(
            <TextFieldContainer
              label={p_key}
              value={parameter.value}
              type='text'
              variant='outlined'
              margin='dense'
              style={{
                width: 150,
                paddingRight: 5
              }}
              onChange={val => {
                fieldChanger(
                  props.refresh,
                  props.updateFlowpoints,
                  p_key,
                  val
                )
              }}/>
          )
          break;
            
          case 'select':
          fields.select.push(
            <SelectContainer
              label={p_key}
              value={parameter.value}
              options={parameter.options}
              style={{
                width: 150,
                paddingRight: 5
              }}
              onChange={val => {
                fieldChanger(
                  props.refresh,
                  props.updateFlowpoints,
                  p_key,
                  val
                )
              }}/>
          )
          break;

          }

        }

      }

    })
  } else {
    var parameters = point.content;

    // Adding n_dims field
    fields.int.push(
      <TextFieldContainer
        label='n_dimensions'
        value={parameters.n_dims}
        type='number'
        variant='outlined'
        margin='dense'
        style={{
          width: 150,
          paddingRight: 5
        }}
        onChange={val => {

          val = val === '' ? '' : Math.max(Math.min(Math.round(val), Infinity), 1)

          // Making sure dims got the correct number of parameters
          var dims = {};
          Array.from(Array(val).keys()).map(idx => {
            if (idx in parameters.dimensions) {
              dims[idx] = parameters.dimensions[idx];
            } else {
              dims[idx] = 1;
            }
          })

          // Changing parameters
          var state = props.refresh()
          var flowpoints = state.flowpoints;
          const selected = state.settings.selected;
          flowpoints[selected].content.n_dims = val;
          flowpoints[selected].content.dimensions = dims;

          // Updating state
          props.updateFlowpoints(flowpoints)

        }}/>
    )


    // Adding dimensions field
    fields.tuple.push(
      <div style={{paddingTop:15}}><h5 style={{margin:0}}>Dimensions</h5></div>
    )
    Object.keys(parameters.dimensions).map((p_key, idx) => {
      fields.tuple.push(
        <TextFieldContainer
          label={'dimension ' + p_key}
          value={parameters.dimensions[p_key]}
          type='number'
          variant='outlined'
          margin='dense'
          style={{
            width: 150,
            paddingRight: 5
          }}
          onChange={val => {
            var state = props.refresh();
            var flowpoints = state.flowpoints;
            state.flowpoints[state.settings.selected].content.dimensions[p_key] = val;
            props.updateFlowpoints(flowpoints)
          }}/>
      )
    })
  }


  if ((fields.int.length + fields.float.length + fields.select.length + fields.tuple.length + fields.bool.length ) > 0) {
    return (
      <div>
  
        <h3 style={{marginTop:30}}>Parameters</h3>
  
        <div>{ fields.int }</div>
        <div>{ fields.float }</div>
        <div>{ fields.select }</div>
        <div>{ fields.tuple }</div>
        <div>{ fields.string }</div>
        <div style={{paddingTop:10, marginLeft:-5}}>{ fields.bool }</div>
        {
          fields.unknown.length > 0 ? <div>
            <h5 style={{paddingTop:5, paddingBottom:0}}>Unknown datatypes. Put your own code here.</h5>
            { fields.unknown }
          </div> : null
        }
  
      </div>
    )
  }

  return null

}


const ButtonContainer = props => {
  return (
    <Tooltip title={props.tooltip} placement="right">
      <IconButton
        aria-label={props.tooltip}
        style={props.style}
        onClick={() => {
          if (props.onClick) props.onClick()
        }}>
        {
          props.children
        }
      </IconButton>
    </Tooltip>
  )
}

/*
<SelectContainerTooltips
        label='Layer type'
        value={point.base_ref}
        options={environment.availableLayers}
        style={{width:'100%'}}
        onChange={val => {

          // Loading from state
          var state = props.refresh();
          var settings = state.settings;
          var flowpoints = state.flowpoints;

          // Changing flowpoint layer type
          var point = flowpoints[settings.selected];
          point.content = props.getEmptyFlowpointContent(val)
          point.base_ref = val;

          // Updating state
          props.updateFlowpoints(flowpoints)

        }}/>
*/


const SettingsSection = props => {
  var settings = props.state.settings;
  var flowpoints = props.state.flowpoints;
  var environment = props.state.environment;
  const point = flowpoints[settings.selected];
  return (
    <div>

      <h3 style={{marginTop:0}}>Flowpoint settings</h3>

        <Autosuggest
          value={point.base_ref}
          options={environment.availableLayers}
          onChange={val => {

            // Loading from state
            var state = props.refresh();
            var settings = state.settings;
            var flowpoints = state.flowpoints;
  
            // Changing flowpoint layer type
            var point = flowpoints[settings.selected];
            point.content = props.getEmptyFlowpointContent(val)
            point.base_ref = val;
  
            // Updating state
            props.updateFlowpoints(flowpoints)
  
          }}/>


        <div style={{paddingTop:15}}>

          <TextFieldContainer
            label='Layer name'
            value={point.name}
            style={{width:'80%'}}
            onChange={val => {

              // Loading from state
              var state = props.refresh();
              var flowpoints = state.flowpoints;
              var settings = state.settings;

              // Changing layer name
              flowpoints[settings.selected].name = val;

              // Updating state
              props.updateFlowpoints(flowpoints)

            }}/>

            <ButtonContainer
              tooltip='Delete flowpoint'
              style={{
                marginLeft: 15,
                marginTop: 5
              }}
              onClick={props.deleteSelected}>
              <DeleteIcon/>
            </ButtonContainer>

        </div>


        {
          point.base_ref !== 'Input' ? <div style={{paddingTop:10, marginLeft:-10}}>
              <SwitchContainer
                label='Use as output'
                value={point.is_output}
                onChange={val => {

                  // Loading from state
                  var state = props.refresh();
                  var flowpoints = state.flowpoints;

                  // Changing layer output
                  flowpoints[state.settings.selected].is_output = val;

                  // Updating state
                  props.updateFlowpoints(flowpoints);

                }}/>
              <SwitchContainer
                label='Concatenate inputs'
                value={point.concat_inputs}
                onChange={val => {

                  // Loading from state
                  var state = props.refresh();
                  var flowpoints = state.flowpoints;

                  // Changing layer output
                  flowpoints[state.settings.selected].concat_inputs = val;

                  // Updating state
                  props.updateFlowpoints(flowpoints);

              }}/>
            </div> : null
        }

        {
          point.concat_inputs ? <div style={{paddingTop:15}}>
            <TextFieldContainer
              label='Concat dim'
              value={point.concat_dim}
              type='number'
              variant='outlined'
              margin='dense'
              style={{
                width: 150,
                paddingRight: 5
              }}
              onChange={val => {
                var flowpoints = props.refresh().flowpoints;
                flowpoints[settings.selected].concat_dim = val;
                props.updateFlowpoints(flowpoints)
              }}/>
          </div> : null
        }

    </div>
  )
}



export const FlowpointTab = props => {
  var settings = props.state.settings;

  // Nothing selected? Returning msg to select something
  if (settings.selected === null) {
    return (
      <div style={{display:'table', width:'100%', height:'50px'}}>
        <div style={{display:'table-cell', verticalAlign:'middle', textAlign:'center'}}>
          Select a flowpoint to display it's settings.
        </div>
      </div>
    )
  }

  // Layer type doesnt exist in current library?
  var show_param_section = true;
  if (!(props.state.environment.library in props.state.flowpoints[props.state.settings.selected].content) && props.state.flowpoints[props.state.settings.selected].base_ref !== 'Input') {
    show_param_section = false;
  }

  // Paramaters and settings of selected
  var flowpoints = props.state.flowpoints;
  const point = flowpoints[settings.selected];
  return (
    <div style={{padding:15}}>
      
      <SettingsSection
        state={props.state}
        refresh={props.refresh}
        updateFlowpoints={props.updateFlowpoints}
        getEmptyFlowpointContent={props.getEmptyFlowpointContent}
        deleteSelected={props.deleteSelected}/>
      
      {
        show_param_section ? <ParametersSection
          state={props.state}
          refresh={props.refresh}
          updateFlowpoints={props.updateFlowpoints}/> : <div style={{display:'table', width:'100%', height:'50px', paddingTop:20}}>
            <div style={{display:'table-cell', verticalAlign:'middle', textAlign:'center'}}>
              Layer type is not available in your current library.
            </div>
          </div>
      }

      <TransformsSection
        state={props.state}
        refresh={props.refresh}
        updateFlowpoints={props.updateFlowpoints}/>

    </div>
  )

}