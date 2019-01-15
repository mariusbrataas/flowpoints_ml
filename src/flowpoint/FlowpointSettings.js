import React from 'react';

import Typography from '@material-ui/core/Typography';
import TextField from '@material-ui/core/TextField';
import FormGroup from '@material-ui/core/FormGroup';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import Switch from '@material-ui/core/Switch';
import InputLabel from '@material-ui/core/InputLabel';
import Select from '@material-ui/core/Select';
import Divider from '@material-ui/core/Divider';
import ExpandMoreIcon from '@material-ui/icons/ExpandMore';
import ExpandLessIcon from '@material-ui/icons/ExpandLess';
import IconButton from '@material-ui/core/IconButton';

const FlowpointParams = (props) => {
  const localState = props.localState
  var params = null
  if (localState.flowtype in localState.layertypes) {
    params = localState.layertypes[localState.flowtype]
  } else {
    params = localState.activationtypes[localState.flowtype]
  }
  if (params) {
    params = params.params
    return (
      <div>
        <div className='card-contents'>
          <form
            noValidate
            autoComplete="off"
            onSubmit={(e) => {e.preventDefault()}}>
            <TextField
              label="Name"
              margin='none'
              style={{marginTop:'10px', width:'90%'}}
              value={localState.name}
              onChange={(e) => {
                var flowpoints = props.refresh().flowpoints
                var myState = flowpoints[localState.key]
                myState.name = e.target.value;
                props.updateFlowpoints(flowpoints)
              }}
            />
          </form>
        </div>
        <div style={{marginTop:'10px'}}>
          <Typography variant="subtitle1" color="textSecondary">
            Parameters
          </Typography>
          {
            Object.keys(params).map(paramkey => {
              if (params[paramkey].type.includes('int')) {
                return (
                  <TextField
                    label={paramkey}
                    value={params[paramkey].current != 0 ? params[paramkey].current : null}
                    onChange={(e) => {
                      var flowpoints = props.refresh().flowpoints
                      var myState = flowpoints[localState.key]
                      if (myState.flowtype in myState.layertypes) {
                        myState.layertypes[myState.flowtype].params[paramkey].current = Math.round(e.target.value)
                      } else {
                        myState.activationtypes[myState.flowtype].params[paramkey].current = Math.round(e.target.value)
                      }
                      props.updateFlowpoints(flowpoints)
                    }}
                    type="number"
                    margin="dense"
                    variant="outlined"
                    style={{width:'100px', paddingRight:'5px'}}/>
                )
              }
            })
          }
          {
            Object.keys(params).map(paramkey => {
              if (params[paramkey].type.includes('double')) {
                return (
                  <TextField
                    label={paramkey}
                    value={params[paramkey].current != 0 ? params[paramkey].current : null}
                    onChange={(e) => {
                      var flowpoints = props.refresh().flowpoints
                      var myState = flowpoints[localState.key]
                      if (myState.flowtype in myState.layertypes) {
                        myState.layertypes[myState.flowtype].params[paramkey].current = e.target.value
                      } else {
                        myState.activationtypes[myState.flowtype].params[paramkey].current = e.target.value
                      }
                      props.updateFlowpoints(flowpoints)
                    }}
                    type="number"
                    margin="dense"
                    variant="outlined"
                    style={{width:'100px', paddingRight:'5px'}}/>
                )
              }
            })
          }
        </div>
        <div>
          {
            Object.keys(params).map(paramkey => {
              if (params[paramkey].type.includes('tuple')) {
                return (
                  <div>
                    {
                      params[paramkey].current.map((val, index) => {
                        return (
                          <TextField
                            label={paramkey + ' ' + index}
                            value={params[paramkey].current[index] != 0 ? params[paramkey].current[index] : null}
                            onChange={(e) => {
                              var flowpoints = props.refresh().flowpoints
                              var myState = flowpoints[localState.key]
                              if (myState.flowtype in myState.layertypes) {
                                myState.layertypes[myState.flowtype].params[paramkey].current[index] = Math.round(e.target.value)
                              } else {
                                myState.activationtypes[myState.flowtype].params[paramkey].current[index] = Math.round(e.target.value)
                              }
                              props.updateFlowpoints(flowpoints)
                            }}
                            type="number"
                            margin="dense"
                            variant="outlined"
                            style={{width:'100px', paddingRight:'5px'}}/>
                        )
                      })
                    }
                  </div>
                )
              }
            })
          }
        </div>
        <div>
          <FormGroup row>
          {
            Object.keys(params).map(paramkey => {
              if (params[paramkey].type.includes('bool')) {
                return (
                  <FormControlLabel
                    control={
                      <Switch
                        checked={params[paramkey].current.includes('True')}
                        onChange={(e) => {
                          var flowpoints = props.refresh().flowpoints
                          var myState = flowpoints[localState.key]
                          if (myState.flowtype in myState.layertypes) {
                            var current = myState.layertypes[myState.flowtype].params[paramkey].current
                            myState.layertypes[myState.flowtype].params[paramkey].current = current.includes('True') ? 'False' : 'True'
                          } else {
                            var current = myState.activationtypes[myState.flowtype].params[paramkey].current
                            myState.activationtypes[myState.flowtype].params[paramkey].current = current.includes('True') ? 'False' : 'True'
                          }
                          props.updateFlowpoints(flowpoints)
                        }}/>
                    }
                    label={paramkey}/>
                )
              }
            })
          }
          </FormGroup>
        </div>
      </div>
    )
  } else {
    if (localState.flowtype.includes('input')) {
      return (
        <div>
          <div className='card-contents'>
            <form
              noValidate
              autoComplete="off"
              onSubmit={(e) => {e.preventDefault()}}>
              <TextField
                label="Name"
                margin='none'
                style={{marginTop:'10px', width:'90%'}}
                value={localState.name}
                onChange={(e) => {
                  var flowpoints = props.refresh().flowpoints
                  var myState = flowpoints[localState.key]
                  myState.name = e.target.value;
                  props.updateFlowpoints(flowpoints)
                }}
              />
            </form>
          </div>
          <div style={{marginTop:'10px'}}>
            <Typography variant="subtitle1" color="textSecondary">
              Parameters
            </Typography>
            <TextField
              label='Dimensions'
              value={localState.input_point_dims != 0 ? localState.input_point_dims : null}
              onChange={(e) => {
                var flowpoints = props.refresh().flowpoints
                var myState = flowpoints[localState.key]
                myState.input_point_dims = Math.min(10, Math.max(0, Math.round(e.target.value)))
                const diff = myState.input_point_dims - myState.output_shape.length
                if (diff > 0) {
                  Array.from(Array(diff).keys()).map((val, index) => {
                    myState.output_shape.push(1)
                  })
                }
                if (diff < 0) {
                  const current = myState.output_shape
                  myState.output_shape = []
                  Array.from(Array(myState.input_point_dims).keys()).map((val, index) => {
                    myState.output_shape.push(current[index])
                  })
                }
                flowpoints[localState.key] = myState
                props.updateFlowpoints(flowpoints)
              }}
              type="number"
              margin="dense"
              variant="outlined"
              style={{width:'100px', paddingRight:'5px'}}/>
            <FormGroup row>
              {
                localState.output_shape.map((val, index) => {
                  return (
                    <TextField
                      label={'dim_' + index}
                      value={val != 0 ? val : null}
                      onChange={(e) => {
                        var flowpoints = props.refresh().flowpoints
                        var myState = flowpoints[localState.key]
                        myState.output_shape[index] = Math.max(0, Math.round(e.target.value))
                        props.updateFlowpoints(flowpoints)
                      }}
                      type="number"
                      margin="dense"
                      variant="outlined"
                      style={{width:'100px', paddingRight:'5px'}}/>
                  )
                })
              }
            </FormGroup>
          </div>
        </div>
      )
    }
    return (
      <Typography variant="subtitle1" color="textSecondary">
        No parameters
      </Typography>
    )
  }
}


export const FlowpointSettings = (props) => {
  const localState = props.localState
  const key = props.localState.key
  const layertypes = props.localState.layertypes
  const activationtypes = props.localState.activationtypes
  return (
    <div className='card-contents'>
      <form
        noValidate
        autoComplete="off"
        onSubmit={(e) => {e.preventDefault()}}>
        <Select
          native
          margin='none'
          style={{marginLeft:'10px', width:'70%'}}
          value={localState.flowtype}
          onChange={(e) => {
            var flowpoints = props.refresh().flowpoints
            var myState = flowpoints[key]
            myState.flowtype = e.target.value
            if (myState.flowtype.includes('input')) {
              myState.inputs.map(inpkey => {
                flowpoints[inpkey].outputs.splice(flowpoints[inpkey].outputs.indexOf(localState.key))
              })
              myState.inputs = []
            }
            props.updateFlowpoints(flowpoints)
          }}>
          <option value='input'>Input</option>
          <Divider />
          {
            Object.keys(layertypes).map(layerkey => {
              return <option value={layerkey}>{layertypes[layerkey].title}</option>
            })
          }
          <Divider />
          {
            Object.keys(activationtypes).map(activationkey => {
              return <option value={activationkey}>{activationtypes[activationkey].title}</option>
            })
          }
        </Select>
        <IconButton
          style={{marginLeft:'10px', top:'5px'}}
          onClick={() => {
            var flowpoints = props.refresh().flowpoints
            var myState = flowpoints[localState.key]
            myState.isOpen ^= true
            props.updateFlowpoints(flowpoints)
          }}>
          {
            localState.isOpen ? <ExpandLessIcon/> : <ExpandMoreIcon/>
          }
        </IconButton>
      </form>
      {
        localState.isOpen ? <div style={{marginLeft:'5px'}}>
          <FlowpointParams
            localState={localState}
            refresh={props.refresh}
            updateFlowpoints={props.updateFlowpoints}/>
        </div> : null
      }
    </div>
  )
}
