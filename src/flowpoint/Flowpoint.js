import React, { Component } from 'react';

import Draggable from 'react-draggable';

import Card from '@material-ui/core/Card';
import DeleteIcon from '@material-ui/icons/Delete';
import CloseIcon from '@material-ui/icons/Close';
import IconButton from '@material-ui/core/IconButton';
import RightIcon from '@material-ui/icons/KeyboardArrowRight';

import Tooltip from '@material-ui/core/Tooltip';

import { FlowpointSettings } from './FlowpointSettings'

function shapeMsg(shape) {
  var msg = '['
  shape.map(val => {msg += val + ','})
  msg = msg.substring(0, msg.length-1)
  msg += ']'
  return msg
}

const ConnectInput = (props) => {
  const localState = props.localState
  var settings = props.settings
  const refresh = props.refresh
  const updateView = props.updateView
  const currentOutput = settings.currentOutput
  const currentInput = settings.currentInput
  return (
    <div style={{width:'auto', float:'left', pointerEvents:'auto'}}>
      {
        (!localState.flowtype.includes('input')) ? <IconButton
          aria-label="Input"
          style={{'color':'#006ed6', 'boxShadow':'none', 'background':(localState.key === currentInput ? '#bfd7ff' : null)}}
          onClick={() => {
            var flowpoints = refresh().flowpoints
            var myState = flowpoints[localState.key]
            if (currentOutput != null) {
              var point = flowpoints[currentOutput]
              if (point.outputs.includes(myState.key)) {
                point.outputs.splice(point.outputs.indexOf(myState.key), 1)
                myState.inputs.splice(myState.inputs.indexOf(currentOutput), 1)
              } else {
                point.outputs.push(myState.key)
                myState.inputs.push(currentOutput)
              }
              settings.currentOutput = null
              settings.currentInput = null
            } else {
              if (currentInput === myState.key) {
                settings.currentInput = null
              } else {
                settings.currentInput = myState.key
              }
            }
            updateView(flowpoints, settings)
          }}>
          <RightIcon fontSize='small'/>
        </IconButton> : <IconButton
          aria-label="Output"
          disabled
          style={{'color':'#006ed6', 'boxShadow':'none', 'background':null}}>
          <CloseIcon fontSize='small'/>
        </IconButton>
      }
    </div>
  )
}

const ConnectOutput = (props) => {
  const localState = props.localState
  var settings = props.settings
  const refresh = props.refresh
  const updateView = props.updateView
  const currentOutput = settings.currentOutput
  const currentInput = settings.currentInput
  return (
    <div style={{width:'auto', height:'100%', float:'left', pointerEvents:'auto'}}>
      {
        localState.gotOutput ? <IconButton
          aria-label="Input"
          style={{'color':'#006ed6', 'boxShadow':'none', 'background':(localState.key === currentOutput ? '#bfd7ff' : null)}}
          onClick={() => {
            var flowpoints = refresh().flowpoints
            var myState = flowpoints[localState.key]
            if (currentInput != null) {
              var point = flowpoints[currentInput]
              if (point.inputs.includes(myState.key)) {
                point.inputs.splice(point.inputs.indexOf(myState.key), 1)
                myState.outputs.splice(myState.outputs.indexOf(currentInput), 1)
              } else {
                point.inputs.push(myState.key)
                myState.outputs.push(currentInput)
              }
              settings.currentOutput = null
              settings.currentInput = null
            } else {
              if (currentOutput === myState.key) {
                settings.currentOutput = null
              } else {
                settings.currentOutput = myState.key
              }
            }
            updateView(flowpoints, settings)
          }}>
          <RightIcon fontSize='small'/>
        </IconButton> : <IconButton
          aria-label="Output"
          disabled
          style={{'color':'#006ed6', 'boxShadow':'none', 'background':null}}>
          <CloseIcon fontSize='small'/>
        </IconButton>
      }
    </div>
  )
}

export const Flowpoint = (props) => {
  const localState = props.localState
  if (localState) {
    const refresh = props.refresh
    const updateFlowpoints = props.updateFlowpoints
    const updateSettings = props.updateSettings
    const updateView = props.updateView
    const key = localState.key
    const settings = refresh().settings
    const snapX = settings.snapX
    const snapY = settings.snapY
    return (
      <Draggable
        grid={[snapX, snapY]}
        bounds={{left: 0, top: 0}}
        cancel='.card-contents'
        defaultPosition={{x: localState.x - (localState.x % snapX), y: localState.y - (localState.y % snapY)}}
        onDrag={(event, dragElement) => {
          var flowpoints = refresh().flowpoints
          var myState = flowpoints[key]
          myState.x = dragElement.x
          myState.y = dragElement.y
          myState.height = dragElement.node.clientHeight
          props.updateLastPos(myState.x, myState.y)
          updateFlowpoints(flowpoints)
        }}>
        <div style={{'width':localState.width, 'padding':'0px', paddingBottom:'5px', position:'absolute', pointerEvents:'none', zIndex:localState.isOpen ? 1 : 0}}>
          <ConnectInput
            localState={localState}
            settings={settings}
            refresh={refresh}
            updateView={updateView}/>
          <div style={{width:'70%', float:'left', pointerEvents:'auto'}}>
            <Card>
              <div style={{marginLeft:'10px', marginTop:'2px', marginRight:'10px'}}>
                <FlowpointSettings
                  localState={localState}
                  settings={settings}
                  refresh={refresh}
                  updateFlowpoints={updateFlowpoints}/>
              </div>
              <div style={{padding:'2px', float:'left'}}>
                <Tooltip title="Delete">
                  <IconButton
                    aria-label='Delete'
                    onClick={() => {
                      var flowpoints = refresh().flowpoints
                      var myState = flowpoints[localState.key]
                      myState.inputs.map((testkey) => {
                        var point = flowpoints[testkey]
                        point.outputs.splice(point.outputs.indexOf(myState.key), 1)
                      })
                      myState.outputs.map((testkey) => {
                        var point = flowpoints[testkey]
                        point.inputs.splice(point.inputs.indexOf(myState.key), 1)
                      })
                      delete flowpoints[key]
                      if (Object.keys(flowpoints).length === 0) {
                        props.updateLastPos(0, 0)
                      }
                      updateFlowpoints(flowpoints)
                    }}>
                    <DeleteIcon/>
                  </IconButton>
                </Tooltip>{'  '}
                {
                  shapeMsg(localState.output_shape)
                }
              </div>
            </Card>
          </div>
          <ConnectOutput
            localState={localState}
            settings={settings}
            refresh={refresh}
            updateView={updateView}/>
        </div>
      </Draggable>
    )
  }
}
