import React, { Component } from 'react';

import { Flowpoint } from './Flowpoint.js'

export const DrawPoints = (props) => {
  const flowpoints = props.refresh().flowpoints
  const points = Object.keys(flowpoints).map((key, index) => {
    return (
      <Flowpoint
        key={key}
        localState={flowpoints[key]}
        updateLastPos={props.updateLastPos}
        lastPosX={props.lastPosX}
        lastPosY={props.lastPosY}
        state={props.state}
        refresh={props.refresh}
        updateFlowpoints={props.updateFlowpoints}
        updateSettings={props.updateSettings}
        updateView={props.updateView}/>
    )
  })
  return (
    <div style={{width:'100%', height:'100%'}}>
      {
        points
      }
    </div>
  )
}
