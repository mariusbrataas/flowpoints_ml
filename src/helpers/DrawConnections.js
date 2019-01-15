import React, { Component } from 'react';

export const DrawConnections = (props) => {
  const flowpoints = props.flowpoints
  var connections = []
  Object.keys(flowpoints).map((key1) => {
    const point1 = flowpoints[key1]
    if (point1) {
      point1.outputs.map((key2) => {
        const point2 = flowpoints[key2]
        const x1 = point1.x + point1.width - 35
        const y1 = point1.y + 22
        const x2 = point2.x + 15
        const y2 = point2.y + 22
        const bezierOffset = 100
        connections.push(
          <path
            d={'M' + x1 + ',' + y1 +
              'C' + (x1+bezierOffset) + ',' + y1 + ' ' + (x2-bezierOffset) + ',' + y2 + ' ' +
              (x2-0.0001) + ',' + (y2-0.0001)}
            fill="none"
            stroke={x1 < x2 ? 'url(#grad1)' : 'url(#grad2)'}
            strokeWidth={3}
          />
        )
      })
    }
  })
  return (
    <svg style={{width:'100%', height:'100%', position:'absolute', overflow:'visible'}}>
      <linearGradient id="grad1" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0" stop-color='#6e00ff' />
        <stop offset="1" stop-color='#00e9ff' />
      </linearGradient>
      <linearGradient id="grad2" x1="1" y1="0" x2="0" y2="0">
        <stop offset="0" stop-color='#6e00ff' />
        <stop offset="1" stop-color='#00e9ff' />
      </linearGradient>
      {
        connections
      }
    </svg>
  )
}
