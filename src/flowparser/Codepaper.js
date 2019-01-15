import React, { Component } from 'react';

import Paper from '@material-ui/core/Paper';

import SyntaxHighlighter from 'react-syntax-highlighter';
import { github } from 'react-syntax-highlighter/dist/styles/hljs';

import { parseFlowPoints } from './Flowparser.js'

export const Codepaper = (props) => {
  if (props.settings.showPaper) {
    return (
      <div style={{position:'fixed', bottom:75, right:10, width:'35%', zIndex:5}}>
        <Paper elevation={2} style={{background:'#f8f8f8'}}>
          <div style={{maxHeight:'81vh',overflowY:'scroll'}}>
            <SyntaxHighlighter
              language='python'
              style={github}>
              {
                parseFlowPoints(props.flowpoints)
              }
            </SyntaxHighlighter>
          </div>
        </Paper>
      </div>
    )
  }
  return null
}
