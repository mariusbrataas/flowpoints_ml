import React, { Component } from 'react';
import SyntaxHighlighter from 'react-syntax-highlighter';
import { atelierForestDark } from 'react-syntax-highlighter/dist/styles/hljs';
import github from 'react-syntax-highlighter/dist/styles/hljs/github';

github.hljs.background = '#ffffff';

export const CodeTab = props => {
  const codetheme = props.state.visual.darkTheme ? atelierForestDark : github
  return (
    <div style={{fontSize:12}}>

      <SyntaxHighlighter language='python' style={codetheme} showLineNumbers>
      {
        props.state.environment.code
      }
      </SyntaxHighlighter>

    </div>
  )
}